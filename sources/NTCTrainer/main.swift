// MIT License
//
// Copyright (c) 2026 Kyriakos Gavras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import NTCCore
import NTCAssets
import NTCShared
import Metal
import Foundation
import QuartzCore
import AppKit

// MARK: Input selection

// Set INPUT_OVERRIDE to compress a fixed path and skip the open panel
//   "/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG"
//   "/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCRenderer/assets/models/flighthelmet/scene.gltf"
let INPUT_OVERRIDE: String? = ""

let inputURL: URL
if let path = INPUT_OVERRIDE, !path.isEmpty {
    inputURL = URL(fileURLWithPath: path)
} else {
    inputURL = pickInput()
}

let QUALITY = pickQuality()

let ctx      = MetalContext(bundle: NTCCoreResources.bundle)
let trainPso = ctx.makeComputePipelineState(function: "grid_mlp_train")
let adamPso  = ctx.makeComputePipelineState(function: "adam_step")

@MainActor
func trainModel(_ model: Manifest.Model, dir: URL) throws {
    let textureSet = TextureSet(model: model, dir: dir)

    let materialImages: [LoadedImage]
    let SRC_W: Int
    let SRC_H: Int
    (materialImages, SRC_W, SRC_H) = try textureSet.loadImages()

    guard let PRESET = pyramidPreset(srcW: SRC_W, quality: QUALITY) else {
        throw TrainerError.msg("no mip map for \(SRC_W)x\(SRC_H); add one to MIP_MAPS")
    }
    let PYRAMID_SIZES: [Int] = PRESET.sizes
    precondition(PYRAMID_SIZES.count == K_GRIDS)
    precondition(PRESET.mipMap.count == MAX_LODS)

    print("quality \(QUALITY.rawValue) (gridScale \(QUALITY.gridScale)) -> grids \(PYRAMID_SIZES)")

    let PYRAMID_SLOT_FLOATS: [Int] = PYRAMID_SIZES.map { $0 * $0 * F_PER_GRID }

    let N_MATERIAL_SLICES = textureSet.slots.count
    let K_OUT             = textureSet.kOut
    precondition(K_OUT <= K_OUT_MAX, "K_OUT (\(K_OUT)) exceeds K_OUT_MAX (\(K_OUT_MAX))")

    let QUANT: (q: Float, lo: Float, hi: Float) = fake_quant(bits: BITS)

    var PYRAMID_OFFSETS: [Int] = Array(repeating: 0, count: K_GRIDS + 1)
    for i in 0..<K_GRIDS {
        PYRAMID_OFFSETS[i + 1] = PYRAMID_OFFSETS[i] + PYRAMID_SLOT_FLOATS[i]
    }

    let OFFSET_MLP  = PYRAMID_OFFSETS[K_GRIDS]
    let OFFSET_W1   = OFFSET_MLP
    let OFFSET_B1   = OFFSET_W1 + F_IN * K_HIDDEN
    let OFFSET_W2   = OFFSET_B1 + K_HIDDEN
    let OFFSET_B2   = OFFSET_W2 + K_HIDDEN * K_HIDDEN
    let OFFSET_W3   = OFFSET_B2 + K_HIDDEN
    let OFFSET_B3   = OFFSET_W3 + K_HIDDEN * K_OUT_MAX
    let TOTAL       = OFFSET_B3 + K_OUT_MAX

    let mipCount = Int(log2(Double(SRC_W))) + 1
    precondition(mipCount <= MAX_LODS, "mipCount \(mipCount) exceeds MAX_LODS (\(MAX_LODS))")

    let NM_FOR_LOD: [UInt32] = PRESET.mipMap

    let POS_SCALE = Float(SRC_W) / 8.0

    let paramsBuffer = ctx.device.makeBuffer(length: TOTAL * 2 * MemoryLayout<Float>.stride,
                                             options: .storageModeShared)!
    paramsBuffer.label = "NTC.params"
    let paramsFloats = paramsBuffer.contents().bindMemory(to: Float.self, capacity: TOTAL * 2)

    let samplesBuffer = ctx.device.makeBuffer(length: K_BATCH * SAMPLE_STRIDE * MemoryLayout<Float>.stride,
                                              options: .storageModeShared)!
    samplesBuffer.label = "NTC.trainSamples"
    let samplesFloats = samplesBuffer.contents().bindMemory(to: Float.self, capacity: K_BATCH * SAMPLE_STRIDE)


    let channelImportanceBuffer = ctx.device.makeBuffer(length: K_OUT_MAX * MemoryLayout<Float>.stride,
                                                        options: .storageModeShared)!
    channelImportanceBuffer.label = "NTC.channelImportance"
    let channelImportance = channelImportanceBuffer.contents().bindMemory(to: Float.self, capacity: K_OUT_MAX)
    for i in 0..<K_OUT_MAX {
        channelImportance[i] = 1.0
    }

    var rawWeightSum: Float = 0
    for slot in textureSet.slots {
        rawWeightSum += (IMPORTANCE_WEIGHTS[slot.semantic] ?? 1.0) * Float(slot.channels)
    }
    let importanceNorm = rawWeightSum > 0 ? Float(K_OUT) / rawWeightSum : 1.0

    for slot in textureSet.slots {
        let w = (IMPORTANCE_WEIGHTS[slot.semantic] ?? 1.0) * importanceNorm
        for c in 0..<slot.channels {
            channelImportance[slot.channelOffset + c] = w
        }
    }
    
    let sourceTexDesc = MTLTextureDescriptor()
    sourceTexDesc.textureType     = .type2DArray
    sourceTexDesc.pixelFormat     = .rgba32Float
    sourceTexDesc.width           = SRC_W
    sourceTexDesc.height          = SRC_H
    sourceTexDesc.arrayLength     = N_MATERIAL_SLICES
    sourceTexDesc.mipmapLevelCount = 1
    sourceTexDesc.usage           = [.shaderRead]
    sourceTexDesc.storageMode     = .shared
    let sourceTexture = ctx.device.makeTexture(descriptor: sourceTexDesc)!
    sourceTexture.label = "NTC.sourceMaterials"

    func packRGBA(_ img: LoadedImage) -> [SIMD4<Float>] {
        precondition(img.channels == 1 || img.channels == 3)
        var arr = [SIMD4<Float>](repeating: .zero, count: SRC_H * SRC_W)
        if img.channels == 3 {
            for i in 0..<(SRC_H * SRC_W) {
                arr[i] = SIMD4<Float>(img.pixels[i * 3 + 0],
                                      img.pixels[i * 3 + 1],
                                      img.pixels[i * 3 + 2],
                                      0)
            }
        } else {
            for i in 0..<(SRC_H * SRC_W) {
                arr[i] = SIMD4<Float>(img.pixels[i], 0, 0, 0)
            }
        }
        return arr
    }

    let bytesPerRow   = SRC_W * MemoryLayout<SIMD4<Float>>.stride
    let bytesPerImage = SRC_H * bytesPerRow

    func uploadSlice(_ img: LoadedImage, into texture: any MTLTexture, slice: Int) {
        let packed = packRGBA(img)
        packed.withUnsafeBufferPointer { buf in
            texture.replace(region: MTLRegionMake2D(0, 0, SRC_W, SRC_H),
                            mipmapLevel: 0,
                            slice: slice,
                            withBytes: buf.baseAddress!,
                            bytesPerRow: bytesPerRow,
                            bytesPerImage: bytesPerImage)
        }
    }
    for (slot, img) in zip(textureSet.slots, materialImages) {
        uploadSlice(img, into: sourceTexture, slice: slot.sliceIndex)
    }

    let pyramidBuilder = MipPyramidBuilder(ctx: ctx, srcW: SRC_W, srcH: SRC_H, sourceTexture: sourceTexture)

    let mBuffer = ctx.device.makeBuffer(length: TOTAL * MemoryLayout<Float>.stride,
                                        options: .storageModeShared)!
    mBuffer.label = "Adam.m"
    let vBuffer = ctx.device.makeBuffer(length: TOTAL * MemoryLayout<Float>.stride,
                                        options: .storageModeShared)!
    vBuffer.label = "Adam.v"
    memset(mBuffer.contents(), 0, TOTAL * MemoryLayout<Float>.stride)
    memset(vBuffer.contents(), 0, TOTAL * MemoryLayout<Float>.stride)

    let adamConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<AdamConstants>.stride,
                                                 options: .storageModeShared)!
    adamConstsBuffer.label = "Adam.consts"
    let adamConstsPtr = adamConstsBuffer.contents().bindMemory(to: AdamConstants.self, capacity: 1)

    let stepConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<StepConstants>.stride,
                                                 options: .storageModeShared)!
    stepConstsBuffer.label = "NTC.stepConsts"
    let stepConstsPtr = stepConstsBuffer.contents().bindMemory(to: StepConstants.self, capacity: 1)
    stepConstsPtr.pointee = StepConstants(
        kBatch:             UInt32(K_BATCH),
        pyramidOffsets:     (UInt32(PYRAMID_OFFSETS[0]), UInt32(PYRAMID_OFFSETS[1]),
                             UInt32(PYRAMID_OFFSETS[2]), UInt32(PYRAMID_OFFSETS[3]),
                             UInt32(PYRAMID_OFFSETS[4]), UInt32(PYRAMID_OFFSETS[5]),
                             UInt32(PYRAMID_OFFSETS[6]), UInt32(PYRAMID_OFFSETS[7]),
                             UInt32(PYRAMID_OFFSETS[8])),
        pyramidSizes:       (UInt32(PYRAMID_SIZES[0]),   UInt32(PYRAMID_SIZES[1]),
                             UInt32(PYRAMID_SIZES[2]),   UInt32(PYRAMID_SIZES[3]),
                             UInt32(PYRAMID_SIZES[4]),   UInt32(PYRAMID_SIZES[5]),
                             UInt32(PYRAMID_SIZES[6]),   UInt32(PYRAMID_SIZES[7])),
        offsetW1:           UInt32(OFFSET_W1),
        offsetB1:           UInt32(OFFSET_B1),
        offsetW2:           UInt32(OFFSET_W2),
        offsetB2:           UInt32(OFFSET_B2),
        offsetW3:           UInt32(OFFSET_W3),
        offsetB3:           UInt32(OFFSET_B3),
        total:              UInt32(TOTAL),
        bits:               BITS,
        q:                  QUANT.q,
        lo:                 QUANT.lo,
        hi:                 QUANT.hi,
        adamOffset:         0,
        inferLod:           0,
        neuralMipsForLod:   (NM_FOR_LOD[0],  NM_FOR_LOD[1],  NM_FOR_LOD[2],  NM_FOR_LOD[3],
                             NM_FOR_LOD[4],  NM_FOR_LOD[5],  NM_FOR_LOD[6],  NM_FOR_LOD[7],
                             NM_FOR_LOD[8],  NM_FOR_LOD[9],  NM_FOR_LOD[10], NM_FOR_LOD[11],
                             NM_FOR_LOD[12]),
        kOut:                UInt32(K_OUT),
        nSlices:             UInt32(N_MATERIAL_SLICES),
        sliceChannels:       (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
        sliceChannelOffsets: (0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0),
        srcW:                UInt32(SRC_W),
        srcH:                UInt32(SRC_H),
        mipCount:            UInt32(mipCount),
        posScale:            POS_SCALE
    )

    withUnsafeMutablePointer(to: &stepConstsPtr.pointee.sliceChannels) { tup in
        tup.withMemoryRebound(to: UInt32.self, capacity: K_OUT_MAX) { p in
            for (i, slot) in textureSet.slots.enumerated() {
                p[i] = UInt32(slot.channels)
            }
        }
    }
    withUnsafeMutablePointer(to: &stepConstsPtr.pointee.sliceChannelOffsets) { tup in
        tup.withMemoryRebound(to: UInt32.self, capacity: K_OUT_MAX) { p in
            for (i, slot) in textureSet.slots.enumerated() {
                p[i] = UInt32(slot.channelOffset)
            }
        }
    }

    #if NTC_DEBUG
    printPyramidLayout(sizes: PYRAMID_SIZES, offsets: PYRAMID_OFFSETS, slotFloats: PYRAMID_SLOT_FLOATS)
    #endif

    let setDesc = MTLResidencySetDescriptor()
    setDesc.label = "grid_mlp_train.residency"
    setDesc.initialCapacity = 12
    let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
    residencySet.addAllocation(paramsBuffer)
    residencySet.addAllocation(samplesBuffer)
    residencySet.addAllocation(stepConstsBuffer)
    residencySet.addAllocation(channelImportanceBuffer)
    residencySet.addAllocation(mBuffer)
    residencySet.addAllocation(vBuffer)
    residencySet.addAllocation(adamConstsBuffer)
    for alloc in pyramidBuilder.residencyAllocations {
        residencySet.addAllocation(alloc)
    }
    residencySet.commit()
    ctx.queue.addResidencySet(residencySet)
    defer { ctx.queue.removeResidencySet(residencySet) }

    let trainArgTable = ctx.makeArgumentTable(buffers: 4, textures: 1)
    trainArgTable.setAddress(paramsBuffer.gpuAddress,            index: 0)
    trainArgTable.setAddress(samplesBuffer.gpuAddress,           index: 1)
    trainArgTable.setAddress(stepConstsBuffer.gpuAddress,        index: 2)
    trainArgTable.setAddress(channelImportanceBuffer.gpuAddress, index: 3)
    trainArgTable.setTexture(pyramidBuilder.pyramidTexture.gpuResourceID, index: 0)

    let adamArgTable = ctx.makeArgumentTable(buffers: 5)
    adamArgTable.setAddress(paramsBuffer.gpuAddress,     index: 0)
    adamArgTable.setAddress(mBuffer.gpuAddress,          index: 1)
    adamArgTable.setAddress(vBuffer.gpuAddress,          index: 2)
    adamArgTable.setAddress(adamConstsBuffer.gpuAddress, index: 3)
    adamArgTable.setAddress(stepConstsBuffer.gpuAddress, index: 4)

    // https://en.wikipedia.org/wiki/Continuous_uniform_distribution
    // Kaiming He uniform. Float.random() is uniform
    // target Var(X) = 2/in_dim. Uniform(-a, a) has variance a^2/3
    // (b - a)^2 / 12 where a and b are interval endpoints. Our a is the half-width:
    // our lower endpoint is -a and upper is +a
    // width = upper - lower = a - (-a) = 2a
    // width^2 = (2a)^2 = 4a^2
    // variance = 4a^2 / 12 = a^2/3

    // a^2/3 = 2 / in_dim -> uniform variance = target variance
    // a^2 = 6 / in_dim
    // so a = sqrt(6/in_dim)
    let w1Bound = sqrtf(6.0 / Float(F_IN_RAW))
    let w2Bound = sqrtf(6.0 / Float(K_HIDDEN))

    for i in 0..<K_GRIDS {
        let lo = PYRAMID_OFFSETS[i]
        let hi = PYRAMID_OFFSETS[i + 1]
        for j in lo..<hi { paramsFloats[j] = Float.random(in: -0.05...0.05) }
    }
    for i in OFFSET_W1..<OFFSET_B1  { paramsFloats[i] = Float.random(in: -w1Bound...w1Bound) }
    for i in OFFSET_B1..<OFFSET_W2  { paramsFloats[i] = 0 }
    for i in OFFSET_W2..<OFFSET_B2  { paramsFloats[i] = Float.random(in: -w2Bound...w2Bound) }
    for i in OFFSET_B2..<OFFSET_W3  { paramsFloats[i] = 0 }
    for i in OFFSET_W3..<OFFSET_B3  { paramsFloats[i] = Float.random(in: -w2Bound...w2Bound) }
    for i in OFFSET_B3..<TOTAL      { paramsFloats[i] = 0 }
    for i in TOTAL..<(TOTAL * 2)    { paramsFloats[i] = 0 }

    let event = ctx.device.makeSharedEvent()!
    var signalValue: UInt64 = 0

    func fillBatch(lod: Int) {
        let wL = SRC_W >> lod
        let hL = SRC_H >> lod
        for s in 0..<K_BATCH {
            let base = s * SAMPLE_STRIDE
            samplesFloats[base + SAMPLE_X]   = Float(Int.random(in: 0..<wL))
            samplesFloats[base + SAMPLE_Y]   = Float(Int.random(in: 0..<hL))
            samplesFloats[base + SAMPLE_LOD] = Float(lod)
        }
    }

    func encodeStep(adamSlots: Int) {
        let tgSize = 256

        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)

        let trainEnc = cmd.makeComputeCommandEncoder()!
        trainEnc.setComputePipelineState(trainPso)
        trainEnc.setArgumentTable(trainArgTable)
        let trainTgx = (K_BATCH + tgSize - 1) / tgSize
        trainEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: trainTgx, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: tgSize,   height: 1, depth: 1))
        trainEnc.endEncoding()

        let adamEnc = cmd.makeComputeCommandEncoder()!
        adamEnc.setComputePipelineState(adamPso)
        adamEnc.setArgumentTable(adamArgTable)
        let adamTgx = (adamSlots + tgSize - 1) / tgSize
        adamEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: adamTgx, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize,  height: 1, depth: 1))
        adamEnc.endEncoding()

        cmd.endCommandBuffer()
        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
    }

    #if NTC_DEBUG
    func logMeanLoss(_ label: String, step: Int) {
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)
        var sumLoss: Float = 0
        for s in 0..<K_BATCH {
            sumLoss += samplesFloats[s * SAMPLE_STRIDE + SAMPLE_LOSS]
        }
        print("\(label) step \(step)\tmean loss = \(sumLoss / Float(K_BATCH))")
    }
    #endif

    let pyramidCmd = ctx.device.makeCommandBuffer()!
    pyramidCmd.beginCommandBuffer(allocator: ctx.allocator)
    pyramidBuilder.encode(into: pyramidCmd)
    pyramidCmd.endCommandBuffer()
    ctx.queue.commit([pyramidCmd])
    signalValue += 1
    ctx.queue.signalEvent(event, value: signalValue)

    var t: UInt32 = 0
    let lodMax = pyramidBuilder.mipCount - 2

    let tTrainStart = CACurrentMediaTime()

    for step in 0..<nSteps {
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        fillBatch(lod: sampleBatchLod(lodMax: lodMax))

        t += 1
        let bc1    = 1.0 - powf(BETA1, Float(t))
        let bc2    = 1.0 - powf(BETA2, Float(t))
        let lrGrid = cosineLr(step: step, total: nSteps, lrMax: LR_GRID_MAX)
        let lrMlp  = cosineLr(step: step, total: nSteps, lrMax: LR_MLP_MAX)
        adamConstsPtr.pointee = AdamConstants(lrGrid: lrGrid, lrMlp: lrMlp, bc1: bc1, bc2: bc2)

        encodeStep(adamSlots: TOTAL)

        #if NTC_DEBUG
        if step % logEvery == 0 {
            logMeanLoss("train", step: step)
        }
        #endif
    }

    event.wait(untilSignaledValue: signalValue, timeoutMS: 5000)

    let tTrainEnd = CACurrentMediaTime()
    let trainDT = tTrainEnd - tTrainStart
    print(String(format: "TRAIN  %d steps in %.3f s ", nSteps, trainDT))

    // keep the bytes for writeNTC and dequantize back into paramsFloats so the
    // MLP fine-tunes against the values the decoder will actually see at
    // inference. Grid stays frozen for the rest of this run.
    var gridBytes = [UInt8](repeating: 0, count: OFFSET_MLP)
    let invQ   = 1.0 / QUANT.q
    let offset = 1 << (Int(BITS) - 1)   // offset-binary center, 8 at 4 bits
    let maxc   = (1 << Int(BITS)) - 1   // largest code, 15 at 4 bits
    for j in 0..<OFFSET_MLP {
        let coded   = Int((paramsFloats[j] * invQ).rounded()) + offset
        let clamped = max(0, min(maxc, coded))
        gridBytes[j]    = UInt8(clamped)
        paramsFloats[j] = Float(clamped - offset) * QUANT.q
    }

    stepConstsPtr.pointee.adamOffset = UInt32(OFFSET_MLP)
    stepConstsPtr.pointee.q = 0

    let nFineTune = nSteps / 20   // 5% fine tuning as in the nvidia paper
    let mlpSlots = TOTAL - OFFSET_MLP

    for step in 0..<nFineTune {
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        fillBatch(lod: sampleBatchLod(lodMax: lodMax))

        t += 1
        let bc1 = 1.0 - powf(BETA1, Float(t))
        let bc2 = 1.0 - powf(BETA2, Float(t))
        // grid frozen (adamOffset = OFFSET_MLP), so lrGrid is unused
        let lrMlp = cosineLr(step: step, total: nFineTune, lrMax: LR_MLP_MAX)
        adamConstsPtr.pointee = AdamConstants(lrGrid: 0, lrMlp: lrMlp, bc1: bc1, bc2: bc2)

        encodeStep(adamSlots: mlpSlots)

        #if NTC_DEBUG
        if step % logEvery == 0 {
            logMeanLoss("fine-tune", step: step)
        }
        #endif
    }

    event.wait(untilSignaledValue: signalValue, timeoutMS: 5000)

    // pack + write the .ntc file
    let ntcSlots: [NTCSlotInfo] = textureSet.slots.map {
        NTCSlotInfo(semantic:      $0.semantic,
                    swizzle:       $0.swizzle,
                    channels:      $0.channels,
                    channelOffset: $0.channelOffset,
                    sliceIndex:    $0.sliceIndex,
                    isSRGB:        $0.isSRGB)
    }

    let ntcFile = packNTC(srcW: SRC_W, srcH: SRC_H, mipCount: mipCount,
                          kGrids: K_GRIDS, fPerGrid: F_PER_GRID,
                          kHidden: K_HIDDEN, kOutMax: K_OUT_MAX, kOut: K_OUT,
                          peWaves: PE_WAVES,
                          quantScale: QUANT.q, quantBits: Int(BITS),
                          pyramidSizes: PYRAMID_SIZES,
                          neuralMipsForLod: Array(NM_FOR_LOD.prefix(mipCount)),
                          slots: ntcSlots,
                          gridBytes: gridBytes,
                          mlpFloats:  paramsFloats.advanced(by: OFFSET_MLP))

    let ntcURL   = textureSet.manifestDir.appendingPathComponent(QUALITY.ntcFileName(base: textureSet.name))
    let ntcBytes = try writeNTC(ntcFile, to: ntcURL)
    print(String(format: "wrote %@  (%d bytes = %.2f MB)",
                 ntcURL.path, ntcBytes, Double(ntcBytes) / (1024 * 1024)))

    // The .ntc is finished above. Everything past here only measures and
    // pictures it, so a release build stops here.
    #if NTC_DEBUG
    try reportQuality(ctx:              ctx,
                      textureSet:       textureSet,
                      pyramidBuilder:   pyramidBuilder,
                      paramsBuffer:     paramsBuffer,
                      stepConstsBuffer: stepConstsBuffer,
                      srcW:             SRC_W,
                      srcH:             SRC_H,
                      kOut:             K_OUT,
                      event:            event,
                      signalValue:      &signalValue)
    #endif
}

let (manifest, dir) = try loadManifest(at: inputURL)
print("== \(dir.lastPathComponent): \(manifest.models.count) model(s) ==")

var failures: [String] = []
for (i, model) in manifest.models.enumerated() {
    print("-- [\(i + 1)/\(manifest.models.count)] training \(model.name) (\(model.textures.count) textures) --")
    // one model failing must not cost the models already written or the ones
    // still to run; so the batch keeps going and reports what it skipped
    do {
        try trainModel(model, dir: dir)
    } catch {
        FileHandle.standardError.write(Data("SKIP model \(model.name): \(error)\n".utf8))
        failures.append(model.name)
    }
}

if failures.isEmpty {
    print("done")
} else {
    print("done with \(failures.count) skipped: \(failures.joined(separator: ", "))")
}
