import NTCCore
import NTCAssets
import NTCShared
import Metal
import Foundation
import QuartzCore

//let inputPath = "/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG"
let inputPath = "/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/models/flighthelmet/scene.gltf"
let inputURL  = URL(fileURLWithPath: inputPath)

func discoverGLTFs(_ url: URL) -> [URL] {
    var isDir: ObjCBool = false
    FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
    if !isDir.boolValue { return [url] }
    var out: [URL] = []
    if let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) {
        for case let f as URL in en where f.pathExtension.lowercased() == "gltf" {
            out.append(f)
        }
    }
    return out.sorted { $0.path < $1.path }
}

enum TrainerError: Error, CustomStringConvertible {
    case msg(String)
    var description: String { switch self { case .msg(let m): return m } }
}

/// Resolve the input to (manifest, directory) pairs. A .gltf generates its
/// manifest; a manifest.json loads directly (new `{models}` or old flat
/// `{textures}` format); a directory with a manifest.json loads it as a texture
/// set, otherwise it is scanned for .gltf files.
func loadManifests(_ url: URL) throws -> [(manifest: Manifest, dir: URL)] {
    var isDir: ObjCBool = false
    FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)
    if isDir.boolValue {
        let manifestURL = url.appendingPathComponent("manifest.json")
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            return [(try Manifest.load(from: manifestURL), url)]
        }
        return try discoverGLTFs(url).map { (try ManifestGen.generate(fromGLTF: $0), $0.deletingLastPathComponent()) }
    }
    switch url.pathExtension.lowercased() {
        case "gltf": return [(try ManifestGen.generate(fromGLTF: url), url.deletingLastPathComponent())]
        case "json": return [(try Manifest.load(from: url),           url.deletingLastPathComponent())]
        default:     throw TrainerError.msg("unsupported input \(url.lastPathComponent); expected .gltf, manifest.json, or a directory")
    }
}

struct PyramidPreset {
    let sizes:  [Int]      // K_GRIDS grid resolutions, finest first
    let mipMap: [UInt32]   // lod -> index of the first grid in its pair
}

let PYRAMID_PRESETS: [Int: PyramidPreset] = [
    4096: PyramidPreset(sizes:  [1024, 512, 256, 128, 64, 32, 16, 8],
                        mipMap: [0, 0, 0, 0,  2, 2, 2,  4, 4, 4,  6, 6, 6]),
    2048: PyramidPreset(sizes:  [512, 256, 128, 64, 32, 16, 8, 4],
                        mipMap: [0, 0, 0,  2, 2, 2,  4, 4, 4,  6, 6, 6,  6]),
    1024: PyramidPreset(sizes:  [256, 128, 64, 32, 16, 8, 4, 2],
                        mipMap: [0, 0, 0,  2, 2, 2,  4, 4, 4,  6, 6,  6, 6]),
]

let K_BATCH = 4096
let BITS: UInt32 = 4

func fake_quant(bits: UInt32) -> (q: Float, lo: Float, hi: Float) {
    if bits == 0 {
        return (0, 0, 0)
    }
    let N = Float(1 << bits)
    let q = 1.0 / N
    let lo = -(N - 1) / 2 * q
    let hi =  N / 2 * q
    return (q, lo, hi)
}

let SAMPLE_X      = 0
let SAMPLE_Y      = 1
let SAMPLE_LOD    = 2
let SAMPLE_LOSS   = 3
let SAMPLE_STRIDE = 4

let nSteps  = 10000
let logEvery = 100
let BETA1: Float = 0.9
let BETA2: Float = 0.999
let LR_GRID_MAX: Float = 0.01
let LR_MLP_MAX:  Float = 0.005
let UNIFORM_LOD_FRACTION: Float = 0.05

// cosine annealing to 0 across [0, total)
func cosineLr(step: Int, total: Int, lrMax: Float) -> Float {
    let denom = Float(max(total - 1, 1))
    let t = Float(step) / denom
    return lrMax * 0.5 * (1.0 + cosf(.pi * t))
}

// choose randomly a level proportionally to the mip level's area by sampling
// from an exponential distribution. To mitigate undersampling of low resolution
// mip levels, 5% of the batches sample their LOD from a uniform distribution
// of the entire range of the mip chain
func sampleBatchLod(lodMax: Int) -> Int {
    if Float.random(in: 0..<1) < UNIFORM_LOD_FRACTION {
        return Int.random(in: 0..<lodMax)
    }
    let x = Float.random(in: Float.leastNormalMagnitude..<1.0)
    let lod = Int(floor(-logf(x) / logf(4.0)))
    return min(max(lod, 0), lodMax - 1)
}

struct AdamConstants {
    var lrGrid: Float;
    var lrMlp:  Float;
    var bc1:    Float;
    var bc2:    Float
}

struct StepConstants {
    var kBatch:             UInt32
    var pyramidOffsets:     (UInt32, UInt32, UInt32, UInt32,
                             UInt32, UInt32, UInt32, UInt32,
                             UInt32)
    var pyramidSizes:       (UInt32, UInt32, UInt32, UInt32,
                             UInt32, UInt32, UInt32, UInt32)
    var offsetW1:           UInt32
    var offsetB1:           UInt32
    var offsetW2:           UInt32
    var offsetB2:           UInt32
    var offsetW3:           UInt32
    var offsetB3:           UInt32
    var total:              UInt32
    var bits:               UInt32
    var q:                  Float
    var lo:                 Float
    var hi:                 Float
    var adamOffset:         UInt32
    var inferLod:           UInt32
    // max mips 13 for 4k textures
    var neuralMipsForLod:   (UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32, UInt32, UInt32,
                            UInt32)
    var kOut:                UInt32
    var nSlices:             UInt32
    // K_OUT_MAX slice channels and offsets
    var sliceChannels:       (UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32)
    var sliceChannelOffsets: (UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32)
    var srcW:                UInt32
    var srcH:                UInt32
    var mipCount:            UInt32
    var posScale:            Float
}

// ---------------------------------------------------------------------------
// Metal context + pipelines: built once, reused for every model.
// ---------------------------------------------------------------------------
let ctx = try MetalContext(bundle: NTCCoreResources.bundle)
let trainPso = try ctx.makeComputePipelineState(function: "grid_mlp_train")
let inferPso = try ctx.makeComputePipelineState(function: "grid_mlp_infer")
let adamPso   = try ctx.makeComputePipelineState(function: "adam_step")

// ---------------------------------------------------------------------------
// Train one model (submesh/material) and write <name>.ntc beside the glTF.
// ---------------------------------------------------------------------------
@MainActor
func trainModel(_ model: Manifest.Model, dir: URL) throws {
    let textureSet = TextureSet(model: model, dir: dir)

    let materialImages: [LoadedImage]
    let SRC_W: Int
    let SRC_H: Int
    (materialImages, SRC_W, SRC_H) = try textureSet.loadImages() 

    guard let PRESET = PYRAMID_PRESETS[SRC_W] else {
        // Thrown, not fatal: the driver skips this model and keeps going so one
        // unsupported resolution does not abort the whole batch.
        throw TrainerError.msg("no pyramid preset for \(SRC_W)x\(SRC_H); add one to PYRAMID_PRESETS")
    }
    let PYRAMID_SIZES: [Int] = PRESET.sizes
    precondition(PYRAMID_SIZES.count == K_GRIDS)
    precondition(PRESET.mipMap.count == MAX_LODS)

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

    let outputBuffer = ctx.device.makeBuffer(length: SRC_H * SRC_W * K_OUT * MemoryLayout<Float>.stride,
                                             options: .storageModeShared)!
    outputBuffer.label = "NTC.inferOutput"

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

    let pyramidBuilder = try MipPyramidBuilder(ctx: ctx, srcW: SRC_W, srcH: SRC_H, sourceTexture: sourceTexture)

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

    print("pyramid layout: K_GRIDS=\(K_GRIDS)  F_PER_GRID=\(F_PER_GRID)")
    for i in 0..<K_GRIDS {
        print(String(format: "  pyramid[%d] %4dx%-4d x %d ch    offset=%-10d  floats=%d",
                     i, PYRAMID_SIZES[i], PYRAMID_SIZES[i], F_PER_GRID,
                     PYRAMID_OFFSETS[i], PYRAMID_SLOT_FLOATS[i]))
    }

    let setDesc = MTLResidencySetDescriptor()
    setDesc.label = "grid_mlp_train.residency"
    setDesc.initialCapacity = 11
    let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
    residencySet.addAllocation(paramsBuffer)
    residencySet.addAllocation(samplesBuffer)
    residencySet.addAllocation(stepConstsBuffer)
    residencySet.addAllocation(outputBuffer)
    residencySet.addAllocation(mBuffer)
    residencySet.addAllocation(vBuffer)
    residencySet.addAllocation(adamConstsBuffer)
    for alloc in pyramidBuilder.residencyAllocations {
        residencySet.addAllocation(alloc)
    }
    residencySet.commit()
    ctx.queue.addResidencySet(residencySet)
    defer { ctx.queue.removeResidencySet(residencySet) }

    let trainArgDesc = MTL4ArgumentTableDescriptor()
    trainArgDesc.maxBufferBindCount  = 4
    trainArgDesc.maxTextureBindCount = 1
    let trainArgTable = try ctx.device.makeArgumentTable(descriptor: trainArgDesc)
    trainArgTable.setAddress(paramsBuffer.gpuAddress,     index: 0)
    trainArgTable.setAddress(samplesBuffer.gpuAddress,    index: 1)
    trainArgTable.setAddress(stepConstsBuffer.gpuAddress, index: 2)
    trainArgTable.setTexture(pyramidBuilder.pyramidTexture.gpuResourceID, index: 0)

    let adamArgDesc = MTL4ArgumentTableDescriptor()
    adamArgDesc.maxBufferBindCount = 5
    let adamArgTable = try ctx.device.makeArgumentTable(descriptor: adamArgDesc)
    adamArgTable.setAddress(paramsBuffer.gpuAddress,     index: 0)
    adamArgTable.setAddress(mBuffer.gpuAddress,          index: 1)
    adamArgTable.setAddress(vBuffer.gpuAddress,          index: 2)
    adamArgTable.setAddress(adamConstsBuffer.gpuAddress, index: 3)
    adamArgTable.setAddress(stepConstsBuffer.gpuAddress, index: 4)

    let inferArgDesc = MTL4ArgumentTableDescriptor()
    inferArgDesc.maxBufferBindCount = 3
    let inferArgTable = try ctx.device.makeArgumentTable(descriptor: inferArgDesc)
    inferArgTable.setAddress(paramsBuffer.gpuAddress,     index: 0)
    inferArgTable.setAddress(outputBuffer.gpuAddress,     index: 1)
    inferArgTable.setAddress(stepConstsBuffer.gpuAddress, index: 2)

    // Kaiming He uniform: a = sqrt(6/fan_in) gives uniform variance = 2/fan_in.
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

        let batchLod = sampleBatchLod(lodMax: lodMax)
        let wL = SRC_W >> batchLod
        let hL = SRC_H >> batchLod
        for s in 0..<K_BATCH {
            let base = s * SAMPLE_STRIDE
            samplesFloats[base + SAMPLE_X]   = Float(Int.random(in: 0..<wL))
            samplesFloats[base + SAMPLE_Y]   = Float(Int.random(in: 0..<hL))
            samplesFloats[base + SAMPLE_LOD] = Float(batchLod)
        }

        t += 1
        let bc1 = 1.0 - powf(BETA1, Float(t))
        let bc2 = 1.0 - powf(BETA2, Float(t))
        let lrGrid = cosineLr(step: step, total: nSteps, lrMax: LR_GRID_MAX)
        let lrMlp  = cosineLr(step: step, total: nSteps, lrMax: LR_MLP_MAX)
        adamConstsPtr.pointee = AdamConstants(lrGrid: lrGrid, lrMlp: lrMlp, bc1: bc1, bc2: bc2)

        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)

        let trainEnc = cmd.makeComputeCommandEncoder()!
        trainEnc.setComputePipelineState(trainPso)
        trainEnc.setArgumentTable(trainArgTable)
        let tgSize = 256
        let trainTgx = (K_BATCH + tgSize - 1) / tgSize
        trainEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: trainTgx,   height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        trainEnc.endEncoding()

        let adamEnc = cmd.makeComputeCommandEncoder()!
        adamEnc.setComputePipelineState(adamPso)
        adamEnc.setArgumentTable(adamArgTable)
        let adamTgx = (TOTAL + tgSize - 1) / tgSize
        adamEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: adamTgx,   height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        adamEnc.endEncoding()

        cmd.endCommandBuffer()
        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)

        if step % logEvery == 0 {
            event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)
            var sumLoss: Float = 0
            for s in 0..<K_BATCH {
                sumLoss += samplesFloats[s * SAMPLE_STRIDE + SAMPLE_LOSS]
            }
            print("step \(step)\tmean loss = \(sumLoss / Float(K_BATCH))")
        }
    }

    event.wait(untilSignaledValue: signalValue, timeoutMS: 5000)

    let tTrainEnd = CACurrentMediaTime()
    let trainDT = tTrainEnd - tTrainStart
    print(String(format: "TRAIN  %d steps in %.3f s ", nSteps, trainDT))

    // keep the bytes for writeNTC and dequantize back into paramsFloats so the
    // MLP fine-tunes against the values the decoder will actually see at
    // inference. Grid stays frozen for the rest of this run.
    var gridBytes = [UInt8](repeating: 0, count: OFFSET_MLP)
    if QUANT.q > 0 {
        let invQ   = 1.0 / QUANT.q
        let offset = 1 << (Int(BITS) - 1)   // offset-binary center: 8 for 4-bit, 128 for 8-bit
        let maxc   = (1 << Int(BITS)) - 1   // 15 for 4-bit, 255 for 8-bit
        for j in 0..<OFFSET_MLP {
            let coded   = Int((paramsFloats[j] * invQ).rounded()) + offset
            let clamped = max(0, min(maxc, coded))
            gridBytes[j]    = UInt8(clamped)
            paramsFloats[j] = Float(clamped - offset) * QUANT.q
        }
    }

    stepConstsPtr.pointee.adamOffset = UInt32(OFFSET_MLP)
    stepConstsPtr.pointee.q = 0

    let nFineTune = BITS != 0 ? nSteps / 20 : 0 // 5% fine tuning as in the nvidia paper
    let mlpSlots = TOTAL - OFFSET_MLP

    for step in 0..<nFineTune {
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        let batchLod = sampleBatchLod(lodMax: lodMax)
        let wL = SRC_W >> batchLod
        let hL = SRC_H >> batchLod
        for s in 0..<K_BATCH {
            let base = s * SAMPLE_STRIDE
            samplesFloats[base + SAMPLE_X]   = Float(Int.random(in: 0..<wL))
            samplesFloats[base + SAMPLE_Y]   = Float(Int.random(in: 0..<hL))
            samplesFloats[base + SAMPLE_LOD] = Float(batchLod)
        }

        t += 1
        let bc1 = 1.0 - powf(BETA1, Float(t))
        let bc2 = 1.0 - powf(BETA2, Float(t))
        // grid frozen (adamOffset = OFFSET_MLP), so lrGrid is unused
        let lrMlp = cosineLr(step: step, total: nFineTune, lrMax: LR_MLP_MAX)
        adamConstsPtr.pointee = AdamConstants(lrGrid: 0, lrMlp: lrMlp, bc1: bc1, bc2: bc2)

        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)

        let trainEnc = cmd.makeComputeCommandEncoder()!
        trainEnc.setComputePipelineState(trainPso)
        trainEnc.setArgumentTable(trainArgTable)
        let tgSize = 256
        let trainTgx = (K_BATCH + tgSize - 1) / tgSize
        trainEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: trainTgx, height: 1, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: tgSize,   height: 1, depth: 1))
        trainEnc.endEncoding()

        let adamEnc = cmd.makeComputeCommandEncoder()!
        adamEnc.setComputePipelineState(adamPso)
        adamEnc.setArgumentTable(adamArgTable)
        let adamTgx = (mlpSlots + tgSize - 1) / tgSize
        adamEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: adamTgx, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgSize,  height: 1, depth: 1))
        adamEnc.endEncoding()

        cmd.endCommandBuffer()
        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)

        if step % logEvery == 0 {
            event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)
            var sumLoss: Float = 0
            for s in 0..<K_BATCH {
                sumLoss += samplesFloats[s * SAMPLE_STRIDE + SAMPLE_LOSS]
            }
            print("fine-tune step \(step)\tmean loss = \(sumLoss / Float(K_BATCH))")
        }
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

    let ntcURL = textureSet.manifestDir.appendingPathComponent("\(textureSet.name).ntc")
    try writeNTC(ntcFile, to: ntcURL)

    let ntcBytes = (try? FileManager.default.attributesOfItem(atPath: ntcURL.path))?[.size] as? Int ?? 0
    print(String(format: "wrote %@  (%d bytes = %.2f MB)", ntcURL.path, ntcBytes, Double(ntcBytes) / (1024 * 1024)))

    // infer all mips and write them to two [4096+2048, 4096] texture atlases
    let ATLAS_W = SRC_W + SRC_W / 2
    let ATLAS_H = SRC_H

    var atlases: [[Float]] = textureSet.slots.map { s in
        [Float](repeating: 0, count: ATLAS_W * ATLAS_H * s.channels)
    }

    let outPtr = outputBuffer.contents().bindMemory(to: Float.self,
                                                    capacity: SRC_H * SRC_W * K_OUT)

    // pyramid mip texels back to CPU for PSNR (one slice at a time)
    var pyramidScratch = [SIMD4<Float>](repeating: .zero, count: SRC_H * SRC_W)

    func readPyramidSlice(lod: Int, slice: Int, outWL: Int, outHL: Int) {
        let region = MTLRegionMake2D(0, 0, outWL, outHL)
        pyramidScratch.withUnsafeMutableBufferPointer { buf in
            pyramidBuilder.pyramidTexture.getBytes(buf.baseAddress!,
                                                   bytesPerRow: outWL * MemoryLayout<SIMD4<Float>>.stride,
                                                   bytesPerImage: outWL * outHL * MemoryLayout<SIMD4<Float>>.stride,
                                                   from: region,
                                                   mipmapLevel: lod,
                                                   slice: slice)
        }
    }

    func materialMse(_ s: TextureSlot, outWL: Int, outHL: Int) -> Double {
        var mse: Double = 0
        let base = s.channelOffset
        if s.channels == 3 {
            for i in 0..<(outWL * outHL) {
                let gt = pyramidScratch[i]
                let dr = Double(outPtr[i * K_OUT + base + 0] - gt.x)
                let dg = Double(outPtr[i * K_OUT + base + 1] - gt.y)
                let db = Double(outPtr[i * K_OUT + base + 2] - gt.z)
                mse += dr * dr + dg * dg + db * db
            }
        } else {
            for i in 0..<(outWL * outHL) {
                let gt = pyramidScratch[i]
                let d = Double(outPtr[i * K_OUT + base] - gt.x)
                mse += d * d
            }
        }
        return mse / Double(outWL * outHL * s.channels)
    }

    for lod in 0..<pyramidBuilder.mipCount {
        let outWL = max(SRC_W >> lod, 1)
        let outHL = max(SRC_H >> lod, 1)

        stepConstsPtr.pointee.inferLod = UInt32(lod)

        let inferCmd = ctx.device.makeCommandBuffer()!
        inferCmd.beginCommandBuffer(allocator: ctx.allocator)

        let inferEnc = inferCmd.makeComputeCommandEncoder()!
        inferEnc.setComputePipelineState(inferPso)
        inferEnc.setArgumentTable(inferArgTable)
        let tgx = (outWL + 15) / 16
        let tgy = (outHL + 15) / 16
        inferEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: tgx, height: tgy, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16,  height: 16, depth: 1))
        inferEnc.endEncoding()

        inferCmd.endCommandBuffer()
        ctx.queue.commit([inferCmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 10000)

        // mip 0 goes on the left, rest of the mips to the right and down
        let xOff: Int
        let yOff: Int
        if lod == 0 {
            xOff = 0
            yOff = 0
        } else {
            xOff = SRC_W
            // cumulative height of mips 1 to lod-1
            var cum = 0
            for k in 1..<lod {
                cum += max(SRC_H >> k, 1)
            }
            yOff = cum
        }

        // copy this mip into every slot's atlas
        for (si, slot) in textureSet.slots.enumerated() {
            for y in 0..<outHL {
                for x in 0..<outWL {
                    let srcIdx = (y * outWL + x) * K_OUT + slot.channelOffset
                    let dstIdx = ((yOff + y) * ATLAS_W + (xOff + x)) * slot.channels
                    for c in 0..<slot.channels {
                        atlases[si][dstIdx + c] = outPtr[srcIdx + c]
                    }
                }
            }
        }
    }

    let outDir = textureSet.manifestDir.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    for (si, slot) in textureSet.slots.enumerated() {
        let img = LoadedImage(pixels: atlases[si],
                              height: ATLAS_H,
                              width:  ATLAS_W,
                              channels: slot.channels)
        try save_image(img, to: outDir.appendingPathComponent("grid_mlp_\(textureSet.name)_\(slot.semantic.lowercased())_lod_atlas.png"))
    }
}

let sources = try loadManifests(inputURL)
guard !sources.isEmpty else { fatalError("no trainable input at \(inputPath)") }

var failures: [String] = []
for (manifest, dir) in sources {
    print("== \(dir.lastPathComponent): \(manifest.models.count) model(s) ==")
    for (i, model) in manifest.models.enumerated() {
        print("-- [\(i + 1)/\(manifest.models.count)] training \(model.name) (\(model.textures.count) textures) --")
        // one model failing must not lose the models already written or the ones still to run
        do {
            try trainModel(model, dir: dir)
        } catch {
            FileHandle.standardError.write(Data("SKIP model \(model.name): \(error)\n".utf8))
            failures.append("\(model.name)")
        }
    }
}
if !failures.isEmpty {
    print("done with \(failures.count) skipped: \(failures.joined(separator: ", "))")
}
