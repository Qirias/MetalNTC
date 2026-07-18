import NTCCore
import NTCAssets
import Metal
import Foundation

let K_GRIDS     = 8
let F_PER_GRID  = 8
let MAX_LODS    = 13

let PYRAMID_SIZES: [Int] = [1024, 512, 256, 128, 64, 32, 16, 8]
precondition(PYRAMID_SIZES.count == K_GRIDS)

let PYRAMID_SLOT_FLOATS: [Int] = PYRAMID_SIZES.map { $0 * $0 * F_PER_GRID }

let F_TOTAL = 2 * F_PER_GRID

let PE_WAVES = 3
let PE_DIM   = 4 * PE_WAVES
let F_IN     = F_TOTAL + PE_DIM

let K_HIDDEN = 64
let K_OUT = 3
let K_BATCH = 4096

let SRC_W = 4096
let SRC_H = 4096

let BITS: UInt32 = 8

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
let OFFSET_B3   = OFFSET_W3 + K_HIDDEN * K_OUT
let TOTAL       = OFFSET_B3 + K_OUT



//   level 0 (grids 0,1 = 1024, 512):  mips 0-3     res 4096, 2048, 1024, 512
//   level 1 (grids 2,3 = 256, 128):   mips 4-6     res 256, 128, 64
//   level 2 (grids 4,5 = 64, 32):     mips 7-9     res 32, 16, 8
//   level 3 (grids 6,7 = 16, 8):      mips 10-12   res 4, 2, 1
//
// TODO: create preset tables for various resolutions
let mipCount = Int(log2(Double(SRC_W))) + 1
let NM_FOR_LOD: [UInt32] = [0, 0, 0, 0,   2, 2, 2,   4, 4, 4,   6, 6, 6]

let SAMPLE_X      = 0
let SAMPLE_Y      = 1
let SAMPLE_LOD    = 2
let SAMPLE_LOSS   = 3
let SAMPLE_STRIDE = 4

let srcURL = URL(fileURLWithPath: "/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG/ManholeCover010_4K-PNG_Color.png")
let img = try load_image(at: srcURL, channels: K_OUT)

let ctx = try MetalContext(bundle: NTCCoreResources.bundle)
let trainPso = try ctx.makeComputePipelineState(function: "grid_mlp_train")
let inferPso = try ctx.makeComputePipelineState(function: "grid_mlp_infer")
let adamPso   = try ctx.makeComputePipelineState(function: "adam_step")

let paramsBuffer = ctx.device.makeBuffer(length: TOTAL * 2 * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
let paramsFloats = paramsBuffer.contents().bindMemory(to: Float.self, capacity: TOTAL * 2)

let samplesBuffer = ctx.device.makeBuffer(length: K_BATCH * SAMPLE_STRIDE * MemoryLayout<Float>.stride,
                                          options: .storageModeShared)!
let samplesFloats = samplesBuffer.contents().bindMemory(to: Float.self, capacity: K_BATCH * SAMPLE_STRIDE)

let outputBuffer = ctx.device.makeBuffer(length: SRC_H * SRC_W * K_OUT * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!

let sourceTexDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: SRC_W, height: SRC_H, mipmapped: false)
sourceTexDesc.usage       = [.shaderRead]
sourceTexDesc.storageMode = .shared
let sourceTexture = ctx.device.makeTexture(descriptor: sourceTexDesc)!

var rgbaPixels = [SIMD4<Float>](repeating: .zero, count: SRC_H * SRC_W)
for i in 0..<(SRC_H * SRC_W) {
    rgbaPixels[i] = SIMD4<Float>(img.pixels[i * K_OUT + 0],
                                 img.pixels[i * K_OUT + 1],
                                 img.pixels[i * K_OUT + 2],
                                 0)
}

rgbaPixels.withUnsafeBufferPointer { buf in
    sourceTexture.replace(region: MTLRegionMake2D(0, 0, SRC_W, SRC_H),
                          mipmapLevel: 0,
                          withBytes: buf.baseAddress!,
                          bytesPerRow: SRC_W * MemoryLayout<SIMD4<Float>>.stride)
}

let pyramidBuilder = try MipPyramidBuilder(ctx: ctx, srcW: SRC_W, srcH: SRC_H, sourceTexture: sourceTexture)

let mBuffer = ctx.device.makeBuffer(length: TOTAL * MemoryLayout<Float>.stride,
                                    options: .storageModeShared)!
let vBuffer = ctx.device.makeBuffer(length: TOTAL * MemoryLayout<Float>.stride,
                                    options: .storageModeShared)!
memset(mBuffer.contents(), 0, TOTAL * MemoryLayout<Float>.stride)
memset(vBuffer.contents(), 0, TOTAL * MemoryLayout<Float>.stride)

struct AdamConstants {
    var lr: Float;
    var bc1: Float;
    var bc2: Float
}

let adamConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<AdamConstants>.stride,
                                             options: .storageModeShared)!
let adamConstsPtr = adamConstsBuffer.contents().bindMemory(to: AdamConstants.self, capacity: 1)

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
}

let stepConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<StepConstants>.stride,
                                             options: .storageModeShared)!
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
                         NM_FOR_LOD[12])
)

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

// https://en.wikipedia.org/wiki/Continuous_uniform_distribution
// Kaiming He uniform. Float.random() is uniform
// target Var(X) = 2/fan_in. Uniform(-a, a) has variance a^2/3
// (b - a)^2 / 12 where a and b are interval endpoints. Our a is the half-width:
// our lower endpoint is -a and upper is +a
// width = upper - lower = a - (-a) = 2a
// width^2 = (2a)^2 = 4a^2
// variance = 4a^2 / 12 = a^2/3

// a^2/3 = 2 / fan_in -> uniform variance = target variance
// a^2 = 6 / fan_in
// so a = sqrt(6/fan_in)

let w1Bound = sqrtf(6.0 / Float(F_IN))
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

let nSteps = 10000
let logEvery = 100
var t: UInt32 = 0
let BETA1: Float = 0.9
let BETA2: Float = 0.999
let LR: Float = 1e-3
let lodMax = pyramidBuilder.mipCount - 2

for step in 0..<nSteps {
    event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

    for s in 0..<K_BATCH {
        let base = s * SAMPLE_STRIDE
        let lod = Int.random(in: 0..<lodMax)
        let wL  = SRC_W >> lod
        let hL  = SRC_H >> lod
        samplesFloats[base + SAMPLE_X]   = Float(Int.random(in: 0..<wL))
        samplesFloats[base + SAMPLE_Y]   = Float(Int.random(in: 0..<hL))
        samplesFloats[base + SAMPLE_LOD] = Float(lod)
    }

    t += 1
    let bc1 = 1.0 - powf(BETA1, Float(t))
    let bc2 = 1.0 - powf(BETA2, Float(t))
    adamConstsPtr.pointee = AdamConstants(lr: LR, bc1: bc1, bc2: bc2)
    
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

// in a later stage we will first pack to uint8_t (using Q_G1 and Q_G2) in the .ntc file and then
// unpack it. The MLP has to be fine tuned with the loss of this procedure, so for now we simulate it
// by doing both operations here
if QUANT.q > 0 {
    for j in 0..<OFFSET_MLP {
        paramsFloats[j] = (paramsFloats[j] / QUANT.q).rounded() * QUANT.q
    }
}

stepConstsPtr.pointee.adamOffset = UInt32(OFFSET_MLP)

let nFineTune = BITS != 0 ? nSteps / 20 : 0 // 5% fine tuning as in the nvidia paper
let mlpSlots = TOTAL - OFFSET_MLP

for step in 0..<nFineTune {
    event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

    for s in 0..<K_BATCH {
        let base = s * SAMPLE_STRIDE
        let lod = Int.random(in: 0..<lodMax)
        let wL  = SRC_W >> lod
        let hL  = SRC_H >> lod
        samplesFloats[base + SAMPLE_X]   = Float(Int.random(in: 0..<wL))
        samplesFloats[base + SAMPLE_Y]   = Float(Int.random(in: 0..<hL))
        samplesFloats[base + SAMPLE_LOD] = Float(lod)
    }

    t += 1
    let bc1 = 1.0 - powf(BETA1, Float(t))
    let bc2 = 1.0 - powf(BETA2, Float(t))
    adamConstsPtr.pointee = AdamConstants(lr: LR, bc1: bc1, bc2: bc2)

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


// infer all mips and write them to a [4096+2048, 4096] texture atlas
let ATLAS_W = SRC_W + SRC_W / 2
let ATLAS_H = SRC_H
var atlas = [Float](repeating: 0, count: ATLAS_W * ATLAS_H * K_OUT)

let outPtr = outputBuffer.contents().bindMemory(to: Float.self,
                                                capacity: SRC_H * SRC_W * K_OUT)

// pyramid mip texels back to CPU for PSNR
var pyramidScratch = [SIMD4<Float>](repeating: .zero, count: SRC_H * SRC_W)

for lod in 0..<pyramidBuilder.mipCount {
    let outWL = max(SRC_W >> lod, 1)
    let outHL = max(SRC_H >> lod, 1)
    let pixelCountLod = outWL * outHL * K_OUT

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

    // read ground-truth mip from the pyramid texture for PSNR
    let region = MTLRegionMake2D(0, 0, outWL, outHL)
    pyramidScratch.withUnsafeMutableBufferPointer { buf in
        pyramidBuilder.pyramidTexture.getBytes(buf.baseAddress!,
                                               bytesPerRow: outWL * MemoryLayout<SIMD4<Float>>.stride,
                                               bytesPerImage: outWL * outHL * MemoryLayout<SIMD4<Float>>.stride,
                                               from: region,
                                               mipmapLevel: lod,
                                               slice: 0)
    }

    var mse: Double = 0
    for i in 0..<(outWL * outHL) {
        let gt = pyramidScratch[i]
        let dr = Double(outPtr[i * K_OUT + 0] - gt.x)
        let dg = Double(outPtr[i * K_OUT + 1] - gt.y)
        let db = Double(outPtr[i * K_OUT + 2] - gt.z)
        mse += dr * dr + dg * dg + db * db
    }
    mse /= Double(pixelCountLod)
    let psnr = 10.0 * log10(1.0 / mse)
    print(String(format: "LOD %2d  %4dx%-4d  MSE = %.6e   PSNR = %.2f dB",
                 lod, outWL, outHL, mse, psnr))

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

    for y in 0..<outHL {
        for x in 0..<outWL {
            let srcIdx = (y * outWL + x) * K_OUT
            let dstIdx = ((yOff + y) * ATLAS_W + (xOff + x)) * K_OUT
            atlas[dstIdx + 0] = outPtr[srcIdx + 0]
            atlas[dstIdx + 1] = outPtr[srcIdx + 1]
            atlas[dstIdx + 2] = outPtr[srcIdx + 2]
        }
    }
}

let atlasImg = LoadedImage(pixels: atlas, height: ATLAS_H, width: ATLAS_W, channels: K_OUT)
let outDir = URL(fileURLWithPath:"/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG/output")
let outURL = outDir.appendingPathComponent("grid_mlp_color_lod_atlas.png")
try save_image(atlasImg, to: outURL)
