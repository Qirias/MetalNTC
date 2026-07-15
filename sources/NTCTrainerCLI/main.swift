import NTCCore
import NTCAssets
import Metal
import Foundation

let GRID_H1 = 1024
let GRID_W1 = 1024
let GRID_F1 = 8
let GRID1_TOTAL = GRID_H1 * GRID_W1 * GRID_F1

let GRID_H2 = 512
let GRID_W2 = 512
let GRID_F2 = 8
let GRID2_TOTAL = GRID_H2 * GRID_W2 * GRID_F2

let GRID_F_TOTAL = GRID_F1 + GRID_F2

let PE_WAVES = 3
let PE_DIM   = 4 * PE_WAVES
let F_IN     = GRID_F_TOTAL + PE_DIM

let K_HIDDEN = 64
let K_OUT = 3
let K_BATCH = 4096

let SRC_W = 4096
let SRC_H = 4096

let BITS_G1: UInt32 = 8
let BITS_G2: UInt32 = 8

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

let (Q_G1, LO_G1, HI_G1) = fake_quant(bits: BITS_G1)
let (Q_G2, LO_G2, HI_G2) = fake_quant(bits: BITS_G2)

let OFFSET_G1   = 0
let OFFSET_G2   = OFFSET_G1 + GRID1_TOTAL
let OFFSET_W1   = OFFSET_G2 + GRID2_TOTAL
let OFFSET_B1   = OFFSET_W1 + F_IN * K_HIDDEN
let OFFSET_W2   = OFFSET_B1 + K_HIDDEN
let OFFSET_B2   = OFFSET_W2 + K_HIDDEN * K_HIDDEN
let OFFSET_W3   = OFFSET_B2 + K_HIDDEN
let OFFSET_B3   = OFFSET_W3 + K_HIDDEN * K_OUT
let TOTAL       = OFFSET_B3 + K_OUT

let SAMPLE_X      = 0
let SAMPLE_Y      = 1
let SAMPLE_LOSS   = 2
let SAMPLE_STRIDE = 3


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
    var kBatch:         UInt32
    var offsetG1:       UInt32
    var offsetG2:       UInt32
    var offsetW1:       UInt32
    var offsetB1:       UInt32
    var offsetW2:       UInt32
    var offsetB2:       UInt32
    var offsetW3:       UInt32
    var offsetB3:       UInt32
    var total:          UInt32
    var bitsPerGrid:    (UInt32, UInt32)
    var qPerGrid:       (Float,  Float)
    var loPerGrid:      (Float,  Float)
    var hiPerGrid:      (Float,  Float)
    var adamOffset:     UInt32 // for fine-tune training after fake quantization
}

let stepConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<StepConstants>.stride,
                                             options: .storageModeShared)!
let stepConstsPtr = stepConstsBuffer.contents().bindMemory(to: StepConstants.self, capacity: 1)
stepConstsPtr.pointee = StepConstants(kBatch:         UInt32(K_BATCH),
                                      offsetG1:       UInt32(OFFSET_G1),
                                      offsetG2:       UInt32(OFFSET_G2),
                                      offsetW1:       UInt32(OFFSET_W1),
                                      offsetB1:       UInt32(OFFSET_B1),
                                      offsetW2:       UInt32(OFFSET_W2),
                                      offsetB2:       UInt32(OFFSET_B2),
                                      offsetW3:       UInt32(OFFSET_W3),
                                      offsetB3:       UInt32(OFFSET_B3),
                                      total:          UInt32(TOTAL),
                                      bitsPerGrid:    (0, 0),
                                      qPerGrid:       (Q_G1,    Q_G2),
                                      loPerGrid:      (LO_G1,   LO_G2),
                                      hiPerGrid:      (HI_G1,   HI_G2),
                                      adamOffset:     0)

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

for i in OFFSET_G1..<OFFSET_W1  { paramsFloats[i] = Float.random(in: -0.05...0.05) }
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

let nSteps = 5000
let logEvery = 100
var t: UInt32 = 0
let BETA1: Float = 0.9
let BETA2: Float = 0.999
let LR: Float = 1e-3

for step in 0..<nSteps {
    event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

    for s in 0..<K_BATCH {
        let base = s * SAMPLE_STRIDE
        samplesFloats[base + SAMPLE_Y] = Float(Int.random(in: 0..<SRC_H))
        samplesFloats[base + SAMPLE_X] = Float(Int.random(in: 0..<SRC_W))
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
if Q_G1 > 0 {
    for i in OFFSET_G1..<OFFSET_G2 {
        paramsFloats[i] = (paramsFloats[i] / Q_G1).rounded() * Q_G1
    }
}
if Q_G2 > 0 {
    for i in OFFSET_G2..<OFFSET_W1 {
        paramsFloats[i] = (paramsFloats[i] / Q_G2).rounded() * Q_G2
    }
}

stepConstsPtr.pointee.adamOffset = UInt32(OFFSET_W1)

let nFineTune = (BITS_G1 == 0 && BITS_G2 == 0) ? 0 : nSteps / 20 // 5% fine tuning as in the nvidia paper
let mlpSlots = TOTAL - OFFSET_W1

for step in 0..<nFineTune {
    event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

    for s in 0..<K_BATCH {
        let base = s * SAMPLE_STRIDE
        samplesFloats[base + SAMPLE_Y] = Float(Int.random(in: 0..<SRC_H))
        samplesFloats[base + SAMPLE_X] = Float(Int.random(in: 0..<SRC_W))
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

let inferCmd = ctx.device.makeCommandBuffer()!
inferCmd.beginCommandBuffer(allocator: ctx.allocator)

let inferEnc = inferCmd.makeComputeCommandEncoder()!
inferEnc.setComputePipelineState(inferPso)
inferEnc.setArgumentTable(inferArgTable)
let tgx = SRC_W / 16
let tgy = SRC_H / 16
inferEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: tgx, height: tgy, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16,  height: 16, depth: 1))
inferEnc.endEncoding()

inferCmd.endCommandBuffer()
ctx.queue.commit([inferCmd])
signalValue += 1
ctx.queue.signalEvent(event, value: signalValue)
event.wait(untilSignaledValue: signalValue, timeoutMS: 10000)

let pixelCount = SRC_H * SRC_W * K_OUT
let outPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: pixelCount)
let outPixels = Array(UnsafeBufferPointer(start: outPtr, count: pixelCount))
let outImg = LoadedImage(pixels: outPixels, height: SRC_H, width: SRC_W, channels: K_OUT)

var mse: Double = 0
for i in 0..<pixelCount {
    let d = Double(outPixels[i] - img.pixels[i])
    mse += d * d
}
mse /= Double(pixelCount)
let psnr = 10.0 * log10(1.0 / mse)
print(String(format: "infer MSE = %.6e   PSNR = %.2f dB", mse, psnr))

let outDir = URL(fileURLWithPath:"/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG/output")
let outURL = outDir.appendingPathComponent("grid_mlp_color.png")
try save_image(outImg, to: outURL)
print("saved infer output to \(outURL.path)")
