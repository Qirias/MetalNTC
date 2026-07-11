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

let K_HIDDEN = 64
let K_OUT = 3
let K_BATCH = 1024

let SRC_W = 4096
let SRC_H = 4096

let OFFSET_G1   = 0
let OFFSET_G2   = OFFSET_G1 + GRID1_TOTAL
let OFFSET_W1   = OFFSET_G2 + GRID2_TOTAL
let OFFSET_B1   = OFFSET_W1 + GRID_F_TOTAL * K_HIDDEN
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

let sourceBuffer = ctx.device.makeBuffer(length: img.pixels.count * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!

let outputBuffer = ctx.device.makeBuffer(length: SRC_H * SRC_W * K_OUT * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
img.pixels.withUnsafeBufferPointer { buf in
    sourceBuffer.contents().copyMemory(from: buf.baseAddress!, byteCount: buf.count * MemoryLayout<Float>.stride)
}

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
    var kBatch:   UInt32
    var offsetG1: UInt32
    var offsetG2: UInt32
    var offsetW1: UInt32
    var offsetB1: UInt32
    var offsetW2: UInt32
    var offsetB2: UInt32
    var offsetW3: UInt32
    var offsetB3: UInt32
    var total:    UInt32
}

let stepConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<StepConstants>.stride,
                                             options: .storageModeShared)!
let stepConstsPtr = stepConstsBuffer.contents().bindMemory(to: StepConstants.self, capacity: 1)
stepConstsPtr.pointee = StepConstants(
    kBatch:   UInt32(K_BATCH),
    offsetG1: UInt32(OFFSET_G1),
    offsetG2: UInt32(OFFSET_G2),
    offsetW1: UInt32(OFFSET_W1),
    offsetB1: UInt32(OFFSET_B1),
    offsetW2: UInt32(OFFSET_W2),
    offsetB2: UInt32(OFFSET_B2),
    offsetW3: UInt32(OFFSET_W3),
    offsetB3: UInt32(OFFSET_B3),
    total:    UInt32(TOTAL)
)

let setDesc = MTLResidencySetDescriptor()
setDesc.label = "grid_mlp_train.residency"
setDesc.initialCapacity = 7
let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
residencySet.addAllocation(paramsBuffer)
residencySet.addAllocation(samplesBuffer)
residencySet.addAllocation(sourceBuffer)
residencySet.addAllocation(stepConstsBuffer)
residencySet.addAllocation(outputBuffer)
residencySet.addAllocation(mBuffer)
residencySet.addAllocation(vBuffer)
residencySet.addAllocation(adamConstsBuffer)
residencySet.commit()
ctx.queue.addResidencySet(residencySet)

let trainArgDesc = MTL4ArgumentTableDescriptor()
trainArgDesc.maxBufferBindCount = 4
let trainArgTable = try ctx.device.makeArgumentTable(descriptor: trainArgDesc)
trainArgTable.setAddress(paramsBuffer.gpuAddress,     index: 0)
trainArgTable.setAddress(samplesBuffer.gpuAddress,    index: 1)
trainArgTable.setAddress(sourceBuffer.gpuAddress,     index: 2)
trainArgTable.setAddress(stepConstsBuffer.gpuAddress, index: 3)

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

let w1Bound = sqrtf(6.0 / Float(GRID_F_TOTAL))
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

let nSteps = 10000
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

let outDir = URL(fileURLWithPath:"/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG/output")
let outURL = outDir.appendingPathComponent("grid_mlp_color.png")
try save_image(outImg, to: outURL)
print("saved infer output to \(outURL.path)")
