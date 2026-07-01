import NTCCore
import NTCTrainer
import NTCAssets
import Metal
import Foundation

let GRID_H = 64, GRID_W = 64, GRID_CH = 3
let GRID_TOTAL = GRID_H * GRID_W * GRID_CH
let SRC_H = 4096, SRC_W = 4096
let K_BATCH = 1024
let SAMPLE_Y = 0, SAMPLE_X = 1, SAMPLE_LOSS = 2
let SAMPLE_STRIDE = 3

let srcURL = URL(fileURLWithPath:
    "/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG/ManholeCover010_4K-PNG_Color.png"
)
let img = try load_image(at: srcURL, channels: GRID_CH)

let ctx = try MetalContext(bundle: NTCCoreResources.bundle)
let trainPso = try ctx.makeComputePipelineState(function: "grid_fit_train")
let inferPso = try ctx.makeComputePipelineState(function: "grid_fit_infer")
let adamPso   = try ctx.makeComputePipelineState(function: "adam_step")

// learnable grid and gradient mirror
let paramsBuffer = ctx.device.makeBuffer(length: GRID_TOTAL * 2 * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
let paramsFloats = paramsBuffer.contents().bindMemory(to: Float.self, capacity: GRID_TOTAL * 2)

let samplesBuffer = ctx.device.makeBuffer(length: K_BATCH * SAMPLE_STRIDE * MemoryLayout<Float>.stride,
                                          options: .storageModeShared)!
let samplesFloats = samplesBuffer.contents().bindMemory(to: Float.self, capacity: K_BATCH * SAMPLE_STRIDE)

let sourceBuffer = ctx.device.makeBuffer(length: img.pixels.count * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
let outputBuffer = ctx.device.makeBuffer(length: SRC_H * SRC_W * GRID_CH * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
img.pixels.withUnsafeBufferPointer { buf in
    sourceBuffer.contents().copyMemory(from: buf.baseAddress!, byteCount: buf.count * MemoryLayout<Float>.stride)
}

let mBuffer = ctx.device.makeBuffer(length: GRID_TOTAL * MemoryLayout<Float>.stride,
                                    options: .storageModeShared)!
let vBuffer = ctx.device.makeBuffer(length: GRID_TOTAL * MemoryLayout<Float>.stride,
                                    options: .storageModeShared)!
memset(mBuffer.contents(), 0, GRID_TOTAL * MemoryLayout<Float>.stride)
memset(vBuffer.contents(), 0, GRID_TOTAL * MemoryLayout<Float>.stride)

struct AdamConstants {
    var lr: Float;
    var bc1: Float;
    var bc2: Float
}

let adamConstsBuffer = ctx.device.makeBuffer(length: MemoryLayout<AdamConstants>.stride,
                                             options: .storageModeShared)!
let adamConstsPtr = adamConstsBuffer.contents().bindMemory(to: AdamConstants.self, capacity: 1)

let nFloatsBuffer = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                          options: .storageModeShared)!

let nFloatsPtr = nFloatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
nFloatsPtr.pointee = UInt32(GRID_TOTAL)

let setDesc = MTLResidencySetDescriptor()
setDesc.label = "grid_fit_train.residency"
setDesc.initialCapacity = 7
let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
residencySet.addAllocation(paramsBuffer)
residencySet.addAllocation(samplesBuffer)
residencySet.addAllocation(sourceBuffer)
residencySet.addAllocation(nFloatsBuffer)
residencySet.addAllocation(outputBuffer)
residencySet.addAllocation(mBuffer)
residencySet.addAllocation(vBuffer)
residencySet.addAllocation(adamConstsBuffer)
residencySet.commit()
ctx.queue.addResidencySet(residencySet)

let trainArgDesc = MTL4ArgumentTableDescriptor()
trainArgDesc.maxBufferBindCount = 3
let trainArgTable = try ctx.device.makeArgumentTable(descriptor: trainArgDesc)
trainArgTable.setAddress(paramsBuffer.gpuAddress,  index: 0)
trainArgTable.setAddress(samplesBuffer.gpuAddress, index: 1)
trainArgTable.setAddress(sourceBuffer.gpuAddress,  index: 2)

let adamArgDesc = MTL4ArgumentTableDescriptor()
adamArgDesc.maxBufferBindCount = 5
let adamArgTable = try ctx.device.makeArgumentTable(descriptor: adamArgDesc)
adamArgTable.setAddress(paramsBuffer.gpuAddress,      index: 0)
adamArgTable.setAddress(mBuffer.gpuAddress,           index: 1)
adamArgTable.setAddress(vBuffer.gpuAddress,           index: 2)
adamArgTable.setAddress(adamConstsBuffer.gpuAddress,  index: 3)
adamArgTable.setAddress(nFloatsBuffer.gpuAddress,     index: 4)

let inferArgDesc = MTL4ArgumentTableDescriptor()
inferArgDesc.maxBufferBindCount = 2
let inferArgTable = try ctx.device.makeArgumentTable(descriptor: inferArgDesc)
inferArgTable.setAddress(paramsBuffer.gpuAddress, index: 0)
inferArgTable.setAddress(outputBuffer.gpuAddress, index: 1)

for i in 0..<GRID_TOTAL { paramsFloats[i] = Float.random(in: -0.05...0.05) }
for i in GRID_TOTAL..<(GRID_TOTAL * 2) { paramsFloats[i] = 0 }

let event = ctx.device.makeSharedEvent()!
var signalValue: UInt64 = 0

let nSteps = 5000
let logEvery = 100
var t: UInt32 = 0
let BETA1: Float = 0.9
let BETA2: Float = 0.999
let LR: Float = 1e-3

for step in 0..<nSteps {
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
    trainEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,       height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: K_BATCH, height: 1, depth: 1))
    trainEnc.endEncoding()

    let adamEnc = cmd.makeComputeCommandEncoder()!
    adamEnc.setComputePipelineState(adamPso)
    adamEnc.setArgumentTable(adamArgTable)
    let tgSize = 256
    let tgx = (GRID_TOTAL + tgSize - 1) / tgSize
    adamEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: tgx,   height: 1, depth: 1),
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

let pixelCount = SRC_H * SRC_W * GRID_CH
let outPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: pixelCount)
let outPixels = Array(UnsafeBufferPointer(start: outPtr, count: pixelCount))
let outImg = LoadedImage(pixels: outPixels, height: SRC_H, width: SRC_W, channels: GRID_CH)

let outDir = URL(fileURLWithPath:"/Users/kiriakosgavras/Documents/MetalNTC/sources/NTCAssets/textures/ManholeCover010_4K-PNG/output")
let outURL = outDir.appendingPathComponent("grid_fit_color.png")
try save_image(outImg, to: outURL)
print("saved infer output to \(outURL.path)")
