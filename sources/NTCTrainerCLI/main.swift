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
let sgdPso   = try ctx.makeComputePipelineState(function: "sgd_step")

// learnable grid and gradient mirror
let paramsBuffer = ctx.device.makeBuffer(length: GRID_TOTAL * 2 * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!
let paramsFloats = paramsBuffer.contents().bindMemory(to: Float.self, capacity: GRID_TOTAL * 2)

let samplesBuffer = ctx.device.makeBuffer(length: K_BATCH * SAMPLE_STRIDE * MemoryLayout<Float>.stride,
                                          options: .storageModeShared)!
let samplesFloats = samplesBuffer.contents().bindMemory(to: Float.self, capacity: K_BATCH * SAMPLE_STRIDE)

let sourceBuffer = ctx.device.makeBuffer(length: img.pixels.count * MemoryLayout<Float>.stride,
                                         options: .storageModeShared)!

img.pixels.withUnsafeBufferPointer { buf in
    sourceBuffer.contents().copyMemory(from: buf.baseAddress!, byteCount: buf.count * MemoryLayout<Float>.stride)
}

let lrBuffer = ctx.device.makeBuffer(length: MemoryLayout<Float>.stride,
                                     options: .storageModeShared)!
let lrPtr = lrBuffer.contents().bindMemory(to: Float.self, capacity: 1)
lrPtr.pointee = 0.05

let nFloatsBuffer = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                          options: .storageModeShared)!

let nFloatsPtr = nFloatsBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
nFloatsPtr.pointee = UInt32(GRID_TOTAL)

let setDesc = MTLResidencySetDescriptor()
setDesc.label = "grid_fit_train.residency"
setDesc.initialCapacity = 5
let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
residencySet.addAllocation(paramsBuffer)
residencySet.addAllocation(samplesBuffer)
residencySet.addAllocation(sourceBuffer)
residencySet.addAllocation(lrBuffer)
residencySet.addAllocation(nFloatsBuffer)
residencySet.commit()
ctx.queue.addResidencySet(residencySet)

let trainArgDesc = MTL4ArgumentTableDescriptor()
trainArgDesc.maxBufferBindCount = 3
let trainArgTable = try ctx.device.makeArgumentTable(descriptor: trainArgDesc)
trainArgTable.setAddress(paramsBuffer.gpuAddress,  index: 0)
trainArgTable.setAddress(samplesBuffer.gpuAddress, index: 1)
trainArgTable.setAddress(sourceBuffer.gpuAddress,  index: 2)

let sgdArgDesc = MTL4ArgumentTableDescriptor()
sgdArgDesc.maxBufferBindCount = 3
let sgdArgTable = try ctx.device.makeArgumentTable(descriptor: sgdArgDesc)
sgdArgTable.setAddress(paramsBuffer.gpuAddress,   index: 0)
sgdArgTable.setAddress(lrBuffer.gpuAddress,       index: 1)
sgdArgTable.setAddress(nFloatsBuffer.gpuAddress,  index: 2)

for i in 0..<GRID_TOTAL { paramsFloats[i] = Float.random(in: -0.05...0.05) }
for i in GRID_TOTAL..<(GRID_TOTAL * 2) { paramsFloats[i] = 0 }

let event = ctx.device.makeSharedEvent()!
var signalValue: UInt64 = 0

let nSteps = 5000
let logEvery = 100

for step in 0..<nSteps {
    for s in 0..<K_BATCH {
        let base = s * SAMPLE_STRIDE
        samplesFloats[base + SAMPLE_Y] = Float(Int.random(in: 0..<SRC_H))
        samplesFloats[base + SAMPLE_X] = Float(Int.random(in: 0..<SRC_W))
    }

    let cmd = ctx.device.makeCommandBuffer()!
    cmd.beginCommandBuffer(allocator: ctx.allocator)

    let trainEnc = cmd.makeComputeCommandEncoder()!
    trainEnc.setComputePipelineState(trainPso)
    trainEnc.setArgumentTable(trainArgTable)
    trainEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,       height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: K_BATCH, height: 1, depth: 1))
    trainEnc.endEncoding()

    let sgdEnc = cmd.makeComputeCommandEncoder()!
    sgdEnc.setComputePipelineState(sgdPso)
    sgdEnc.setArgumentTable(sgdArgTable)
    let tgSize = 256
    let tgx = (GRID_TOTAL + tgSize - 1) / tgSize
    sgdEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: tgx,   height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    sgdEnc.endEncoding()

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
print("done. final params at paramsBuffer[0..<\(GRID_TOTAL)].")
