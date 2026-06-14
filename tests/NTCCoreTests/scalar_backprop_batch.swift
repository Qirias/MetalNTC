import XCTest
import Metal
@testable import NTCCore

struct BatchParams {
    var w: Float
    var b: Float
    var n: UInt32
    var _pad: UInt32 = 0
}

final class ScalarBackpropBatchTest: XCTestCase {
     func testScalarBackpropBatchAtomicSum() throws {
         let ctx = try MetalContext(bundle: .module)
         let pso = try ctx.makeComputePipelineState(function: "scalar_backprop_batch")

         let N = 4096
         let w0: Float = 0.3
         let b0: Float = -0.1

         var xHost = [Float](repeating: 0, count: N)
         var yHost = [Float](repeating: 0, count: N)
         for i in 0..<N {
             let xi = Float(i + 1) / Float(N)
             xHost[i] = xi
             yHost[i] = 2.0 * xi + 1.0
         }

         // CPU reference (sum, not mean):
         // pred_i = w0*x_i + b0
         // diff_i = pred_i - y_i
         // loss_i = diff_i*diff_i
         // dw_i   = 2*diff_i*x_i
         // db_i   = 2*diff_i
         
         var dwRef: Double = 0
         var dbRef: Double = 0
         var lossRef: Double = 0
         for i in 0..<N {
             let xi = Double(xHost[i])
             let yi = Double(yHost[i])
             let diff = Double(w0) * xi + Double(b0) - yi
             lossRef += diff * diff
             let dpred = 2.0 * diff
             dwRef   += dpred * xi
             dbRef   += dpred
         }

         let xBuffer = ctx.device.makeBuffer(bytes: xHost,
                                             length: MemoryLayout<Float>.stride * N,
                                             options: .storageModeShared)!

          let yBuffer = ctx.device.makeBuffer(bytes: yHost,
                                              length: MemoryLayout<Float>.stride * N,
                                              options: .storageModeShared)!

         // 3 atomic_int slots: [dw_sum, db_sum, loss_sum]
         let accumBuffer = ctx.device.makeBuffer(length: MemoryLayout<Int32>.stride * 3,
                                                 options: .storageModeShared)!
         memset(accumBuffer.contents(), 0, accumBuffer.length)

         var params = BatchParams(w: w0, b: b0, n: UInt32(N))
         let paramsBuffer = ctx.device.makeBuffer(bytes: &params,
                                                  length: MemoryLayout<BatchParams>.stride,
                                                  options: .storageModeShared)!

         let setDesc = MTLResidencySetDescriptor()
         setDesc.label = "scalar_backprop_batch.residency"
         setDesc.initialCapacity = 4
         let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
         residencySet.addAllocation(accumBuffer)
         residencySet.addAllocation(xBuffer)
         residencySet.addAllocation(yBuffer)
         residencySet.addAllocation(paramsBuffer)
         residencySet.commit()
         ctx.queue.addResidencySet(residencySet)

         let argDesc = MTL4ArgumentTableDescriptor()
         argDesc.maxBufferBindCount = 4
         let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
         argTable.setAddress(accumBuffer.gpuAddress,  index: 0)
         argTable.setAddress(xBuffer.gpuAddress,      index: 1)
         argTable.setAddress(yBuffer.gpuAddress,      index: 2)
         argTable.setAddress(paramsBuffer.gpuAddress, index: 3)

         let event = ctx.device.makeSharedEvent()!
         var signalValue: UInt64 = 0

         let cmd = ctx.device.makeCommandBuffer()!
         cmd.beginCommandBuffer(allocator: ctx.allocator)
         let enc = cmd.makeComputeCommandEncoder()!
         enc.setComputePipelineState(pso)
         enc.setArgumentTable(argTable)

         let tgSize = 64
         let groups = (N + tgSize - 1) / tgSize
         enc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: groups, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
         enc.endEncoding()
         cmd.endCommandBuffer()

         ctx.queue.commit([cmd])
         signalValue += 1
         ctx.queue.signalEvent(event, value: signalValue)
         event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

         // must match SCALE in scalar_backprop_batch.metal
         let scale: Double = Double(1 << 14)
         let raw = accumBuffer.contents().bindMemory(to: Int32.self, capacity: 3)
         let dwGPU   = Double(raw[0]) / scale
         let dbGPU   = Double(raw[1]) / scale
         let lossGPU = Double(raw[2]) / scale

         print("dw  gpu=\(dwGPU)   ref=\(dwRef)")
         print("db  gpu=\(dbGPU)   ref=\(dbRef)")
         print("loss gpu=\(lossGPU) ref=\(lossRef)")

         let tolerance: Double = 1e-3
         func relativelyClose(_ gpu: Double, _ ref: Double) -> Bool {
             let denom = max(abs(gpu), abs(ref), 1e-3)
             return abs(gpu - ref) / denom < tolerance
         }
         XCTAssertTrue(relativelyClose(dwGPU,   dwRef),   "dw mismatch")
         XCTAssertTrue(relativelyClose(dbGPU,   dbRef),   "db mismatch")
         XCTAssertTrue(relativelyClose(lossGPU, lossRef), "loss mismatch")
     }
 }
