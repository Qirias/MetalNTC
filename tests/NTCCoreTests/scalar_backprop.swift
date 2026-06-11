import XCTest
import Metal
@testable import NTCCore

struct ScalarIO {
    var w: Float = 0
    var b: Float = 0
    var x: Float = 0
    var y: Float = 0
    var loss: Float = 0
    var dw: Float = 0
    var db: Float = 0
}

final class ScalarbackpopTest: XCTestCase {
    func testScalarBackprop() throws {
        let ctx = try MetalContext()
        let pso = try ctx.makeComputePipelineState(function: "scalar_backprop")

        // init input data
        let buffer = ctx.device.makeBuffer(length: MemoryLayout<ScalarIO>.stride, options: .storageModeShared)!

        // pointer into the buffer memory
        let io = buffer.contents().bindMemory(to: ScalarIO.self, capacity: 1)
        io.pointee.w = 2.0
        io.pointee.b = 0.5
        io.pointee.x = 3.0
        io.pointee.y = 4.0

        // one buffer slot at index 0
        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 1;
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(buffer.gpuAddress, index: 0)
        
        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        // one dispatch
        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)

        let encoder = cmd.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pso)
        encoder.setArgumentTable(argTable)
        encoder.dispatchThreadgroups(threadgroupsPerGrid: MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.endCommandBuffer()

        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        let result = io.pointee
        print(result.loss, result.dw, result.db)

        XCTAssertEqual(result.loss, 6.25, accuracy: 1e-6)
        XCTAssertEqual(result.dw,   15.0, accuracy: 1e-6)
        XCTAssertEqual(result.db,    5.0, accuracy: 1e-6)
    }
}
