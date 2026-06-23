import XCTest
import Metal
@testable import NTCCore

private let kGridH  = 8
private let kGridW  = 8
private let kGridCh = 3

private enum GridOffset {
    static let total = kGridH * kGridW * kGridCh   // 192 floats
}

final class BilinearSampleTest: XCTestCase {

    func testBilinearForwardSingleQuery() throws {
        let ctx = try MetalContext(bundle: .module)
        let pso = try ctx.makeComputePipelineState(function: "bilinear_sample_forward")

        // grid buffer
        let gridBufferLength = GridOffset.total * MemoryLayout<Float>.stride
        let gridBuffer = ctx.device.makeBuffer(length: gridBufferLength, options: .storageModeShared)!
        let gridFloats = gridBuffer.contents().bindMemory(to: Float.self, capacity: GridOffset.total)
        for i in 0..<GridOffset.total { gridFloats[i] = Float(i) } // 0 to 191

        // query buffer
        let queryBuffer = ctx.device.makeBuffer(length: 2 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let queryFloats = queryBuffer.contents().bindMemory(to: Float.self, capacity: 2)
        queryFloats[0] = 5.3 // ix
        queryFloats[1] = 2.7 // iy

        // out buffer
        let outBuffer = ctx.device.makeBuffer(length: kGridCh * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let outFloats = outBuffer.contents().bindMemory(to: Float.self, capacity: kGridCh)
        for i in 0..<kGridCh { outFloats[i] = 0 }

        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "bilinear_sample.residency"
        setDesc.initialCapacity = 3
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(gridBuffer)
        residencySet.addAllocation(queryBuffer)
        residencySet.addAllocation(outBuffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)

        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 3
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(gridBuffer.gpuAddress,  index: 0)
        argTable.setAddress(queryBuffer.gpuAddress, index: 1)
        argTable.setAddress(outBuffer.gpuAddress,   index: 2)

        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setArgumentTable(argTable)
        enc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cmd.endCommandBuffer()

        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        //   iy=2.7, ix=5.3  ->  iy0=2, ix0=5; fy=0.7, fx=0.3
        //   w00 = (1-0.7)(1-0.3) = 0.21   corner (2,5)
        //   w01 = (1-0.7)(0.3)   = 0.09   corner (2,6)
        //   w10 = (0.7)(1-0.3)   = 0.49   corner (3,5)
        //   w11 = (0.7)(0.3)     = 0.21   corner (3,6)
        //   grid[2, 5, 0] = 2*24 + 5*3 + 0 = 48 + 15 + 0 = 63
        //   grid[2, 6, 0] = 2*24 + 6*3 + 0 = 48 + 18 + 0 = 66
        //   grid[3, 5, 0] = 3*24 + 5*3 + 0 = 72 + 15 + 0 = 87
        //   grid[3, 6, 0] = 3*24 + 6*3 + 0 = 72 + 18 + 0 = 90
        let w00: Float = 0.21
        let w01: Float = 0.09
        let w10: Float = 0.49
        let w11: Float = 0.21
        let expected: [Float] = [
            w00*63 + w01*66 + w10*87 + w11*90, // channel 0
            w00*64 + w01*67 + w10*88 + w11*91, // channel 1
            w00*65 + w01*68 + w10*89 + w11*92, // channel 2
        ]

        for ch in 0..<kGridCh {
            XCTAssertEqual(outFloats[ch], expected[ch], accuracy: 1e-5, "out[\(ch)] mismatch")
        }
    }
}
