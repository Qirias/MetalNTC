import XCTest
import Metal
@testable import NTCCore

private let kIn     = Shapes.kIn
private let kHidden = Shapes.kHidden
private let kOut    = Shapes.kOut

private enum Offset {
    static let w1    = 0
    static let b1    = w1   + kIn * kHidden          // 4
    static let w2    = b1   + kHidden                // 8
    static let b2    = w2   + kHidden * kOut         // 16
    static let x     = b2   + kOut                   // 18
    static let y     = x    + kIn                    // 19
    static let out   = y    + kOut                   // 21
    static let loss  = out  + kOut                   // 23
    static let dW1   = loss + 1                      // 24
    static let db1   = dW1  + kIn * kHidden          // 28
    static let dW2   = db1  + kHidden                // 32
    static let db2   = dW2  + kHidden * kOut         // 40
    static let total = db2  + kOut                   // 42
}

// MLP test for forward/backward pass at one fixed point (sin(0.5), cos(0.5))
final class SinMLPTest: XCTestCase {
    func testForwardBackward() throws {
        let ctx = try MetalContext(bundle: .module)
        let pso = try ctx.makeComputePipelineState(function: "sin_mlp_forward_backward")

        let bufferLength = Offset.total * MemoryLayout<Float>.stride
        let buffer = ctx.device.makeBuffer(length: bufferLength,
                                           options: .storageModeShared)!
        let floats = buffer.contents().bindMemory(to: Float.self, capacity: Offset.total)

        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 1
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(buffer.gpuAddress, index: 0)

        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "sin_mlp.residency"
        setDesc.initialCapacity = 1
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(buffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)

        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        struct Result {
            var out:  [Float]
            var loss: Float
            var dW1:  [Float]
            var db1:  [Float]
            var dW2:  [Float]
            var db2:  [Float]
        }

        func dispatch(W1: [Float], b1: [Float], W2: [Float], b2: [Float], x: [Float],  y: [Float]) -> Result {
            precondition(W1.count == kIn * kHidden)
            precondition(b1.count == kHidden)
            precondition(W2.count == kHidden * kOut)
            precondition(b2.count == kOut)
            precondition(x.count  == kIn)
            precondition(y.count  == kOut)

            // inputs
            for i in 0..<W1.count { floats[Offset.w1 + i] = W1[i] }
            for i in 0..<b1.count { floats[Offset.b1 + i] = b1[i] }
            for i in 0..<W2.count { floats[Offset.w2 + i] = W2[i] }
            for i in 0..<b2.count { floats[Offset.b2 + i] = b2[i] }
            for i in 0..<x.count  { floats[Offset.x  + i] = x[i]  }
            for i in 0..<y.count  { floats[Offset.y  + i] = y[i]  }

            // outputs
            for i in 0..<kOut             { floats[Offset.out + i] = 0 }
            floats[Offset.loss] = 0
            for i in 0..<(kIn * kHidden)  { floats[Offset.dW1 + i] = 0 }
            for i in 0..<kHidden          { floats[Offset.db1 + i] = 0 }
            for i in 0..<(kHidden * kOut) { floats[Offset.dW2 + i] = 0 }
            for i in 0..<kOut             { floats[Offset.db2 + i] = 0 }

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

            return Result(out:  (0..<kOut).map             { floats[Offset.out + $0] },
                          loss: floats[Offset.loss],
                          dW1:  (0..<(kIn * kHidden)).map  { floats[Offset.dW1 + $0] },
                          db1:  (0..<kHidden).map          { floats[Offset.db1 + $0] },
                          dW2:  (0..<(kHidden * kOut)).map { floats[Offset.dW2 + $0] },
                          db2:  (0..<kOut).map             { floats[Offset.db2 + $0] })
        }

        let W1: [Float] = [0.3, -0.5, 0.7, -0.2]
        let b1: [Float] = [0, 0, 0, 0]
        let W2: [Float] = [0.4, -0.3,
                           0.1,  0.5,
                          -0.2,  0.6,
                           0.3, -0.4]   // row-major: W2[j*kOut + k]
        let b2: [Float] = [0, 0]
        let x:  [Float] = [0.5]
        let y:  [Float] = [Float(sin(0.5)), Float(cos(0.5))]

        let nominal = dispatch(W1: W1, b1: b1, W2: W2, b2: b2, x: x, y: y)
        print("out  =", nominal.out)
        print("loss =", nominal.loss)
        print("dW1  =", nominal.dW1)
        print("db1  =", nominal.db1)
        print("dW2  =", nominal.dW2)
        print("db2  =", nominal.db2)


        XCTAssertEqual(nominal.loss,    0.463200, accuracy: 1e-5, "loss mismatch")
        XCTAssertEqual(nominal.out[0], -0.034062, accuracy: 1e-5, "out[0] mismatch")
        XCTAssertEqual(nominal.out[1],  0.063500, accuracy: 1e-5, "out[1] mismatch")
        // backward
        XCTAssertEqual(nominal.dW1[2], -0.130191, accuracy: 1e-5, "dW1[0,2] mismatch")
        XCTAssertEqual(nominal.db1[1], -0.171896, accuracy: 1e-5, "db1[1] mismatch")
        XCTAssertEqual(nominal.dW2[4], -0.105586, accuracy: 1e-5, "dW2[2,0] mismatch")
        XCTAssertEqual(nominal.db2[1], -0.814083, accuracy: 1e-5, "db2[1] mismatch")

        func gradientsAgree(estimate: Float, analytic: Float) -> Bool {
            let denom = max(abs(estimate), abs(analytic), 1e-3)
            return abs(estimate - analytic) / denom < 1e-3
        }

        let h: Float = 1e-3

        // W1[0,2]
        var W1Pert = W1
        W1Pert[2] += h
        let W1Plus  = dispatch(W1: W1Pert, b1: b1, W2: W2, b2: b2, x: x, y: y)
        W1Pert[2] -= 2 * h
        let W1Minus = dispatch(W1: W1Pert, b1: b1, W2: W2, b2: b2, x: x, y: y)
        let dW1_02_fd = (W1Plus.loss - W1Minus.loss) / (2 * h)
        XCTAssert(gradientsAgree(estimate: dW1_02_fd, analytic: nominal.dW1[2]), "FD dW1[0,2]: estimate=\(dW1_02_fd) analytic=\(nominal.dW1[2])")

        // b1[1]
        var b1Pert = b1
        b1Pert[1] += h
        let b1Plus  = dispatch(W1: W1, b1: b1Pert, W2: W2, b2: b2, x: x, y: y)
        b1Pert[1] -= 2 * h
        let b1Minus = dispatch(W1: W1, b1: b1Pert, W2: W2, b2: b2, x: x, y: y)
        let db1_1_fd = (b1Plus.loss - b1Minus.loss) / (2 * h)
        XCTAssert(gradientsAgree(estimate: db1_1_fd, analytic: nominal.db1[1]), "FD db1[1]: estimate=\(db1_1_fd) analytic=\(nominal.db1[1])")

        // W2[2,0]
        var W2Pert = W2
        W2Pert[4] += h
        let W2Plus  = dispatch(W1: W1, b1: b1, W2: W2Pert, b2: b2, x: x, y: y)
        W2Pert[4] -= 2 * h
        let W2Minus = dispatch(W1: W1, b1: b1, W2: W2Pert, b2: b2, x: x, y: y)
        let dW2_20_fd = (W2Plus.loss - W2Minus.loss) / (2 * h)
        XCTAssert(gradientsAgree(estimate: dW2_20_fd, analytic: nominal.dW2[4]), "FD dW2[2,0]: estimate=\(dW2_20_fd) analytic=\(nominal.dW2[4])")

        // b2[1]
        var b2Pert = b2
        b2Pert[1] += h
        let b2Plus  = dispatch(W1: W1, b1: b1, W2: W2, b2: b2Pert, x: x, y: y)
        b2Pert[1] -= 2 * h
        let b2Minus = dispatch(W1: W1, b1: b1, W2: W2, b2: b2Pert, x: x, y: y)
        let db2_1_fd = (b2Plus.loss - b2Minus.loss) / (2 * h)
        XCTAssert(gradientsAgree(estimate: db2_1_fd, analytic: nominal.db2[1]), "FD db2[1]: estimate=\(db2_1_fd) analytic=\(nominal.db2[1])")
    }
}
