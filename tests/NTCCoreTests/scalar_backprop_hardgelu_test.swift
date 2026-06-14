import XCTest
import Metal
@testable import NTCCore


// 1. run the shader once with known inputs to catch bugs in the forward expression
//
// 2. finite-difference check
//    run the forward four more times with
//    w and b nudged up and down by a small step h, and use those
//    losses to estimate dw and db numerically:
//          dw_estimate = (loss(w+h) - loss(w-h)) / (2h)

// five dispatches total: one nominal, two for w (+h, -h), two for b

final class ScalarBackpropHardGELUTest: XCTestCase {

    struct ScalarIO {
        var w: Float = 0
        var b: Float = 0
        var x: Float = 0
        var y: Float = 0
        var loss: Float = 0
        var dw: Float = 0
        var db: Float = 0
    }

    //   w=0.5, b=0, x=1, y=0.2
    //   pred           = 0.5 * 1 + 0              = 0.5
    //   clamp(pred/2)  = clamp(0.25, -1, 1)       = 0.25
    //   out            = 0.5 * 0.5 * (1 + 0.25)   = 0.3125
    //   loss           = (0.3125 - 0.2)^2         = 0.01265625
    //
    //   d(loss)/d(out)   = 2 * (out - y)          = 0.225
    //   hard_gelu_prime(0.5)
    //     = 0.5 * (1 + 0.25) + 0.5 * 0.5 * 0.5 * 1
    //     = 0.625 + 0.125                         = 0.75
    //   d(loss)/d(pred)  = 0.225 * 0.75           = 0.16875
    //   dw = d(loss)/d(pred) * x = 0.16875
    //   db = d(loss)/d(pred)     = 0.16875
    private static let wInit: Float =  0.5
    private static let bInit: Float =  0.0
    private static let xInit: Float =  1.0
    private static let yInit: Float =  0.2
    private static let expectedLoss: Float = 0.01265625
    private static let expectedDW:   Float = 0.16875
    private static let expectedDB:   Float = 0.16875

    func testScalarBackpropHardGELU() throws {
        let ctx = try MetalContext(bundle: .module)
        let pso = try ctx.makeComputePipelineState(function: "scalar_backprop_hardgelu")

        // one shared buffer for inputs and outputs
        let buffer = ctx.device.makeBuffer(length: MemoryLayout<ScalarIO>.stride,
                                           options: .storageModeShared)!
        let io = buffer.contents().bindMemory(to: ScalarIO.self, capacity: 1)

        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 1
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(buffer.gpuAddress, index: 0)

        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        func dispatchKernel(w: Float, b: Float, x: Float, y: Float) -> (loss: Float, dw: Float, db: Float) {
            io.pointee.w = w
            io.pointee.b = b
            io.pointee.x = x
            io.pointee.y = y
            io.pointee.loss = 0
            io.pointee.dw   = 0
            io.pointee.db   = 0

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

            let res = io.pointee
            return (res.loss, res.dw, res.db)
        }

        // run the shader at the nominal inputs and compare its outputs to
        // the numbers we worked out on paper above
        let nominal = dispatchKernel(w: Self.wInit, b: Self.bInit,
                                     x: Self.xInit, y: Self.yInit)
        print("nominal: loss=\(nominal.loss) dw=\(nominal.dw) db=\(nominal.db)")

        XCTAssertEqual(nominal.loss, Self.expectedLoss, accuracy: 1e-6)
        XCTAssertEqual(nominal.dw,   Self.expectedDW,   accuracy: 1e-6)
        XCTAssertEqual(nominal.db,   Self.expectedDB,   accuracy: 1e-6)

    
        // finite-difference gradient
        // we nudge each parameter by a small step h, run the forward
        // twice (once at +h, once at -h), and estimate the gradient from
        // the change in loss:
        //     dw_estimate = (loss(w+h) - loss(w-h)) / (2h)
        // since we know that the forward works, we test against the backward
        // to verify it
        
        func gradientsAgree(estimate: Float, analytic: Float) -> Bool {
            let denom = max(abs(estimate), abs(analytic), 1e-3)
            return abs(estimate - analytic) / denom < 1e-3
        }

        // nudge only w by +h and -h
        let h: Float = 1e-3
        let lossWPlus  = dispatchKernel(w: Self.wInit + h, b: Self.bInit,
                                      x: Self.xInit,    y: Self.yInit).loss
        let lossWMinus = dispatchKernel(w: Self.wInit - h, b: Self.bInit,
                                      x: Self.xInit,    y: Self.yInit).loss
        let dwEstimate = (lossWPlus - lossWMinus) / (2 * h)
        print("FD w: loss(+h)=\(lossWPlus) loss(-h)=\(lossWMinus) estimate=\(dwEstimate) analytic=\(nominal.dw)")
        XCTAssertTrue(gradientsAgree(estimate: dwEstimate, analytic: nominal.dw), "dw mismatch: estimate=\(dwEstimate), analytic=\(nominal.dw)")

        // nudge only b by +h and -h.
        let lossBPlus  = dispatchKernel(w: Self.wInit, b: Self.bInit + h,
                                        x: Self.xInit, y: Self.yInit).loss
        let lossBMinus = dispatchKernel(w: Self.wInit, b: Self.bInit - h,
                                        x: Self.xInit, y: Self.yInit).loss
        let dbEstimate = (lossBPlus - lossBMinus) / (2 * h)
        print("FD b: loss(+h)=\(lossBPlus) loss(-h)=\(lossBMinus) estimate=\(dbEstimate) analytic=\(nominal.db)")
        XCTAssertTrue(gradientsAgree(estimate: dbEstimate, analytic: nominal.db), "db mismatch: estimate=\(dbEstimate), analytic=\(nominal.db)")
    }
}
