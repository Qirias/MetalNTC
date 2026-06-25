import XCTest
import Metal
@testable import NTCCore

final class ScalarLinearRegressionTest: XCTestCase {
    static let kInDim = 2
    static let kOutDim = 3
    
    enum Offset {
        static let W      = 0
        static let b      = 6
        static let x      = 9
        static let y      = 11
        static let loss   = 14
        static let yPred  = 15
        static let dW     = 18
        static let db     = 24
        static let total  = 27
    }
    
    func testScalarLinearRegression() throws {
        let ctx = try MetalContext(bundle: .module)
        let pso = try ctx.makeComputePipelineState(function: "scalar_linear_regression")
        
        let bufferLength = Offset.total * MemoryLayout<Float>.stride
        let buffer = ctx.device.makeBuffer(length: bufferLength,
                                           options: .storageModeShared)!
        
        let floats = buffer.contents().bindMemory(to: Float.self, capacity: Offset.total)
        
        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 1
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(buffer.gpuAddress, index: 0)
        
        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "scalar_linear_regression.residency"
        setDesc.initialCapacity = 1
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(buffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)
        
        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0
        
        func dispatch(W: [Float], b: [Float], x: [Float], y: [Float]) -> (loss: Float, yPred: [Float], dW: [Float], db: [Float]) {
            precondition(W.count == Self.kInDim * Self.kOutDim)
            precondition(b.count == Self.kOutDim)
            precondition(x.count == Self.kInDim)
            precondition(y.count == Self.kOutDim)
            
            for i in 0..<W.count { floats[Offset.W + i] = W[i] }
            for i in 0..<b.count { floats[Offset.b + i] = b[i] }
            for i in 0..<x.count { floats[Offset.x + i] = x[i] }
            for i in 0..<y.count { floats[Offset.y + i] = y[i] }
            floats[Offset.loss] = 0
            for i in 0..<Self.kOutDim { floats[Offset.yPred + i] = 0 }
            for i in 0..<Self.kOutDim * Self.kInDim { floats[Offset.dW + i] = 0 }
            for i in 0..<Self.kOutDim { floats[Offset.db + i] = 0 }
            
            let cmd = ctx.device.makeCommandBuffer()!
            cmd.beginCommandBuffer(allocator: ctx.allocator)
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setArgumentTable(argTable)
            enc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Int(Self.kOutDim), height: 1, depth: 1))
            enc.endEncoding()
            cmd.endCommandBuffer()

            ctx.queue.commit([cmd])
            signalValue += 1
            ctx.queue.signalEvent(event, value: signalValue)
            event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)
            
            let loss = floats[Offset.loss]
            let yPred = (0..<3).map { floats[Offset.yPred + $0] }
            let dW    = (0..<6).map { floats[Offset.dW + $0] }
            let db    = (0..<3).map { floats[Offset.db + $0] }
            return (loss, yPred, dW, db)
        }
        
        let W = [Float(1), 2, 3, 4, 5, 6] // row-major: W[i*kOutDim + j]
        let b: [Float] = [0.1, 0.2, 0.3]
        let x: [Float] = [0.5, -0.5]
        let y: [Float] = [0, 0, 0]

        //   y_pred[j] = W[0,j]*x[0] + W[1,j]*x[1] + b[j]
        //   loss      = mean_j(diff^2)                              with diff = y_pred - y
        //   dy[j]     = 2 * diff[j] / kOutDim
        //   db[j]     = dy[j]
        //   dW[i,j]   = dy[j] * x[i]
        let expectedLoss:  Float   = 1.6966667                    // = 5.09 / 3
        let expectedYPred: [Float] = [-1.4, -1.3, -1.2]
        let expectedDB:    [Float] = [-0.9333333, -0.8666667, -0.8]
        let expectedDW:    [Float] = [-0.4666667, -0.4333333, -0.4,
                                       0.4666667,  0.4333333,  0.4]

        let nominal = dispatch(W: W, b: b, x: x, y: y)
        print("nominal: loss=\(nominal.loss) yPred=\(nominal.yPred)")
        print("         db=\(nominal.db)")
        print("         dW=\(nominal.dW)")

        XCTAssertEqual(nominal.loss, expectedLoss, accuracy: 1e-5, "loss mismatch")
        for j in 0..<Self.kOutDim {
            XCTAssertEqual(nominal.yPred[j], expectedYPred[j], accuracy: 1e-5, "yPred[\(j)] mismatch")
            XCTAssertEqual(nominal.db[j],    expectedDB[j],    accuracy: 1e-5, "db[\(j)] mismatch")
        }
        for i in 0..<Self.kInDim {
            for j in 0..<Self.kOutDim {
                let idx = i * Self.kOutDim + j
                XCTAssertEqual(nominal.dW[idx], expectedDW[idx], accuracy: 1e-5, "dW[\(i),\(j)] mismatch")
            }
        }
        
        func gradientsAgree(estimate: Float, analytic: Float) -> Bool {
            let denom = max(abs(estimate), abs(analytic), 1e-3)
            return abs(estimate - analytic) / denom < 1e-3
        }
        
        // nudge only w by +h and -h
        let h: Float = 1e-3
        var WPert = W
        WPert[0] += h
        let WPlus = dispatch(W: WPert, b: b, x:x, y: y)
        WPert[0] -= 2.0 * h
        let WMinus = dispatch(W: WPert, b: b, x: x, y: y)
        
        let dW00Estimate = (WPlus.loss - WMinus.loss) / (2 * h)
        XCTAssert(gradientsAgree(estimate: dW00Estimate, analytic: nominal.dW[0]))
        
        var bPert = b
        bPert[1] += h
        let bPlus  = dispatch(W: W, b: bPert, x: x, y: y)
        bPert[1] -= 2 * h
        let bMinus = dispatch(W: W, b: bPert, x: x, y: y)
        let db1Estimate = (bPlus.loss - bMinus.loss) / (2 * h)
        XCTAssert(gradientsAgree(estimate: db1Estimate, analytic: nominal.db[1]))
    }
}
