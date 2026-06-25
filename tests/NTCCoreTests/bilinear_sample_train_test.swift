import XCTest
import Metal
@testable import NTCCore

private let kGridH     = Shapes.gridH
private let kGridW     = Shapes.gridW
private let kGridCh    = Shapes.gridCh
private let kGridTotal = Shapes.gridTotal // 192
private let kQueries   = Shapes.kQueries

private enum SampleOffset {
    static let ix   = 0
    static let iy   = 1
    static let y    = 2
    static let out  = 5
    static let loss = 8
    static let stride = 9
}

final class BilinearSampleTrainTest: XCTestCase {

    // forward + backward, single query, no atomics
    func testBilinearForwardBackwardSingleQuery() throws {
        let ctx = try MetalContext(bundle: .module)
        let pso = try ctx.makeComputePipelineState(function: "bilinear_sample_train")

        let gridBuffer = ctx.device.makeBuffer(length: kGridTotal * MemoryLayout<Float>.stride,
                                               options: .storageModeShared)!
        let grid = gridBuffer.contents().bindMemory(to: Float.self, capacity: kGridTotal)

        let gradBuffer = ctx.device.makeBuffer(length: kGridTotal * MemoryLayout<Float>.stride,
                                               options: .storageModeShared)!
        let gridGrad = gradBuffer.contents().bindMemory(to: Float.self, capacity: kGridTotal)

        let sampleBuffer = ctx.device.makeBuffer(length: SampleOffset.stride * MemoryLayout<Float>.stride,
                                                 options: .storageModeShared)!
        let sample = sampleBuffer.contents().bindMemory(to: Float.self, capacity: SampleOffset.stride)

        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "bilinear_sample_train.residency"
        setDesc.initialCapacity = 3
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(gridBuffer)
        residencySet.addAllocation(gradBuffer)
        residencySet.addAllocation(sampleBuffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)

        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 3
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(gridBuffer.gpuAddress,   index: 0)
        argTable.setAddress(gradBuffer.gpuAddress,   index: 1)
        argTable.setAddress(sampleBuffer.gpuAddress, index: 2)

        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0


        for i in 0..<kGridTotal { grid[i] = Float(i) }

        sample[SampleOffset.ix]     = 0.3
        sample[SampleOffset.iy]     = 0.7
        sample[SampleOffset.y + 0]  = 18.0
        sample[SampleOffset.y + 1]  = 19.0
        sample[SampleOffset.y + 2]  = 20.0

        func dispatch() {
            for i in 0..<kGridTotal { gridGrad[i] = 0 }

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
        }

        // nominal
        dispatch()

        //   iy=0.7, ix=0.3
        //   iy0=floor(iy), ix0=floor(ix)
        //   fy=0.7, fx=0.3)
        //   w00 = (1-fx)(1-fy) = 0.7*0.3 = 0.21   corner (0,0)
        //   w01 = fx*(1-fy)    = 0.3*0.3 = 0.09   corner (0,1)
        //   w10 = (1-fx)*fy    = 0.7*0.7 = 0.49   corner (1,0)
        //   w11 = fx*fy        = 0.3*0.7 = 0.21   corner (1,1)
        //   grid[0,0,*] = 0,1,2     grid[0,1,*] = 3,4,5
        //   grid[1,0,*] = 24,25,26  grid[1,1,*] = 27,28,29
        //
        //   out[0] = 0.21*0  + 0.09*3 + 0.49*24 + 0.21*27 = 17.70
        //   out[1] = 0.21*1  + 0.09*4 + 0.49*25 + 0.21*28 = 18.70
        //   out[2] = 0.21*2  + 0.09*5 + 0.49*26 + 0.21*29 = 19.70
        //   y      = [18, 19, 20]  → diff = [-0.30, -0.30, -0.30]
        //   loss   = (0.09 + 0.09 + 0.09) = 0.27
        XCTAssertEqual(sample[SampleOffset.out + 0], 17.70, accuracy: 1e-4, "out[0] mismatch")
        XCTAssertEqual(sample[SampleOffset.out + 1], 18.70, accuracy: 1e-4, "out[1] mismatch")
        XCTAssertEqual(sample[SampleOffset.out + 2], 19.70, accuracy: 1e-4, "out[2] mismatch")
        XCTAssertEqual(sample[SampleOffset.loss],     0.27,  accuracy: 1e-5, "loss mismatch")


        //   dg[0,0,*] = 0.21 * -0.60 = -0.126
        //   dg[0,1,*] = 0.09 * -0.60 = -0.054
        //   dg[1,0,*] = 0.49 * -0.60 = -0.294
        //   dg[1,1,*] = 0.21 * -0.60 = -0.126
        func gradAt(_ r: Int, _ c: Int, _ ch: Int) -> Float {
            gridGrad[(r * kGridW + c) * kGridCh + ch]
        }

        let analytic: Float = 1e-5
        for ch in 0..<kGridCh {
            XCTAssertEqual(gradAt(0, 0, ch), -0.126, accuracy: analytic, "dg[0,0,\(ch)] mismatch")
            XCTAssertEqual(gradAt(0, 1, ch), -0.054, accuracy: analytic, "dg[0,1,\(ch)] mismatch")
            XCTAssertEqual(gradAt(1, 0, ch), -0.294, accuracy: analytic, "dg[1,0,\(ch)] mismatch")
            XCTAssertEqual(gradAt(1, 1, ch), -0.126, accuracy: analytic, "dg[1,1,\(ch)] mismatch")
        }
    }

    // test atomics by using multiple queries that overlap
    func testBilinearBackwardAtomicHazard() throws {
        let ctx = try MetalContext(bundle: .module)
        let pso = try ctx.makeComputePipelineState(function: "bilinear_sample_train_atomic")

        let combinedFloats = 2 * kGridTotal
        let gridBuffer = ctx.device.makeBuffer(length: combinedFloats * MemoryLayout<Float>.stride,
                                               options: .storageModeShared)!
        let gridFloats: UnsafeMutablePointer<Float> = gridBuffer.contents().bindMemory(to: Float.self, capacity: combinedFloats)
        let gridGradInt: UnsafeMutablePointer<Int32> = gridBuffer.contents().assumingMemoryBound(to: Int32.self)

        let samplesCount = kQueries * SampleOffset.stride
        let samplesBuffer = ctx.device.makeBuffer(length: samplesCount * MemoryLayout<Float>.stride,
                                                  options: .storageModeShared)!
        let samples = samplesBuffer.contents().bindMemory(to: Float.self, capacity: samplesCount)

        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "bilinear_sample_train_atomic.residency"
        setDesc.initialCapacity = 2
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(gridBuffer)
        residencySet.addAllocation(samplesBuffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)

        let argDesc = MTL4ArgumentTableDescriptor()
        argDesc.maxBufferBindCount = 2
        let argTable = try ctx.device.makeArgumentTable(descriptor: argDesc)
        argTable.setAddress(gridBuffer.gpuAddress,    index: 0)
        argTable.setAddress(samplesBuffer.gpuAddress, index: 1)

        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        for i in 0..<kGridTotal { gridFloats[i] = Float(i) }
        
        for i in kGridTotal..<combinedFloats { gridGradInt[i] = 0 }

        // Q0:(fy=fx=0)         -> (0,0) only,           w00=1
        // Q1: fy=fx=0.5        -> (2,1)(2,2)(3,1)(3,2), all w=0.25
        // Q2: fy=0.2, fx=0.3   -> (1,2)(1,3)(2,2)(2,3), w=0.56/0.24/0.14/0.06
        // Q3: fy=0.7, fx=0.6   -> (1,2)(1,3)(2,2)(2,3), w=0.12/0.18/0.28/0.42
        let queries: [(ix: Float, iy: Float)] = [(ix: 0.0, iy: 0.0),
                                                 (ix: 1.5, iy: 2.5),
                                                 (ix: 2.3, iy: 1.2),
                                                 (ix: 2.6, iy: 1.7)]
        precondition(queries.count == kQueries)

        //   Q0: [grid[0,0,0], grid[0,0,1], grid[0,0,2]]                        = [0, 1, 2]
        //   Q1: 0.25*(g[2,1]   + g[2,2]        + g[3,1]        + g[3,2])       = [64.5, 65.5, 66.5]
        //   Q2: 0.56*g[1,2]    + 0.24*g[1,3]   + 0.14*g[2,2]   + 0.06*g[2,3]   = [35.7, 36.7, 37.7]
        //   Q3: 0.12*g[1,2]    + 0.18*g[1,3]   + 0.28*g[2,2]   + 0.42*g[2,3]   = [48.6, 49.6, 50.6]
        let expectedOut: [[Float]] = [[ 0.0,  1.0,  2.0],
                                      [64.5, 65.5, 66.5],
                                      [35.7, 36.7, 37.7],
                                      [48.6, 49.6, 50.6]]

        // set y so diff = -1 per channel per query and d_out = -2
        for q in 0..<kQueries {
            let base = q * SampleOffset.stride
            samples[base + SampleOffset.ix] = queries[q].ix
            samples[base + SampleOffset.iy] = queries[q].iy
            for ch in 0..<kGridCh {
                samples[base + SampleOffset.y + ch] = expectedOut[q][ch] + 1.0
            }
        }

        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setArgumentTable(argTable)
        enc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,        height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: kQueries, height: 1, depth: 1))
        enc.endEncoding()
        cmd.endCommandBuffer()

        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        let scale: Float = Float(1 << 18)
        func grad(_ r: Int, _ c: Int, _ ch: Int) -> Float {
            Float(gridGradInt[kGridTotal + (r * kGridW + c) * kGridCh + ch]) / scale
        }

        for q in 0..<kQueries {
            let base = q * SampleOffset.stride
            for ch in 0..<kGridCh {
                XCTAssertEqual(samples[base + SampleOffset.out + ch], expectedOut[q][ch], accuracy: 1e-4, "Q\(q) out[\(ch)] mismatch")
            }
            XCTAssertEqual(samples[base + SampleOffset.loss], Float(kGridCh), accuracy: 1e-4, "Q\(q) loss mismatch")
        }

        //   (0,0): Q0 w00=1                              -> -2.0
        //   (2,1): Q1 w01=0.25                           -> -0.5
        //   (3,1): Q1 w10=0.25                           -> -0.5
        //   (3,2): Q1 w11=0.25                           -> -0.5
        //   (1,2): Q2 w00=0.56 + Q3 w00=0.12 = 0.68      -> -1.36   (2-way)
        //   (1,3): Q2 w01=0.24 + Q3 w01=0.18 = 0.42      -> -0.84   (2-way)
        //   (2,2): Q1 0.25 + Q2 0.14 + Q3 0.28 = 0.67    -> -1.34   (3-way)
        //   (2,3): Q2 w11=0.06 + Q3 w11=0.42 = 0.48      -> -0.96   (2-way)
        let expectedGrad: [(r: Int, c: Int, value: Float)] = [(0, 0, -2.00),
                                                              (2, 1, -0.50),
                                                              (3, 1, -0.50),
                                                              (3, 2, -0.50),
                                                              (1, 2, -1.36), // Q2 + Q3, hazard
                                                              (1, 3, -0.84), // Q2 + Q3, hazard
                                                              (2, 2, -1.34), // Q1 + Q2 + Q3, hazard
                                                              (2, 3, -0.96)] // Q2 + Q3, hazard
        
        let gradTol: Float = 1e-4
        for (r, c, expected) in expectedGrad {
            for ch in 0..<kGridCh {
                XCTAssertEqual(grad(r, c, ch), expected, accuracy: gradTol, "grad(\(r),\(c),\(ch)) mismatch")
            }
        }
    }
}
