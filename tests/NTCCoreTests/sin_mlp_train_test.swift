import XCTest
import Metal
@testable import NTCCore

private let kIn     = 1
private let kHidden = 4
private let kOut    = 2
private let kBatch   = 64

private enum ParamsOffset {
    static let w1 = 0
    static let b1 = w1 + kIn * kHidden
    static let w2 = b1 + kHidden
    static let b2 = w2 + kHidden * kOut
    static let floatTotal = b2 + kOut
    
    static let dW1 = floatTotal
    static let db1 = dW1 + kIn * kHidden
    static let dW2 = db1 + kHidden
    static let db2 = dW2 + kHidden * kOut
    static let total = db2 + kOut
}

private enum SampleOffset {
    static let x   = 0
    static let y   = x + kIn
    static let out = y + kOut
    static let loss = out + kOut
    static let stride = loss + 1
}

final class SinMLPTrainTest: XCTestCase {
    // single step unit test (train + sgd)
    // assert each stage with known values
    func testMLPTraining() throws {
        let ctx = try MetalContext(bundle: .module)
        let trainPso = try ctx.makeComputePipelineState(function: "sin_mlp_train")
        let sgdPso = try ctx.makeComputePipelineState(function: "sgd_step")
        
        // params buffer
        let paramsBufferLength = ParamsOffset.total * MemoryLayout<Float>.stride
        let paramsBuffer = ctx.device.makeBuffer(length: paramsBufferLength, options:  .storageModeShared)!
        
        let paramFloats: UnsafeMutablePointer<Float> = paramsBuffer.contents().bindMemory(to: Float.self, capacity: ParamsOffset.total)
        let paramGrads: UnsafeMutablePointer<Int32> = paramsBuffer.contents().assumingMemoryBound(to: Int32.self)
        
        // samples buffer
        let samplesBufferLength = kBatch * SampleOffset.stride * MemoryLayout<Float>.stride
        let samplesBuffer = ctx.device.makeBuffer(length: samplesBufferLength, options: .storageModeShared)!
        let samplesFloats = samplesBuffer.contents().bindMemory(to: Float.self, capacity: kBatch * SampleOffset.stride)
        
        // LR constants buffer
        let lrBuffer = ctx.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        let lr_io = lrBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        lr_io.pointee = 0.05
        
        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "sin_mlp_train.residency"
        setDesc.initialCapacity = 3
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(paramsBuffer)
        residencySet.addAllocation(samplesBuffer)
        residencySet.addAllocation(lrBuffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)
        
        // train argument table
        let trainArgDesc = MTL4ArgumentTableDescriptor()
        trainArgDesc.maxBufferBindCount = 2
        let trainArgTable = try ctx.device.makeArgumentTable(descriptor: trainArgDesc)
        trainArgTable.setAddress(paramsBuffer.gpuAddress, index: 0)
        trainArgTable.setAddress(samplesBuffer.gpuAddress, index: 1)
        
        // sgd argument table
        let sgdArgDesc = MTL4ArgumentTableDescriptor()
        sgdArgDesc.maxBufferBindCount = 2
        let sgdArgTable = try ctx.device.makeArgumentTable(descriptor: sgdArgDesc)
        sgdArgTable.setAddress(paramsBuffer.gpuAddress, index: 0)
        sgdArgTable.setAddress(lrBuffer.gpuAddress, index: 1)
        
        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        // init params
        for i in 0..<ParamsOffset.total { paramFloats[i] = 0 }

        // W1 [K_IN=1, K_HIDDEN=4]
        paramFloats[ParamsOffset.w1 + 0] =  0.3
        paramFloats[ParamsOffset.w1 + 1] = -0.5
        paramFloats[ParamsOffset.w1 + 2] =  0.7
        paramFloats[ParamsOffset.w1 + 3] = -0.2
        // b1 left zero
        // W2 [K_HIDDEN=4, K_OUT=2]
        paramFloats[ParamsOffset.w2 + 0] =  0.4; paramFloats[ParamsOffset.w2 + 1] = -0.3
        paramFloats[ParamsOffset.w2 + 2] =  0.1; paramFloats[ParamsOffset.w2 + 3] =  0.5
        paramFloats[ParamsOffset.w2 + 4] = -0.2; paramFloats[ParamsOffset.w2 + 5] =  0.6
        paramFloats[ParamsOffset.w2 + 6] =  0.3; paramFloats[ParamsOffset.w2 + 7] = -0.4
        // b2 left zero

        // fill all 64 samples with (sin(0.5), cos(0.5))
        for s in 0..<kBatch {
            let base = s * SampleOffset.stride
            samplesFloats[base + SampleOffset.x]     = 0.5
            samplesFloats[base + SampleOffset.y]     = Float(sin(0.5))
            samplesFloats[base + SampleOffset.y + 1] = Float(cos(0.5))
            samplesFloats[base + SampleOffset.out]     = 0
            samplesFloats[base + SampleOffset.out + 1] = 0
            samplesFloats[base + SampleOffset.loss]    = 0
        }

        // dispatch
        let cmd = ctx.device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: ctx.allocator)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(trainPso)
        enc.setArgumentTable(trainArgTable)
        enc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,      height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: kBatch, height: 1, depth: 1))
        enc.endEncoding()
        cmd.endCommandBuffer()

        ctx.queue.commit([cmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        // forward assertions
        for s in 0..<kBatch {
            let base = s * SampleOffset.stride
            XCTAssertEqual(samplesFloats[base + SampleOffset.out],     -0.034062, accuracy: 1e-5, "out[0] mismatch at sample \(s)")
            XCTAssertEqual(samplesFloats[base + SampleOffset.out + 1],  0.063500, accuracy: 1e-5, "out[1] mismatch at sample \(s)")
            XCTAssertEqual(samplesFloats[base + SampleOffset.loss],     0.463200, accuracy: 1e-5, "loss mismatch at sample \(s)")
        }

        // grad assertions BATCH*(grad/BATCH)
        let scale: Float = Float(1 << 14)
        func grad(_ idx: Int) -> Float { Float(paramGrads[idx]) / scale }

        // each thread converts its float grad contribution (int(rint(grad * SCALE))
        // rint rounds to the nearest, so each thread introduces at most 0.5 of error (0.5 / SCALE)
        // stacking across the batch: K_BATCH * 0.5 / SCALE
        // 64 * 0.5 / 16384 = 1 / 512 = 1.95e-3
        let gradTol: Float = 2e-3
        XCTAssertEqual(grad(ParamsOffset.dW1 + 2), -0.130191, accuracy: gradTol, "dW1[0,2] mismatch")
        XCTAssertEqual(grad(ParamsOffset.db1 + 1), -0.171896, accuracy: gradTol, "db1[1] mismatch")
        XCTAssertEqual(grad(ParamsOffset.dW2 + 4), -0.105586, accuracy: gradTol, "dW2[2,0] mismatch")
        XCTAssertEqual(grad(ParamsOffset.db2 + 1), -0.814083, accuracy: gradTol, "db2[1] mismatch")

        // sgd_step
        // capture param values before the update so we can assert the nudge
        let w1_02_pre: Float = paramFloats[ParamsOffset.w1 + 2]   //  0.7
        let b1_1_pre:  Float = paramFloats[ParamsOffset.b1 + 1]   //  0.0
        let w2_20_pre: Float = paramFloats[ParamsOffset.w2 + 4]   // -0.2
        let b2_1_pre:  Float = paramFloats[ParamsOffset.b2 + 1]   //  0.0
        let lr: Float = lr_io.pointee

        let sgdCmd = ctx.device.makeCommandBuffer()!
        sgdCmd.beginCommandBuffer(allocator: ctx.allocator)
        let sgdEnc = sgdCmd.makeComputeCommandEncoder()!
        sgdEnc.setComputePipelineState(sgdPso)
        sgdEnc.setArgumentTable(sgdArgTable)
        sgdEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,  height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))   // 1 SIMD group, 18 active
        sgdEnc.endEncoding()
        sgdCmd.endCommandBuffer()

        ctx.queue.commit([sgdCmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)

        // params updated by -lr * grad. lr scales the 2e-3 grad quant down to ~1e-4.
        let paramTol: Float = 2e-4
        XCTAssertEqual(paramFloats[ParamsOffset.w1 + 2], w1_02_pre - lr * -0.130191, accuracy: paramTol, "W1[0,2] not nudged")
        XCTAssertEqual(paramFloats[ParamsOffset.b1 + 1], b1_1_pre  - lr * -0.171896, accuracy: paramTol, "b1[1] not nudged")
        XCTAssertEqual(paramFloats[ParamsOffset.w2 + 4], w2_20_pre - lr * -0.105586, accuracy: paramTol, "W2[2,0] not nudged")
        XCTAssertEqual(paramFloats[ParamsOffset.b2 + 1], b2_1_pre  - lr * -0.814083, accuracy: paramTol, "b2[1] not nudged")

        for i in ParamsOffset.floatTotal..<ParamsOffset.total {
            XCTAssertEqual(paramGrads[i], 0, "grad slot \(i) not cleared by sgd_step")
        }
    }
    
    // multi step unit test (train + sgd)
    func testMLPTrainingLoop() throws {
        let ctx = try MetalContext(bundle: .module)
        let trainPso = try ctx.makeComputePipelineState(function: "sin_mlp_train")
        let sgdPso   = try ctx.makeComputePipelineState(function: "sgd_step")

        let paramsBufferLength = ParamsOffset.total * MemoryLayout<Float>.stride
        let paramsBuffer = ctx.device.makeBuffer(length: paramsBufferLength,
                                  options: .storageModeShared)!
        let paramFloats: UnsafeMutablePointer<Float> = paramsBuffer.contents().bindMemory(to: Float.self, capacity: ParamsOffset.total)
        let paramGrads: UnsafeMutablePointer<Int32> = paramsBuffer.contents().assumingMemoryBound(to: Int32.self)

        let samplesBufferLength = kBatch * SampleOffset.stride * MemoryLayout<Float>.stride
        let samplesBuffer = ctx.device.makeBuffer(length: samplesBufferLength, options: .storageModeShared)!
        let samplesFloats = samplesBuffer.contents().bindMemory(to: Float.self, capacity: kBatch * SampleOffset.stride)

        let lrBuffer = ctx.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        let lr_io = lrBuffer.contents().bindMemory(to: Float.self, capacity: 1)
        lr_io.pointee = 0.05

        let setDesc = MTLResidencySetDescriptor()
        setDesc.label = "sin_mlp_train_loop.residency"
        setDesc.initialCapacity = 3
        let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
        residencySet.addAllocation(paramsBuffer)
        residencySet.addAllocation(samplesBuffer)
        residencySet.addAllocation(lrBuffer)
        residencySet.commit()
        ctx.queue.addResidencySet(residencySet)

        let trainArgDesc = MTL4ArgumentTableDescriptor()
        trainArgDesc.maxBufferBindCount = 2
        let trainArgTable = try ctx.device.makeArgumentTable(descriptor: trainArgDesc)
        trainArgTable.setAddress(paramsBuffer.gpuAddress,  index: 0)
        trainArgTable.setAddress(samplesBuffer.gpuAddress, index: 1)

        let sgdArgDesc = MTL4ArgumentTableDescriptor()
        sgdArgDesc.maxBufferBindCount = 2
        let sgdArgTable = try ctx.device.makeArgumentTable(descriptor: sgdArgDesc)
        sgdArgTable.setAddress(paramsBuffer.gpuAddress, index: 0)
        sgdArgTable.setAddress(lrBuffer.gpuAddress,     index: 1)

        let event = ctx.device.makeSharedEvent()!
        var signalValue: UInt64 = 0

        // init params
        for i in 0..<ParamsOffset.total { paramFloats[i] = 0 }

        for i in ParamsOffset.w1..<ParamsOffset.b1 { paramFloats[i] = Float.random(in: -0.3...0.3) }
        for i in ParamsOffset.w2..<ParamsOffset.b2 { paramFloats[i] = Float.random(in: -0.3...0.3) }
        
        
        let nSteps   = 2000
        let logEvery = 50
        var lossLog: [Float] = []
        
        for step in 0..<nSteps {
            for s in 0..<kBatch {
                let base = s * SampleOffset.stride
                let x = Float.random(in: 0 ..< 2 * .pi)
                samplesFloats[base + SampleOffset.x] = x
                samplesFloats[base + SampleOffset.y] = sin(x)
                samplesFloats[base + SampleOffset.y + 1] = cos(x)
            }
            
            if step == 1000 {
                lr_io.pointee = 0.01
            }
            
            let cmd = ctx.device.makeCommandBuffer()!
            cmd.beginCommandBuffer(allocator: ctx.allocator)
            let trainEnc = cmd.makeComputeCommandEncoder()!
            trainEnc.setComputePipelineState(trainPso)
            trainEnc.setArgumentTable(trainArgTable)
            trainEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,      height: 1, depth: 1),
                                          threadsPerThreadgroup: MTLSize(width: kBatch, height: 1, depth: 1))
            trainEnc.endEncoding()

            let sgdEnc = cmd.makeComputeCommandEncoder()!
            sgdEnc.setComputePipelineState(sgdPso)
            sgdEnc.setArgumentTable(sgdArgTable)
            sgdEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: 1,  height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            sgdEnc.endEncoding()
            cmd.endCommandBuffer()

            ctx.queue.commit([cmd])
            signalValue += 1
            ctx.queue.signalEvent(event, value: signalValue)
            
            if step % logEvery == 0 {
                event.wait(untilSignaledValue: signalValue, timeoutMS: 1000)
                var sumLoss: Float = 0
                for s in 0..<kBatch {
                    sumLoss += samplesFloats[s * SampleOffset.stride + SampleOffset.loss]
                }
                let mean = sumLoss / Float(kBatch)
                lossLog.append(mean)
                print("step \(step)\tmean loss = \(mean)")
            }
        }
    }
}
