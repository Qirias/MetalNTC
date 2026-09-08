// MIT License
//
// Copyright (c) 2026 Kyriakos Gavras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import Metal
import Foundation
import NTCCore
import NTCAssets
import NTCShared

#if NTC_DEBUG

func printPyramidLayout(sizes: [Int], offsets: [Int], slotFloats: [Int]) {
    print("pyramid layout: K_GRIDS=\(K_GRIDS)  F_PER_GRID=\(F_PER_GRID)")
    for i in 0..<K_GRIDS {
        print(String(format: "  pyramid[%d] %4dx%-4d x %d ch    offset=%-10d  floats=%d",
                     i, sizes[i], sizes[i], F_PER_GRID, offsets[i], slotFloats[i]))
    }
}

/// Decode every mip of the trained model back out, score it against the source
/// pyramid, print the PSNR table and write one LOD-atlas PNG per texture slot.
@MainActor
func reportQuality(ctx:              MetalContext,
                   textureSet:       TextureSet,
                   pyramidBuilder:   MipPyramidBuilder,
                   paramsBuffer:     any MTLBuffer,
                   stepConstsBuffer: any MTLBuffer,
                   srcW:             Int,
                   srcH:             Int,
                   kOut:             Int,
                   event:            any MTLSharedEvent,
                   signalValue:      inout UInt64) throws {
    let inferPso = ctx.makeComputePipelineState(function: "grid_mlp_infer")

    let outputBuffer = ctx.device.makeBuffer(length: srcH * srcW * kOut * MemoryLayout<Float>.stride,
                                             options: .storageModeShared)!
    outputBuffer.label = "NTC.inferOutput"
    let outPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: srcH * srcW * kOut)

    let setDesc = MTLResidencySetDescriptor()
    setDesc.label = "grid_mlp_infer.residency"
    setDesc.initialCapacity = 3
    let residencySet = try ctx.device.makeResidencySet(descriptor: setDesc)
    residencySet.addAllocation(outputBuffer)
    residencySet.addAllocation(paramsBuffer)
    residencySet.addAllocation(stepConstsBuffer)
    residencySet.commit()
    ctx.queue.addResidencySet(residencySet)
    defer { ctx.queue.removeResidencySet(residencySet) }

    let inferArgTable = ctx.makeArgumentTable(buffers: 3)
    inferArgTable.setAddress(paramsBuffer.gpuAddress,     index: 0)
    inferArgTable.setAddress(outputBuffer.gpuAddress,     index: 1)
    inferArgTable.setAddress(stepConstsBuffer.gpuAddress, index: 2)

    let stepConstsPtr = stepConstsBuffer.contents().bindMemory(to: StepConstants.self, capacity: 1)

    // The source mip being scored against, read back one slice at a time.
    let pyramidScratch = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: srcH * srcW)
    defer { pyramidScratch.deallocate() }

    func readPyramidSlice(lod: Int, slice: Int, outWL: Int, outHL: Int) {
        let region = MTLRegionMake2D(0, 0, outWL, outHL)
        pyramidBuilder.pyramidTexture.getBytes(pyramidScratch,
                                               bytesPerRow: outWL * MemoryLayout<SIMD4<Float>>.stride,
                                               bytesPerImage: outWL * outHL * MemoryLayout<SIMD4<Float>>.stride,
                                               from: region,
                                               mipmapLevel: lod,
                                               slice: slice)
    }

    func materialMse(_ s: TextureSlot, outWL: Int, outHL: Int) -> Double {
        var mse: Double = 0
        let base = s.channelOffset
        if s.channels == 3 {
            for i in 0..<(outWL * outHL) {
                let gt = pyramidScratch[i]
                let dr = Double(outPtr[i * kOut + base + 0] - gt.x)
                let dg = Double(outPtr[i * kOut + base + 1] - gt.y)
                let db = Double(outPtr[i * kOut + base + 2] - gt.z)
                mse += dr * dr + dg * dg + db * db
            }
        } else {
            for i in 0..<(outWL * outHL) {
                let gt = pyramidScratch[i]
                let d = Double(outPtr[i * kOut + base] - gt.x)
                mse += d * d
            }
        }
        return mse / Double(outWL * outHL * s.channels)
    }

    // mip 0 fills the left of the atlas, the rest stack down its right side
    let atlasW = srcW + srcW / 2
    let atlasH = srcH
    var atlases: [[Float]] = textureSet.slots.map { slot in
        [Float](repeating: 0, count: atlasW * atlasH * slot.channels)
    }

    var psnrTable = [[Double]](repeating: [Double](repeating: 0, count: pyramidBuilder.mipCount),
                               count: textureSet.slots.count)

    for lod in 0..<pyramidBuilder.mipCount {
        let outWL = max(srcW >> lod, 1)
        let outHL = max(srcH >> lod, 1)

        stepConstsPtr.pointee.inferLod = UInt32(lod)

        let inferCmd = ctx.device.makeCommandBuffer()!
        inferCmd.beginCommandBuffer(allocator: ctx.allocator)

        let inferEnc = inferCmd.makeComputeCommandEncoder()!
        inferEnc.setComputePipelineState(inferPso)
        inferEnc.setArgumentTable(inferArgTable)
        let tgx = (outWL + 15) / 16
        let tgy = (outHL + 15) / 16
        inferEnc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: tgx, height: tgy, depth: 1),
                                      threadsPerThreadgroup: MTLSize(width: 16,  height: 16, depth: 1))
        inferEnc.endEncoding()

        inferCmd.endCommandBuffer()
        ctx.queue.commit([inferCmd])
        signalValue += 1
        ctx.queue.signalEvent(event, value: signalValue)
        event.wait(untilSignaledValue: signalValue, timeoutMS: 10000)

        // collected here, printed as one semantic-per-row table after the loop
        for (si, slot) in textureSet.slots.enumerated() {
            readPyramidSlice(lod: lod, slice: slot.sliceIndex, outWL: outWL, outHL: outHL)
            let mse = materialMse(slot, outWL: outWL, outHL: outHL)
            if mse > 0 {
                psnrTable[si][lod] = 10.0 * log10(1.0 / mse)
            } else {
                psnrTable[si][lod] = Double.infinity
            }
        }

        let xOff: Int
        let yOff: Int
        if lod == 0 {
            xOff = 0
            yOff = 0
        } else {
            xOff = srcW
            // cumulative height of mips 1 to lod-1
            var cum = 0
            for k in 1..<lod {
                cum += max(srcH >> k, 1)
            }
            yOff = cum
        }

        // copy this mip into every slot's atlas
        for (si, slot) in textureSet.slots.enumerated() {
            for y in 0..<outHL {
                for x in 0..<outWL {
                    let srcIdx = (y * outWL + x) * kOut + slot.channelOffset
                    let dstIdx = ((yOff + y) * atlasW + (xOff + x)) * slot.channels
                    for c in 0..<slot.channels {
                        atlases[si][dstIdx + c] = outPtr[srcIdx + c]
                    }
                }
            }
        }
    }

    print("\nPSNR (dB) for \(textureSet.name), \(srcW)x\(srcH)")
    var header = String(repeating: " ", count: 14)
    for lod in 0..<pyramidBuilder.mipCount {
        header += String(format: "%7d", max(srcW >> lod, 1))
    }
    print(header)
    for (si, slot) in textureSet.slots.enumerated() {
        var line = slot.semantic.padding(toLength: 14, withPad: " ", startingAt: 0)
        for psnr in psnrTable[si] {
            line += String(format: "%7.2f", psnr)
        }
        print(line)
    }
    print("")

    let outDir = textureSet.manifestDir.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    for (si, slot) in textureSet.slots.enumerated() {
        let img = LoadedImage(pixels:   atlases[si],
                              height:   atlasH,
                              width:    atlasW,
                              channels: slot.channels)
        let name = "grid_mlp_\(textureSet.name)_\(slot.semantic.lowercased())_lod_atlas.png"
        try save_image(img, to: outDir.appendingPathComponent(name))
    }
}

#endif
