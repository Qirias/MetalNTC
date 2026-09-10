// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import Metal
import Foundation

public struct SPDConstants {
    public var numWorkgroups: UInt32 = 0
    public var mipCount:      UInt32 = 0
    public init() {}
}

public final class MipPyramidBuilder {
    public let pyramidTexture: any MTLTexture
    public let mipCount:       Int
    public let mipSizes:       [(w: Int, h: Int)]
    public let srcW:           Int
    public let srcH:           Int

    private let pso:           any MTLComputePipelineState
    private let constsBuffer:  any MTLBuffer
    private let atomicBuffer:  any MTLBuffer
    private let argTable:      any MTL4ArgumentTable
    private let sourceTexture: any MTLTexture
    private let wgX:           Int
    private let wgY:           Int

    public init(ctx: MetalContext, srcW: Int, srcH: Int, sourceTexture: any MTLTexture) {
        precondition(srcW > 0 && srcH > 0)
        precondition(sourceTexture.width == srcW && sourceTexture.height == srcH,
                     "sourceTexture is \(sourceTexture.width)x\(sourceTexture.height), expected \(srcW)x\(srcH)")

        self.srcW          = srcW
        self.srcH          = srcH
        self.sourceTexture = sourceTexture

        self.mipCount = Int(log2(Double(max(srcW, srcH)))) + 1
        self.mipSizes = (0..<mipCount).map { level in
            (w: max(srcW >> level, 1), h: max(srcH >> level, 1))
        }

        let desc = MTLTextureDescriptor()
        desc.textureType      = .type2DArray
        desc.pixelFormat      = .rgba32Float
        desc.width            = srcW
        desc.height           = srcH
        desc.arrayLength      = sourceTexture.arrayLength
        desc.mipmapLevelCount = mipCount
        desc.usage            = [.shaderRead, .shaderWrite]
        desc.storageMode      = .shared
        self.pyramidTexture       = ctx.device.makeTexture(descriptor: desc)!
        self.pyramidTexture.label = "SPD.pyramid"

        self.pso = ctx.makeComputePipelineState(function: "downsample_2x2")

        self.wgX = (srcW + 63) / 64
        self.wgY = (srcH + 63) / 64

        var consts = SPDConstants()
        consts.numWorkgroups = UInt32(wgX * wgY)
        consts.mipCount      = UInt32(mipCount)

        self.constsBuffer = ctx.device.makeBuffer(length: MemoryLayout<SPDConstants>.stride,
                                                  options: .storageModeShared)!
        self.constsBuffer.label = "SPD.consts"
        constsBuffer.contents().bindMemory(to: SPDConstants.self, capacity: 1).pointee = consts

        self.atomicBuffer = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.stride,
                                                  options: .storageModeShared)!
        self.atomicBuffer.label = "SPD.workgroupCounter"
        memset(atomicBuffer.contents(), 0, MemoryLayout<UInt32>.stride)

        self.argTable = ctx.makeArgumentTable(buffers: 2, textures: 2)
        argTable.setTexture(sourceTexture.gpuResourceID,  index: 0)
        argTable.setTexture(pyramidTexture.gpuResourceID, index: 1)
        argTable.setAddress(atomicBuffer.gpuAddress,      index: 0)
        argTable.setAddress(constsBuffer.gpuAddress,      index: 1)
    }

    public var residencyAllocations: [any MTLResource] {
        return [sourceTexture, pyramidTexture, constsBuffer, atomicBuffer]
    }

    public func encode(into cmd: any MTL4CommandBuffer) {
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setArgumentTable(argTable)
        enc.dispatchThreadgroups(threadgroupsPerGrid:   MTLSize(width: wgX, height: wgY, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16,  height: 16,  depth: 1))
        enc.endEncoding()
    }
}
