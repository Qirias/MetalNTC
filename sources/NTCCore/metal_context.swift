// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import Metal
import Foundation

public final class MetalContext {
    public let device:    any MTLDevice
    public let queue:     any MTL4CommandQueue
    public let allocator: any MTL4CommandAllocator
    public let compiler:  any MTL4Compiler
    public let library:   any MTLLibrary

    /// `bundle` must belong to the SwiftPM target that owns the `.metal`
    /// sources, so the `default.metallib` loaded is the one built from them.
    /// Callers pass `NTCCoreResources.bundle`.
    public init(bundle: Bundle) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("no Metal device on this machine")
        }
        guard device.supportsFamily(.metal4) else {
            fatalError("\(device.name) does not support Metal 4")
        }
        self.device = device

        guard let queue = device.makeMTL4CommandQueue() else {
            fatalError("failed to create MTL4CommandQueue")
        }
        self.queue = queue

        guard let allocator = device.makeCommandAllocator() else {
            fatalError("failed to create MTL4CommandAllocator")
        }
        self.allocator = allocator

        do {
            self.compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
        } catch {
            fatalError("failed to create MTL4Compiler: \(error)")
        }

        do {
            self.library = try device.makeDefaultLibrary(bundle: bundle)
        } catch {
            fatalError("failed to load default.metallib from \(bundle.bundlePath): \(error)")
        }
    }

    public func makeComputePipelineState(function name: String) -> any MTLComputePipelineState {
        guard library.makeFunction(name: name) != nil else {
            fatalError("no function named '\(name)' in default.metallib")
        }

        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name    = name
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor

        do {
            return try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("failed to compile a compute pipeline for '\(name)': \(error)")
        }
    }

    public func makeArgumentTable(buffers: Int, textures: Int = 0) -> any MTL4ArgumentTable {
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount  = buffers
        descriptor.maxTextureBindCount = textures

        do {
            return try device.makeArgumentTable(descriptor: descriptor)
        } catch {
            fatalError("failed to create an argument table for \(buffers) buffers and \(textures) textures: \(error)")
        }
    }
}
