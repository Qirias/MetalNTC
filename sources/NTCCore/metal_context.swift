import Metal
import Foundation

public final class MetalContext {
    public let device: any MTLDevice
    public let queue: any MTL4CommandQueue
    public let allocator: any MTL4CommandAllocator
    public let compiler: any MTL4Compiler
    public let library: any MTLLibrary
    
    /// Create a MetalContext that loads its default.metallib from the
    /// given bundle.
    ///
    /// The bundle must be one whose SwiftPM target declares `resources`
    /// pointing at a directory of `.metal` files (so Xcode emits a
    /// `default.metallib` into that bundle). Callers normally pass
    /// `Bundle.module` from inside the target that owns the shaders.
    public init(bundle: Bundle) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("This device does not support Metal")
        }
        guard device.supportsFamily(.metal4)
        else {
            fatalError("This device does not support Metal 4")
        }
        self.device = device

        guard let queue = device.makeMTL4CommandQueue() else {
            fatalError("Failed to create command MTL4CommandQueue")
        }
        self.queue = queue

        guard let allocator = device.makeCommandAllocator() else {
            fatalError("Failed to create MTLCommandAllocator")
        }
        self.allocator = allocator

        let compilerDescriptor = MTL4CompilerDescriptor()
        do {
            self.compiler = try device.makeCompiler(descriptor: compilerDescriptor)
        } catch {
            fatalError("Failed to create MTL4Compiler: \(error)")
        }

        do {
            self.library = try device.makeDefaultLibrary(bundle: bundle)
        } catch {
            fatalError("Failed to create MTLLibrary: \(error)")
        }
    }

    public func makeComputePipelineState(function name: String) throws -> any MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            fatalError("Failed to create MTLFunction")
        }

        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = name
        functionDescriptor.library = library

        let pipelineDescriptor = MTL4ComputePipelineDescriptor()
        pipelineDescriptor.computeFunctionDescriptor = functionDescriptor
        
        return try compiler.makeComputePipelineState(descriptor: pipelineDescriptor)
    }
}
