//
//  MetalCompute.swift
//
//  Part of MetalCompute
//
//  Interface for passing compute tasks to Metal
//

import Foundation
import Metal

/// Entry point of MetalCompute: holds the Metal device, the command queue and the library
/// with the kernel functions.
///
/// Create one instance and keep it, e.g. as a static property. Creating it is expensive,
/// running kernels is cheap. The instance is thread safe.
///
/// ```swift
/// let metal = try MetalCompute()                       // kernels from the app's default library
/// let kernel = try metal.kernel("mandelbrot")
/// let result: [UInt32] = try await kernel.run(points, parameters, count: points.count)
/// ```
public final class MetalCompute: @unchecked Sendable {

    /// The Metal device
    public let device: any MTLDevice

    /// Command queue used for all kernel runs
    public let commandQueue: any MTLCommandQueue

    /// Library containing the kernel functions
    public let library: any MTLLibrary

    // Cache of created kernels (pipeline creation is expensive)
    private var kernels: [String: ComputeKernel] = [:]
    private let lock = NSLock()

    /// Creates a MetalCompute object with an existing library
    /// - Parameters:
    ///   - library: Metal library containing the kernel functions
    ///   - device: Metal device, default is the system default device
    public init(library: any MTLLibrary, device: (any MTLDevice)? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalComputeError.noDevice
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalComputeError.commandCreationFailed("command queue")
        }
        self.device = device
        self.commandQueue = commandQueue
        self.library = library
    }

    /// Creates a MetalCompute object with the default Metal library of a bundle.
    /// The kernel functions are the .metal files compiled into this bundle.
    /// - Parameters:
    ///   - bundle: Bundle containing the compiled Metal library, default is the main (app) bundle
    ///   - device: Metal device, default is the system default device
    public convenience init(bundle: Bundle = .main, device: (any MTLDevice)? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalComputeError.noDevice
        }
        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: bundle)
        }
        catch {
            throw MetalComputeError.libraryNotAvailable(underlying: error)
        }
        try self.init(library: library, device: device)
    }

    /// Creates a MetalCompute object by compiling Metal source code at runtime.
    /// By default fast math is disabled to keep IEEE conform floating point results.
    /// - Parameters:
    ///   - source: Metal source code with the kernel functions
    ///   - options: Compile options, default: safe math mode
    ///   - device: Metal device, default is the system default device
    public convenience init(source: String, options: MTLCompileOptions? = nil, device: (any MTLDevice)? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw MetalComputeError.noDevice
        }
        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: options ?? MetalCompute.safeMathOptions())
        }
        catch {
            throw MetalComputeError.libraryNotAvailable(underlying: error)
        }
        try self.init(library: library, device: device)
    }

    /// Compile options with IEEE conform floating point math
    public static func safeMathOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        if #available(macOS 15.0, iOS 18.0, *) {
            options.mathMode = .safe
        }
        else {
            options.fastMathEnabled = false
        }
        return options
    }

    /// Returns the kernel with the given function name.
    /// Kernels are created once and cached.
    /// - Parameter name: Name of the kernel function in the Metal library
    public func kernel(_ name: String) throws -> ComputeKernel {
        lock.lock()
        defer { lock.unlock() }

        if let kernel = kernels[name] {
            return kernel
        }
        guard let function = library.makeFunction(name: name) else {
            throw MetalComputeError.functionNotFound(name)
        }
        let pipeline: any MTLComputePipelineState
        do {
            pipeline = try device.makeComputePipelineState(function: function)
        }
        catch {
            throw MetalComputeError.pipelineCreationFailed(name, underlying: error)
        }
        let kernel = ComputeKernel(name: name, pipeline: pipeline, device: device, commandQueue: commandQueue)
        kernels[name] = kernel
        return kernel
    }

    /// Creates a reusable buffer with `count` elements, initialized with zero bytes
    public func makeBuffer<T>(count: Int, of type: T.Type = T.self) throws -> MetalBuffer<T> {
        try MetalBuffer<T>(device: device, count: count)
    }

    /// Creates a reusable buffer with `count` elements, all set to `value`
    public func makeBuffer<T>(count: Int, repeating value: T) throws -> MetalBuffer<T> {
        let buffer = try makeBuffer(count: count, of: T.self)
        buffer.fill(with: value)
        return buffer
    }

    /// Creates a reusable buffer containing a copy of the array
    public func makeBuffer<T>(_ array: [T]) throws -> MetalBuffer<T> {
        let buffer = try makeBuffer(count: array.count, of: T.self)
        buffer.copy(from: array)
        return buffer
    }
}
