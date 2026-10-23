//
//  ComputeKernel.swift
//
//  Part of MetalCompute
//
//  A Metal compute kernel function, ready to run
//

import Metal

/// A kernel function of a Metal library with its compute pipeline.
///
/// Get it with `MetalCompute.kernel(_:)`. Each run is independent, so a kernel can be
/// executed concurrently from several tasks.
///
/// ## Argument order
///
/// The arguments are bound to the buffer indices in the order they are passed:
/// the first argument to `[[buffer(0)]]`, the second to `[[buffer(1)]]` and so on.
/// The output buffer follows the last argument.
///
/// ```metal
/// kernel void mandelbrot(device const float4      *points [[buffer(0)]],
///                        constant Parameters      &params [[buffer(1)]],
///                        constant float           &potential [[buffer(2)]],
///                        device MandelbrotResult  *result [[buffer(3)]],
///                        uint index [[thread_position_in_grid]])
/// ```
///
/// ```swift
/// let result: [MandelbrotResult] = try await kernel.run(points, params, potential, count: points.count)
/// ```
///
/// With `count` elements the kernel runs `count` threads, `thread_position_in_grid` is 0 ... count-1.
/// For two dimensional grids use `run(..., width:, height:)`, the kernel then gets a `uint2` position.
public final class ComputeKernel: @unchecked Sendable {

    /// Name of the kernel function
    public let name: String

    /// The compute pipeline
    public let pipeline: any MTLComputePipelineState

    /// The Metal device
    public let device: any MTLDevice

    /// Command queue used for the kernel runs
    public let commandQueue: any MTLCommandQueue

    /// Sizes of the kernel's buffer arguments (buffer index -> bytes per element) from the
    /// pipeline reflection. nil if argument validation is disabled.
    public let bufferSizes: [Int: Int]?

    // The kernel doesn't reference the MetalCompute object: MetalCompute caches its kernels,
    // a reference back would create a retain cycle.
    init(name: String, pipeline: any MTLComputePipelineState,
         device: any MTLDevice, commandQueue: any MTLCommandQueue, bufferSizes: [Int: Int]?) {
        self.name = name
        self.pipeline = pipeline
        self.device = device
        self.commandQueue = commandQueue
        self.bufferSizes = bufferSizes
    }

    // MARK: - Run and return an array

    /// Runs the kernel for `count` elements and returns the output buffer as array.
    /// - Parameters:
    ///   - arguments: Kernel arguments, bound to buffer index 0, 1, ...
    ///   - count: Number of elements (threads) to compute
    ///   - type: Type of the output elements, can be omitted if it's inferred from the result type
    ///   - initialValue: Initial value of all output elements, default: zero bytes
    /// - Returns: Array with `count` output elements
    ///
    /// If `Output` conforms to `MetalDecodable`, the output buffer has `Output.MetalLayout`
    /// elements and each element is converted with `init(fromMetal:)` before it is returned.
    /// An `initialValue` is then converted with `toMetal()` (requires `MetalEncodable`).
    public func run<Output>(_ arguments: any MetalArgument..., count: Int,
                            returning type: Output.Type = Output.self,
                            initialValue: Output? = nil) async throws -> [Output] {
        let output = try OutputBuffer<Output>(device: device, count: count, initialValue: initialValue)
        try await execute(arguments, grid: MTLSize(width: count, height: 1, depth: 1),
                          output: output.buffer, outputStride: output.metalStride)
        return output.result()
    }

    /// Runs the kernel on a two dimensional grid of `width` x `height` threads.
    /// The kernel gets the position as `uint2 [[thread_position_in_grid]]` and should write
    /// its result to index `position.y * width + position.x`.
    /// - Returns: Array with `width * height` output elements (row by row)
    public func run<Output>(_ arguments: any MetalArgument..., width: Int, height: Int,
                            returning type: Output.Type = Output.self,
                            initialValue: Output? = nil) async throws -> [Output] {
        let output = try OutputBuffer<Output>(device: device, count: width * height, initialValue: initialValue)
        try await execute(arguments, grid: MTLSize(width: width, height: height, depth: 1),
                          output: output.buffer, outputStride: output.metalStride)
        return output.result()
    }

    // MARK: - Run into a reusable buffer

    /// Runs the kernel for `count` elements and writes the output into an existing buffer.
    /// Avoids allocating a new output buffer for every run. The buffer holds the Metal layout,
    /// convert MetalDecodable results with `output.decoded(as:)`.
    public func run<Output>(_ arguments: any MetalArgument..., count: Int,
                            into output: MetalBuffer<Output>) async throws {
        try checkOutput(output, count: count)
        try await execute(arguments, grid: MTLSize(width: count, height: 1, depth: 1),
                          output: output.buffer, outputStride: output.metalStride)
    }

    /// Runs the kernel on a two dimensional grid and writes the output into an existing buffer.
    public func run<Output>(_ arguments: any MetalArgument..., width: Int, height: Int,
                            into output: MetalBuffer<Output>) async throws {
        try checkOutput(output, count: width * height)
        try await execute(arguments, grid: MTLSize(width: width, height: height, depth: 1),
                          output: output.buffer, outputStride: output.metalStride)
    }

    // MARK: - Blocking variants

    /// Like `run(_:count:)`, but blocks the calling thread until the GPU has finished.
    /// Don't call this on the main thread for long running kernels.
    public func runAndWait<Output>(_ arguments: any MetalArgument..., count: Int,
                                   returning type: Output.Type = Output.self,
                                   initialValue: Output? = nil) throws -> [Output] {
        let output = try OutputBuffer<Output>(device: device, count: count, initialValue: initialValue)
        try executeAndWait(arguments, grid: MTLSize(width: count, height: 1, depth: 1),
                           output: output.buffer, outputStride: output.metalStride)
        return output.result()
    }

    /// Like `run(_:width:height:)`, but blocks the calling thread until the GPU has finished.
    public func runAndWait<Output>(_ arguments: any MetalArgument..., width: Int, height: Int,
                                   returning type: Output.Type = Output.self,
                                   initialValue: Output? = nil) throws -> [Output] {
        let output = try OutputBuffer<Output>(device: device, count: width * height, initialValue: initialValue)
        try executeAndWait(arguments, grid: MTLSize(width: width, height: height, depth: 1),
                           output: output.buffer, outputStride: output.metalStride)
        return output.result()
    }

    /// Like `run(_:count:into:)`, but blocks the calling thread until the GPU has finished.
    public func runAndWait<Output>(_ arguments: any MetalArgument..., count: Int,
                                   into output: MetalBuffer<Output>) throws {
        try checkOutput(output, count: count)
        try executeAndWait(arguments, grid: MTLSize(width: count, height: 1, depth: 1),
                           output: output.buffer, outputStride: output.metalStride)
    }

    // MARK: - Implementation

    private func checkOutput<Output>(_ output: MetalBuffer<Output>, count: Int) throws {
        guard count > 0 else { throw MetalComputeError.invalidCount(count) }
        guard output.count >= count else {
            throw MetalComputeError.outputBufferTooSmall(required: count, available: output.count)
        }
    }

    /// Checks the arguments and the output against the buffer arguments of the kernel function:
    /// the number of buffers and the size of each value / element must match.
    func validate(_ arguments: [any MetalArgument], outputStride: Int) throws {
        guard let bufferSizes else { return }

        let expected = bufferSizes.keys.sorted()
        let given = Array(0...arguments.count)       // arguments + output
        guard expected == given else {
            throw MetalComputeError.argumentCountMismatch(kernel: name, kernelBuffers: expected, given: arguments.count)
        }

        for (index, argument) in arguments.enumerated() {
            if let metalSize = bufferSizes[index], metalSize != argument.metalStride {
                throw MetalComputeError.layoutMismatch(kernel: name, index: index, isOutput: false,
                                                       swiftSize: argument.metalStride, metalSize: metalSize)
            }
        }
        if let metalSize = bufferSizes[arguments.count], metalSize != outputStride {
            throw MetalComputeError.layoutMismatch(kernel: name, index: arguments.count, isOutput: true,
                                                   swiftSize: outputStride, metalSize: metalSize)
        }
    }

    /// Encodes arguments, output buffer and dispatch into a new command buffer (not committed)
    private func encode(_ arguments: [any MetalArgument], grid: MTLSize,
                        output: any MTLBuffer, outputStride: Int) throws -> any MTLCommandBuffer {
        guard grid.width > 0, grid.height > 0 else {
            throw MetalComputeError.invalidCount(grid.width * grid.height)
        }
        // Buffer indices 0 ... 30, the output needs one more
        guard arguments.count < 31 else {
            throw MetalComputeError.tooManyArguments(arguments.count)
        }
        try validate(arguments, outputStride: outputStride)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalComputeError.commandCreationFailed("command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.commandCreationFailed("compute command encoder")
        }

        do {
            encoder.setComputePipelineState(pipeline)
            for (index, argument) in arguments.enumerated() {
                try argument.bind(to: encoder, index: index, device: device)
            }
            encoder.setBuffer(output, offset: 0, index: arguments.count)
        }
        catch {
            // An encoder must always be ended, even if the command buffer is discarded
            encoder.endEncoding()
            throw error
        }

        // Threadgroup size: width is a multiple of the SIMD width, as large as allowed.
        // The grid doesn't need to be a multiple of the threadgroup size (non-uniform
        // threadgroups are supported on all devices running macOS 13 / iOS 16).
        let simdWidth = pipeline.threadExecutionWidth
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let threadgroup: MTLSize
        if grid.height == 1 {
            threadgroup = MTLSize(width: min(maxThreads, grid.width), height: 1, depth: 1)
        }
        else {
            let width = min(simdWidth, grid.width)
            threadgroup = MTLSize(width: width, height: min(max(1, maxThreads / width), grid.height), depth: 1)
        }
        encoder.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()

        return commandBuffer
    }

    private func execute(_ arguments: [any MetalArgument], grid: MTLSize,
                         output: any MTLBuffer, outputStride: Int) async throws {
        let commandBuffer = try encode(arguments, grid: grid, output: output, outputStride: outputStride)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: MetalComputeError.executionFailed(underlying: error))
                }
                else {
                    continuation.resume()
                }
            }
            commandBuffer.commit()
        }
    }

    private func executeAndWait(_ arguments: [any MetalArgument], grid: MTLSize,
                                output: any MTLBuffer, outputStride: Int) throws {
        let commandBuffer = try encode(arguments, grid: grid, output: output, outputStride: outputStride)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MetalComputeError.executionFailed(underlying: error)
        }
    }
}
