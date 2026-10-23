//
//  MetalComputeError.swift
//
//  Part of MetalCompute
//

/// Errors thrown by MetalCompute
public enum MetalComputeError: Error, CustomStringConvertible {
    /// No Metal device available
    case noDevice
    /// The Metal library could not be loaded or compiled
    case libraryNotAvailable(underlying: any Error)
    /// The library doesn't contain a kernel function with this name
    case functionNotFound(String)
    /// The compute pipeline for the kernel function could not be created
    case pipelineCreationFailed(String, underlying: any Error)
    /// Command queue, command buffer or compute encoder could not be created
    case commandCreationFailed(String)
    /// A Metal buffer with the given length could not be allocated
    case bufferCreationFailed(length: Int)
    /// The number of elements is invalid (must be > 0)
    case invalidCount(Int)
    /// An input array or buffer pointer is empty
    case emptyArgument(index: Int)
    /// Too many arguments, Metal supports buffer indices 0 ... 30
    case tooManyArguments(Int)
    /// The output buffer is smaller than the number of elements to compute
    case outputBufferTooSmall(required: Int, available: Int)
    /// The GPU reported an error while executing the kernel
    case executionFailed(underlying: any Error)
    /// The number of arguments doesn't match the buffer arguments of the kernel function.
    /// `kernelBuffers` are the buffer indices of the kernel (inputs and output).
    case argumentCountMismatch(kernel: String, kernelBuffers: [Int], given: Int)
    /// The size of an argument (or one element of it) doesn't match the kernel function
    case layoutMismatch(kernel: String, index: Int, isOutput: Bool, swiftSize: Int, metalSize: Int)

    public var description: String {
        switch self {
        case .noDevice:
            return "No Metal device available"
        case .libraryNotAvailable(let error):
            return "Metal library not available: \(error)"
        case .functionNotFound(let name):
            return "Kernel function '\(name)' not found in Metal library"
        case .pipelineCreationFailed(let name, let error):
            return "Cannot create compute pipeline for '\(name)': \(error)"
        case .commandCreationFailed(let what):
            return "Cannot create \(what)"
        case .bufferCreationFailed(let length):
            return "Cannot create Metal buffer with \(length) bytes"
        case .invalidCount(let count):
            return "Invalid number of elements: \(count)"
        case .emptyArgument(let index):
            return "Argument at buffer index \(index) is empty"
        case .tooManyArguments(let count):
            return "Too many arguments (\(count)), Metal supports up to 31 buffers"
        case .outputBufferTooSmall(let required, let available):
            return "Output buffer too small: \(required) elements required, \(available) available"
        case .executionFailed(let error):
            return "Kernel execution failed: \(error)"
        case .argumentCountMismatch(let kernel, let kernelBuffers, let given):
            return "Kernel '\(kernel)' has buffer arguments \(kernelBuffers) (inputs + output), "
                + "but \(given) arguments + output were passed (buffer 0...\(given))"
        case .layoutMismatch(let kernel, let index, let isOutput, let swiftSize, let metalSize):
            let what = isOutput ? "output buffer" : "argument"
            return "Kernel '\(kernel)', \(what) at buffer index \(index): Swift type has \(swiftSize) bytes, "
                + "Metal expects \(metalSize) bytes. Check the struct layout (Int vs Int32, Double vs Float2, padding) "
                + "and the order of the arguments"
        }
    }
}
