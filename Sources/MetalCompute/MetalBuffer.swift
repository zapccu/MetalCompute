//
//  MetalBuffer.swift
//
//  Part of MetalCompute
//
//  Typed wrapper for a Metal buffer which can be reused between kernel runs
//

import Metal

/// A typed Metal buffer with `count` elements of type `Element`.
///
/// Use it to avoid allocations when a kernel is executed repeatedly with the same sizes,
/// e.g. for redrawing a fractal while zooming. The buffer is in shared memory,
/// so the CPU can read and write it directly.
///
/// Create it with `MetalCompute.makeBuffer(...)`. It can be passed as kernel argument
/// and as output buffer (`ComputeKernel.run(..., into:)`).
public struct MetalBuffer<Element>: MetalArgument, @unchecked Sendable {

    /// The underlying Metal buffer
    public let buffer: any MTLBuffer

    /// Number of elements
    public let count: Int

    /// Wraps an existing Metal buffer. The buffer must hold at least `count` elements
    /// and use shared or managed storage, so the CPU can access it.
    public init(buffer: any MTLBuffer, count: Int) {
        precondition(buffer.length >= MemoryLayout<Element>.stride * count, "Metal buffer is too small")
        self.buffer = buffer
        self.count = count
    }

    /// Allocates a new shared buffer with `count` elements, initialized with zero bytes
    init(device: any MTLDevice, count: Int) throws {
        requirePlainType(Element.self)
        guard count > 0 else { throw MetalComputeError.invalidCount(count) }
        let length = MemoryLayout<Element>.stride * count
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw MetalComputeError.bufferCreationFailed(length: length)
        }
        self.buffer = buffer
        self.count = count
    }

    /// Typed pointer to the buffer memory
    public var pointer: UnsafeMutableBufferPointer<Element> {
        let base = buffer.contents().bindMemory(to: Element.self, capacity: count)
        return UnsafeMutableBufferPointer(start: base, count: count)
    }

    /// Copy of the buffer content as array
    public func toArray() -> [Element] {
        Array(pointer)
    }

    /// Copy of the buffer content as array, converted element by element
    public func map<T>(_ transform: (Element) throws -> T) rethrows -> [T] {
        try pointer.map(transform)
    }

    /// Element access. Don't modify the buffer while a kernel using it is running.
    public subscript(index: Int) -> Element {
        get { pointer[index] }
        nonmutating set { pointer[index] = newValue }
    }

    /// Copies the elements of an array into the buffer (array.count must be <= count)
    public func copy(from array: [Element]) {
        precondition(array.count <= count, "Array has more elements than the buffer")
        _ = pointer.update(fromContentsOf: array)
    }

    /// Sets all elements to the given value
    public func fill(with value: Element) {
        pointer.update(repeating: value)
    }

    /// Converts the buffer content into an array of a MetalDecodable type whose
    /// MetalLayout is the element type of this buffer
    public func decoded<T: MetalDecodable>(as type: T.Type = T.self) -> [T] where T.MetalLayout == Element {
        pointer.map { T(fromMetal: $0) }
    }

    /// Converts the values of a MetalEncodable array into the buffer (array.count must be <= count)
    public func encode<T: MetalEncodable>(_ array: [T]) where T.MetalLayout == Element {
        precondition(array.count <= count, "Array has more elements than the buffer")
        let target = pointer
        for (index, value) in array.enumerated() {
            target[index] = value.toMetal()
        }
    }

    /// Binds the buffer to the compute encoder (no copy)
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        encoder.setBuffer(buffer, offset: 0, index: index)
    }

    /// Size of one element
    public var metalStride: Int {
        MemoryLayout<Element>.stride
    }
}
