//
//  MetalArgument.swift
//
//  Part of MetalCompute
//
//  Types which can be passed as arguments to a Metal compute kernel
//

import Metal

/// A value which can be passed as argument to a Metal compute kernel.
///
/// Conforming types:
///
/// - Scalars: `Float`, `Int32`, `UInt32`, `Int16`, `UInt16`, `Int8`, `UInt8`, `Bool`
///   (and `Float16` on Apple silicon)
/// - SIMD vectors: `SIMD2`, `SIMD3`, `SIMD4`, `SIMD8`, `SIMD16` (e.g. `SIMD2<Float>` = Metal `float2`)
/// - `Double` and `Complex<Double>`: converted to the 64 bit emulation format of Metal64,
///   `float2` (hi, lo) and `float4` (re.hi, re.lo, im.hi, im.lo). `Complex<Float>` = Metal `float2`.
/// - Types conforming to `MetalEncodable`: converted with `toMetal()`
/// - Arrays and buffer pointers of these types: `[T]`, `UnsafeBufferPointer<T>`, `UnsafeMutableBufferPointer<T>`
/// - Persistent buffers: `MetalBuffer<T>`
///
/// Own structs with Metal layout conform by an empty extension, as long as they only contain
/// plain values (no classes, strings, arrays):
///
/// ```swift
/// struct Parameters { var maxIterations: UInt32; var bailout: Float }
/// extension Parameters: MetalArgument {}
/// ```
///
/// Structs with a different Swift layout (e.g. `Int` or `Double` fields) conform to
/// `MetalEncodable` instead and convert themselves with `toMetal()`.
///
/// Single values are passed with `setBytes` (Metal `constant T&` or `constant T*`),
/// arrays and buffer pointers are copied into a Metal buffer (Metal `device const T*`).
///
/// `Int` deliberately doesn't conform: Swift `Int` has 64 bit while Metal `int` has 32 bit.
/// Use `Int32` / `UInt32`, or `Int64` for Metal `long`.
public protocol MetalArgument {
    /// Binds the argument to the compute encoder at the given buffer index.
    /// The default implementation passes the bytes of the value.
    func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws

    /// Size in bytes of one element as it arrives in Metal: the value itself, or one element of
    /// an array / buffer. Used to check the argument against the kernel signature.
    /// The default is the stride of the type.
    var metalStride: Int { get }
}

/// Values up to this size are passed with setBytes, larger values in a buffer (Metal limit)
let maxInlineBytes = 4096

/// Precondition: the type must be a plain value type without references, which is copied 1:1.
/// Rejected are:
/// - Int and UInt, they don't match Metal int / uint
/// - Types conforming to MetalEncodable / MetalDecodable (e.g. Double, Complex<Double>):
///   they must be converted, so they can't be copied 1:1 into a MetalBuffer or used as raw layout
@inline(__always)
func requirePlainType<T>(_ type: T.Type) {
    precondition(_isPOD(type), "\(type) contains references (class, String, Array, ...) and cannot be passed to Metal")
    precondition(type != Int.self && type != UInt.self,
                 "Swift \(type) has 64 bit, use Int32 / UInt32 (Metal int / uint) or Int64 / UInt64 (Metal long / ulong)")
    precondition(!(type is any MetalEncodable.Type) && !(type is any MetalDecodable.Type),
                 "\(type) is converted with toMetal() / init(fromMetal:) and can't be copied 1:1, use its MetalLayout type")
}

/// Creates a shared Metal buffer containing a copy of the given bytes
func makeBuffer(_ bytes: UnsafeRawBufferPointer, device: any MTLDevice) throws -> any MTLBuffer {
    guard let base = bytes.baseAddress, bytes.count > 0,
          let buffer = device.makeBuffer(bytes: base, length: bytes.count, options: .storageModeShared) else {
        throw MetalComputeError.bufferCreationFailed(length: bytes.count)
    }
    return buffer
}

/// Passes a plain value with setBytes (or in a buffer if larger than 4 KB).
/// The value is padded to its stride, so the size matches the Metal struct layout.
func bindValue<T>(_ value: T, to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
    requirePlainType(T.self)

    var bytes = [UInt8](repeating: 0, count: MemoryLayout<T>.stride)
    withUnsafeBytes(of: value) { source in
        bytes.withUnsafeMutableBytes { $0.copyMemory(from: source) }
    }

    if bytes.count <= maxInlineBytes {
        bytes.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }
    else {
        let buffer = try bytes.withUnsafeBytes { try makeBuffer($0, device: device) }
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
}

extension MetalArgument {

    /// Passes the bytes of the value
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try bindValue(self, to: encoder, index: index, device: device)
    }

    /// Stride of the type
    public var metalStride: Int {
        MemoryLayout<Self>.stride
    }
}

// MARK: - Scalars and vectors

extension Float: MetalArgument {}
extension Int32: MetalArgument {}
extension UInt32: MetalArgument {}
extension Int16: MetalArgument {}
extension UInt16: MetalArgument {}
extension Int8: MetalArgument {}
extension UInt8: MetalArgument {}
extension Bool: MetalArgument {}

#if arch(arm64)
extension Float16: MetalArgument {}
#endif

extension SIMD2: MetalArgument {}
extension SIMD3: MetalArgument {}
extension SIMD4: MetalArgument {}
extension SIMD8: MetalArgument {}
extension SIMD16: MetalArgument {}

// MARK: - Arrays and buffer pointers

/// Size of one element in Metal: the MetalLayout stride for MetalEncodable types, otherwise the stride
func elementMetalStride<E>(_ type: E.Type) -> Int {
    if let encodable = type as? any MetalEncodable.Type {
        return encodable.encodedStride
    }
    return MemoryLayout<E>.stride
}

/// Copies the elements into a Metal buffer and binds it.
/// MetalEncodable elements (e.g. Double, Complex<Double>) are converted with toMetal().
func bindElements<E>(_ elements: UnsafeBufferPointer<E>, to encoder: any MTLComputeCommandEncoder,
                     index: Int, device: any MTLDevice) throws {
    guard let base = elements.baseAddress, !elements.isEmpty else {
        throw MetalComputeError.emptyArgument(index: index)
    }

    let buffer: any MTLBuffer
    if let encodable = E.self as? any MetalEncodable.Type {
        buffer = try encodable.makeMetalBuffer(from: UnsafeRawPointer(base), count: elements.count, device: device)
    }
    else {
        requirePlainType(E.self)
        buffer = try makeBuffer(UnsafeRawBufferPointer(elements), device: device)
    }
    encoder.setBuffer(buffer, offset: 0, index: index)
}

extension Array: MetalArgument {

    /// Copies the array into a Metal buffer
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try withUnsafeBufferPointer { try bindElements($0, to: encoder, index: index, device: device) }
    }

    /// Metal size of one element
    public var metalStride: Int {
        elementMetalStride(Element.self)
    }
}

extension UnsafeBufferPointer: MetalArgument {

    /// Copies the memory into a Metal buffer
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try bindElements(self, to: encoder, index: index, device: device)
    }

    /// Metal size of one element
    public var metalStride: Int {
        elementMetalStride(Element.self)
    }
}

extension UnsafeMutableBufferPointer: MetalArgument {

    /// Copies the memory into a Metal buffer
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try UnsafeBufferPointer(self).bind(to: encoder, index: index, device: device)
    }

    /// Metal size of one element
    public var metalStride: Int {
        elementMetalStride(Element.self)
    }
}
