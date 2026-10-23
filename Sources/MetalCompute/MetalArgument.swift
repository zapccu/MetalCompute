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
/// - Arrays and buffer pointers of plain values: `[T]`, `UnsafeBufferPointer<T>`, `UnsafeMutableBufferPointer<T>`
/// - Persistent buffers: `MetalBuffer<T>`
///
/// Own structs conform by an empty extension, as long as they only contain plain values
/// (no classes, strings, arrays):
///
/// ```swift
/// struct Parameters { var maxIterations: UInt32; var bailout: Float }
/// extension Parameters: MetalArgument {}
/// ```
///
/// Single values are passed with `setBytes` (Metal `constant T&` or `constant T*`),
/// arrays and buffer pointers are copied into a Metal buffer (Metal `device const T*`).
///
/// `Int` and `Double` deliberately don't conform: Swift `Int` has 64 bit while Metal `int`
/// has 32 bit, and Metal doesn't support `double`. Use `Int32` / `UInt32` and `Float`.
public protocol MetalArgument {
    /// Binds the argument to the compute encoder at the given buffer index.
    /// The default implementation passes the bytes of the value.
    func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws
}

/// Values up to this size are passed with setBytes, larger values in a buffer (Metal limit)
let maxInlineBytes = 4096

/// Precondition: the type must be a plain value type without references.
/// Int, UInt and Double are rejected, they don't match Metal int / uint / float.
@inline(__always)
func requirePlainType<T>(_ type: T.Type) {
    precondition(_isPOD(type), "\(type) contains references (class, String, Array, ...) and cannot be passed to Metal")
    precondition(type != Int.self && type != UInt.self,
                 "Swift \(type) has 64 bit, use Int32 / UInt32 (Metal int / uint) or Int64 / UInt64 (Metal long / ulong)")
    precondition(type != Double.self, "Metal doesn't support Double, use Float")
}

/// Creates a shared Metal buffer containing a copy of the given bytes
func makeBuffer(_ bytes: UnsafeRawBufferPointer, device: any MTLDevice) throws -> any MTLBuffer {
    guard let base = bytes.baseAddress, bytes.count > 0,
          let buffer = device.makeBuffer(bytes: base, length: bytes.count, options: .storageModeShared) else {
        throw MetalComputeError.bufferCreationFailed(length: bytes.count)
    }
    return buffer
}

extension MetalArgument {

    /// Passes the value with setBytes (or in a buffer if larger than 4 KB).
    /// The value is padded to its stride, so the size matches the Metal struct layout.
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        requirePlainType(Self.self)

        var bytes = [UInt8](repeating: 0, count: MemoryLayout<Self>.stride)
        withUnsafeBytes(of: self) { source in
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

extension Array: MetalArgument {

    /// Copies the array into a Metal buffer
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        requirePlainType(Element.self)
        guard !isEmpty else { throw MetalComputeError.emptyArgument(index: index) }
        let buffer = try withUnsafeBytes { try makeBuffer($0, device: device) }
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
}

extension UnsafeBufferPointer: MetalArgument {

    /// Copies the memory into a Metal buffer
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        requirePlainType(Element.self)
        guard !isEmpty else { throw MetalComputeError.emptyArgument(index: index) }
        let buffer = try makeBuffer(UnsafeRawBufferPointer(self), device: device)
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
}

extension UnsafeMutableBufferPointer: MetalArgument {

    /// Copies the memory into a Metal buffer
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try UnsafeBufferPointer(self).bind(to: encoder, index: index, device: device)
    }
}
