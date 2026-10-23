//
//  MetalArgument+F64.swift
//
//  Part of MetalCompute
//
//  Double and Complex<Double> arguments.
//
//  Metal doesn't support double. These values are converted to the 64 bit emulation format
//  of Metal64 (https://github.com/zapccu/Metal64):
//
//    Double          -> float2 (hi, lo)                  Metal64: Float2 / f64
//    Complex<Double> -> float4 (re.hi, re.lo, im.hi, im.lo)   Metal64: Complex2 / c64
//
//  hi is the Double rounded to Float, lo the rounded rest, hi + lo has ~48 bit precision.
//

import Metal
import ComplexModule

/// Splits a Double into the f64 format (hi, lo), same as Metal64's Float2(Double)
@inline(__always)
func f64Split(_ value: Double) -> SIMD2<Float> {
    let hi = Float(value)
    // For infinity, NaN and values beyond the Float range the rest would be NaN
    let lo = hi.isFinite ? Float(value - Double(hi)) : 0
    return SIMD2(hi, lo)
}

/// Splits a Complex<Double> into the c64 format (re.hi, re.lo, im.hi, im.lo)
@inline(__always)
func c64Split(_ value: Complex<Double>) -> SIMD4<Float> {
    let re = f64Split(value.real)
    let im = f64Split(value.imaginary)
    return SIMD4(re.x, re.y, im.x, im.y)
}

/// True for the types which are converted instead of copied 1:1
func isF64Type<T>(_ type: T.Type) -> Bool {
    type == Double.self || type == Complex<Double>.self
}

/// Converts Double / Complex<Double> elements to f64 / c64.
/// A c64 value (float4) has the same memory layout as two f64 values (float2), so both
/// are returned as array of SIMD2<Float>: one per Double, two (re, im) per Complex<Double>.
/// Returns nil for all other element types, they are copied 1:1.
func convertF64Elements<E>(_ elements: UnsafeBufferPointer<E>) -> [SIMD2<Float>]? {
    if E.self == Double.self {
        return elements.withMemoryRebound(to: Double.self) { values in
            values.map(f64Split)
        }
    }
    if E.self == Complex<Double>.self {
        return elements.withMemoryRebound(to: Complex<Double>.self) { values in
            var result = [SIMD2<Float>]()
            result.reserveCapacity(values.count * 2)
            for value in values {
                result.append(f64Split(value.real))
                result.append(f64Split(value.imaginary))
            }
            return result
        }
    }
    return nil
}

// MARK: - Conformances

extension Double: MetalArgument {

    /// Passes the value as f64 (Metal float2 / Metal64 f64)
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try f64Split(self).bind(to: encoder, index: index, device: device)
    }
}

extension Complex: MetalArgument {

    /// Complex<Double> is passed as c64 (Metal float4 / Metal64 c64),
    /// other complex types 1:1 (Complex<Float> = Metal float2)
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        if let value = self as? Complex<Double> {
            try c64Split(value).bind(to: encoder, index: index, device: device)
        }
        else {
            requirePlainType(Self.self)
            var bytes = [UInt8](repeating: 0, count: MemoryLayout<Self>.stride)
            withUnsafeBytes(of: self) { source in
                bytes.withUnsafeMutableBytes { $0.copyMemory(from: source) }
            }
            bytes.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
        }
    }
}
