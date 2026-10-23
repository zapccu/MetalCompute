//
//  MetalArgument+F64.swift
//
//  Part of MetalCompute
//
//  Double and Complex<Double> arguments and results.
//
//  Metal doesn't support double. These values are converted to the 64 bit emulation format
//  of Metal64 (https://github.com/zapccu/Metal64):
//
//    Double          <-> float2 (hi, lo)                        Metal64: Float2 / f64
//    Complex<Double> <-> float4 (re.hi, re.lo, im.hi, im.lo)    Metal64: Complex2 / c64
//    Complex<Float>  <-> float2 (re, im), unchanged
//
//  hi is the Double rounded to Float, lo the rounded rest, hi + lo has ~48 bit precision.
//

import Metal
import RealModule
import ComplexModule

/// Splits a Double into the f64 format (hi, lo), same as Metal64's Float2(Double)
@inline(__always)
func f64Split(_ value: Double) -> SIMD2<Float> {
    let hi = Float(value)
    // For infinity, NaN and values beyond the Float range the rest would be NaN
    let lo = hi.isFinite ? Float(value - Double(hi)) : 0
    return SIMD2(hi, lo)
}

/// Joins the f64 format (hi, lo) to a Double
@inline(__always)
func f64Join(_ value: SIMD2<Float>) -> Double {
    // inf + (-inf) would be NaN, the low part is only valid for finite high parts
    value.x.isFinite ? Double(value.x) + Double(value.y) : Double(value.x)
}

/// Splits a Complex<Double> into the c64 format (re.hi, re.lo, im.hi, im.lo)
@inline(__always)
func c64Split(_ value: Complex<Double>) -> SIMD4<Float> {
    let re = f64Split(value.real)
    let im = f64Split(value.imaginary)
    return SIMD4(re.x, re.y, im.x, im.y)
}

// MARK: - Double

extension Double: MetalConvertible {

    public typealias MetalLayout = SIMD2<Float>

    /// Converts to f64 (Metal float2 / Metal64 Float2)
    public func toMetal() -> SIMD2<Float> {
        f64Split(self)
    }

    /// Creates a Double from f64 (Metal float2 / Metal64 Float2)
    public init(fromMetal value: SIMD2<Float>) {
        self = f64Join(value)
    }
}

// MARK: - Complex

/// Real type of a complex number which can be passed to Metal. Defines the Metal layout
/// of `Complex<Self>`. Implemented by `Double` (c64, float4) and `Float` (float2).
public protocol MetalComplexScalar: Real {
    associatedtype ComplexMetalLayout
    static func complexToMetal(_ value: Complex<Self>) -> ComplexMetalLayout
    static func complexFromMetal(_ value: ComplexMetalLayout) -> Complex<Self>
}

extension Double: MetalComplexScalar {
    public static func complexToMetal(_ value: Complex<Double>) -> SIMD4<Float> {
        c64Split(value)
    }
    public static func complexFromMetal(_ value: SIMD4<Float>) -> Complex<Double> {
        Complex(f64Join(SIMD2(value.x, value.y)), f64Join(SIMD2(value.z, value.w)))
    }
}

extension Float: MetalComplexScalar {
    public static func complexToMetal(_ value: Complex<Float>) -> SIMD2<Float> {
        SIMD2(value.real, value.imaginary)
    }
    public static func complexFromMetal(_ value: SIMD2<Float>) -> Complex<Float> {
        Complex(value.x, value.y)
    }
}

// A conditional conformance doesn't imply the inherited protocols, so all are declared.
// The members are implemented explicitly instead of relying on the protocol defaults.

extension Complex: MetalArgument where RealType: MetalComplexScalar {

    /// Metal layout, used by MetalEncodable and MetalDecodable
    public typealias MetalLayout = RealType.ComplexMetalLayout

    /// Passes the value in Metal layout
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try bindValue(toMetal(), to: encoder, index: index, device: device)
    }

    public var metalStride: Int {
        MemoryLayout<MetalLayout>.stride
    }
}

extension Complex: MetalEncodable where RealType: MetalComplexScalar {

    /// Converts to c64 (Complex<Double>) or float2 (Complex<Float>)
    public func toMetal() -> RealType.ComplexMetalLayout {
        RealType.complexToMetal(self)
    }
}

extension Complex: MetalDecodable where RealType: MetalComplexScalar {

    /// Creates a complex number from c64 (Complex<Double>) or float2 (Complex<Float>)
    public init(fromMetal value: RealType.ComplexMetalLayout) {
        self = RealType.complexFromMetal(value)
    }
}
