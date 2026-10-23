//
//  MetalComputeTests.swift
//
//  Part of MetalCompute
//
//  The kernels are compiled at runtime from source code, so the tests don't need
//  a Metal library in the test bundle. Without Metal device the tests are skipped.
//

import Testing
import Metal
import ComplexModule
@testable import MetalCompute

// Must match the Metal struct layout below
struct Parameters: MetalArgument {
    var scale: Float
    var offset: Float
    var iterations: UInt32
}

// 20 * 64 floats = 5120 bytes > 4 KB, passed in a buffer instead of setBytes
struct LargeValue: MetalArgument {
    var values: (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>,
                 SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>,
                 SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>,
                 SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>)
}

// Result struct with padding: size 5, stride 8 in Swift, size 8 in Metal
struct EscapeResult: Equatable {
    var iterations: UInt32
    var escaped: Bool
}

let kernelSource = """
#include <metal_stdlib>
using namespace metal;

struct Parameters {
    float scale;
    float offset;
    uint  iterations;
};

struct EscapeResult {
    uint iterations;
    bool escaped;
};

kernel void add_arrays(device const float *a      [[buffer(0)]],
                       device const float *b      [[buffer(1)]],
                       device float       *result [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}

kernel void scale_offset(device const float *values  [[buffer(0)]],
                         constant Parameters &params [[buffer(1)]],
                         constant int &extra         [[buffer(2)]],
                         device float *result        [[buffer(3)]],
                         uint index [[thread_position_in_grid]])
{
    result[index] = values[index] * params.scale + params.offset + float(extra);
}

kernel void keep_initial(device uint *result [[buffer(0)]],
                         uint index [[thread_position_in_grid]])
{
    if (index % 2 == 0) result[index] = index;
}

kernel void grid_index(constant uint &width  [[buffer(0)]],
                       device uint *result   [[buffer(1)]],
                       uint2 position [[thread_position_in_grid]])
{
    result[position.y * width + position.x] = position.x * 1000 + position.y;
}

kernel void sum_large(constant float *large [[buffer(0)]],
                      device float *result  [[buffer(1)]],
                      uint index [[thread_position_in_grid]])
{
    float sum = 0.0;
    for (int i = 0; i < 1280; i++) sum += large[i];
    result[index] = sum;
}

// Copy kernels for the f64 / c64 conversion tests
kernel void copy_float2_value(constant float2 &value  [[buffer(0)]],
                              device float2   *result [[buffer(1)]],
                              uint index [[thread_position_in_grid]])
{
    result[index] = value;
}

kernel void copy_float2_array(device const float2 *values [[buffer(0)]],
                              device float2       *result [[buffer(1)]],
                              uint index [[thread_position_in_grid]])
{
    result[index] = values[index];
}

kernel void copy_float4_value(constant float4 &value  [[buffer(0)]],
                              device float4   *result [[buffer(1)]],
                              uint index [[thread_position_in_grid]])
{
    result[index] = value;
}

kernel void copy_float4_array(device const float4 *values [[buffer(0)]],
                              device float4       *result [[buffer(1)]],
                              uint index [[thread_position_in_grid]])
{
    result[index] = values[index];
}

// Mandelbrot iteration with escape radius 2, points as float2 (re, im)
kernel void mandelbrot(device const float2 *points    [[buffer(0)]],
                       constant Parameters &params    [[buffer(1)]],
                       device EscapeResult *result    [[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    float2 c = points[index];
    float2 z = float2(0.0, 0.0);
    uint i = 0;
    for (; i < params.iterations; i++) {
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (z.x * z.x + z.y * z.y > 4.0) break;
    }
    result[index].iterations = i;
    result[index].escaped = i < params.iterations;
}
"""

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct MetalComputeTests {

    let metal: MetalCompute

    init() throws {
        metal = try MetalCompute(source: kernelSource)
    }

    @Test func addArrays() async throws {
        let a: [Float] = (0..<1000).map { Float($0) }
        let b: [Float] = (0..<1000).map { Float($0) * 0.5 }
        let result: [Float] = try await metal.kernel("add_arrays").run(a, b, count: a.count)
        #expect(result == zip(a, b).map { $0 + $1 })
    }

    @Test func structAndScalarArguments() async throws {
        let values: [Float] = [1, 2, 3, 4]
        let params = Parameters(scale: 2, offset: 0.5, iterations: 0)
        let result: [Float] = try await metal.kernel("scale_offset").run(values, params, Int32(10), count: values.count)
        #expect(result == [12.5, 14.5, 16.5, 18.5])
    }

    @Test func bufferPointerArgument() throws {
        let a: [Float] = [1, 2, 3]
        var b: [Float] = [10, 20, 30]
        let result: [Float] = try b.withUnsafeMutableBufferPointer { pointer in
            try metal.kernel("add_arrays").runAndWait(a, pointer, count: 3)
        }
        #expect(result == [11, 22, 33])
    }

    @Test func initialValue() async throws {
        let result = try await metal.kernel("keep_initial").run(count: 6, returning: UInt32.self, initialValue: 99)
        #expect(result == [0, 99, 2, 99, 4, 99])
    }

    @Test func twoDimensionalGrid() async throws {
        let width = 7, height = 5
        let result: [UInt32] = try await metal.kernel("grid_index").run(UInt32(width), width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                #expect(result[y * width + x] == UInt32(x * 1000 + y))
            }
        }
    }

    @Test func largeValueInBuffer() async throws {
        let v = SIMD64<Float>(repeating: 1)
        let large = LargeValue(values: (v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v))
        #expect(MemoryLayout<LargeValue>.stride > maxInlineBytes)
        let result: [Float] = try await metal.kernel("sum_large").run(large, count: 1)
        #expect(result == [1280])
    }

    @Test func reusableBuffers() async throws {
        let kernel = try metal.kernel("add_arrays")
        let a = try metal.makeBuffer([Float](repeating: 1, count: 100))
        let output = try metal.makeBuffer(count: 100, of: Float.self)

        for step in 1...3 {
            let b = [Float](repeating: Float(step), count: 100)
            try await kernel.run(a, b, count: 100, into: output)
            #expect(output.toArray() == [Float](repeating: Float(step) + 1, count: 100))
        }
    }

    @Test func mandelbrot() async throws {
        // c = 0 and c = -1 stay bounded, c = 1: z = 1, 2, 5 -> escapes in iteration 2 (index),
        // c = 2 + 2i escapes immediately
        let points: [SIMD2<Float>] = [[0, 0], [-1, 0], [1, 0], [2, 2]]
        let params = Parameters(scale: 0, offset: 0, iterations: 100)
        let result: [EscapeResult] = try await metal.kernel("mandelbrot").run(points, params, count: points.count)
        #expect(result == [
            EscapeResult(iterations: 100, escaped: false),
            EscapeResult(iterations: 100, escaped: false),
            EscapeResult(iterations: 2, escaped: true),
            EscapeResult(iterations: 0, escaped: true)
        ])
    }

    @Test func kernelsAreCached() throws {
        #expect(try metal.kernel("add_arrays") === metal.kernel("add_arrays"))
    }

    @Test func missingFunction() {
        #expect(throws: MetalComputeError.self) {
            try metal.kernel("does_not_exist")
        }
    }

    @Test func emptyArrayIsRejected() async {
        await #expect(throws: MetalComputeError.self) {
            let _: [Float] = try await metal.kernel("add_arrays").run([Float](), [Float](), count: 1)
        }
    }

    @Test func outputBufferTooSmall() async throws {
        let output = try metal.makeBuffer(count: 2, of: Float.self)
        await #expect(throws: MetalComputeError.self) {
            try await metal.kernel("add_arrays").run([Float](repeating: 1, count: 4), [Float](repeating: 1, count: 4),
                                                     count: 4, into: output)
        }
    }

    // MARK: - Double and Complex<Double>

    static let doubles: [Double] = [.pi, -1.0 / 3.0, 1e30, -2.5e-30, 0, 123456789.123456789, .infinity]

    /// hi + lo as Double
    static func joined(_ f64: SIMD2<Float>) -> Double { Double(f64.x) + Double(f64.y) }

    @Test func doubleValue() async throws {
        let value = Double.pi
        let result: [SIMD2<Float>] = try await metal.kernel("copy_float2_value").run(value, count: 1)
        #expect(result == [f64Split(value)])
        // ~48 bit precision, much better than Float
        #expect(abs(Self.joined(result[0]) - value) < 1e-14)
        #expect(result[0].y != 0)
    }

    @Test func doubleArray() async throws {
        let result: [SIMD2<Float>] = try await metal.kernel("copy_float2_array").run(Self.doubles, count: Self.doubles.count)
        #expect(result == Self.doubles.map(f64Split))
        #expect(result.last == SIMD2<Float>(.infinity, 0))
        for (value, f64) in zip(Self.doubles.dropLast(), result) {
            #expect(abs(Self.joined(f64) - value) <= abs(value) * 1e-14)
        }
    }

    @Test func doubleBufferPointer() throws {
        let result: [SIMD2<Float>] = try Self.doubles.withUnsafeBufferPointer { pointer in
            try metal.kernel("copy_float2_array").runAndWait(pointer, count: pointer.count)
        }
        #expect(result == Self.doubles.map(f64Split))
    }

    @Test func complexDoubleValue() async throws {
        let value = Complex<Double>(1.0 / 3.0, -2.0 / 7.0)
        let result: [SIMD4<Float>] = try await metal.kernel("copy_float4_value").run(value, count: 1)
        let re = f64Split(value.real), im = f64Split(value.imaginary)
        #expect(result == [SIMD4<Float>(re.x, re.y, im.x, im.y)])
    }

    @Test func complexDoubleArray() async throws {
        let values = [Complex<Double>(.pi, -.pi), Complex<Double>(1e-20, 1e20), Complex<Double>(0, 1)]
        let result: [SIMD4<Float>] = try await metal.kernel("copy_float4_array").run(values, count: values.count)
        #expect(result == values.map(c64Split))
        for (value, c64) in zip(values, result) {
            #expect(abs(Self.joined(SIMD2(c64.x, c64.y)) - value.real) <= abs(value.real) * 1e-14)
            #expect(abs(Self.joined(SIMD2(c64.z, c64.w)) - value.imaginary) <= abs(value.imaginary) * 1e-14)
        }
    }

    @Test func complexFloatIsPassedUnchanged() async throws {
        let value = Complex<Float>(1.5, -2.5)
        let result: [SIMD2<Float>] = try await metal.kernel("copy_float2_value").run(value, count: 2)
        #expect(result == [SIMD2<Float>(1.5, -2.5), SIMD2<Float>(1.5, -2.5)])
    }

    @Test func f64SplitSpecialValues() {
        #expect(f64Split(.infinity) == SIMD2<Float>(.infinity, 0))
        #expect(f64Split(-.infinity) == SIMD2<Float>(-.infinity, 0))
        #expect(f64Split(1e300) == SIMD2<Float>(.infinity, 0))    // beyond Float range
        #expect(f64Split(.nan).x.isNaN)
        #expect(f64Split(1.0) == SIMD2<Float>(1, 0))
    }
}
