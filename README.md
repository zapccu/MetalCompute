# MetalCompute

Interface for passing compute tasks to Metal.

MetalCompute runs a Metal compute kernel with a list of arguments (arrays, buffer pointers,
structs, scalars, SIMD vectors) and returns the result as a Swift array.

```swift
import MetalCompute

let metal = try MetalCompute()                      // kernels from the app's Metal library
let kernel = try metal.kernel("mandelbrot")         // created once, cached

let result: [MandelbrotResult] = try await kernel.run(points, parameters, potential, count: points.count)
```

Requirements: macOS 13 / iOS 16, Swift 6.

## Installation

Add the package in Xcode (*File → Add Package Dependencies…*) with the URL
`https://github.com/zapccu/MetalCompute`, or in `Package.swift`:

```swift
.package(url: "https://github.com/zapccu/MetalCompute", from: "1.0.0")
```

## Usage

### 1. Create a MetalCompute object

Creating the object loads the Metal device, the command queue and the library. Create it once
and keep it, e.g. as a static property. It is thread safe (`Sendable`).

```swift
// Kernels from the .metal files of the app (default library of the main bundle)
static let metal = try? MetalCompute()

// Kernels from the default library of another bundle
let metal = try MetalCompute(bundle: .module)

// Kernels compiled at runtime from source code (safe math mode by default)
let metal = try MetalCompute(source: metalSourceCode)

// Existing library and device
let metal = try MetalCompute(library: myLibrary, device: myDevice)
```

### 2. Get a kernel

```swift
let kernel = try metal.kernel("mandelbrot")
```

The compute pipeline is created on the first call and cached, later calls are cheap.

### 3. Run the kernel

```swift
let result: [Float] = try await kernel.run(a, b, count: a.count)
```

- The arguments are bound to the buffer indices in the order they are passed:
  first argument `[[buffer(0)]]`, second argument `[[buffer(1)]]`, …
- The output buffer follows the last argument.
- `count` threads are started, `thread_position_in_grid` runs from 0 to `count - 1`.
- The output type is inferred from the result type, or given with `returning: T.self`.

```metal
kernel void add(device const float *a      [[buffer(0)]],
                device const float *b      [[buffer(1)]],
                device float       *result [[buffer(2)]],
                uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}
```

#### Variants

| Call | Description |
|------|-------------|
| `run(args..., count: n)` | `n` threads, returns `[Output]` with `n` elements |
| `run(args..., count: n, initialValue: v)` | Output elements are initialized with `v` before the kernel runs |
| `run(args..., width: w, height: h)` | 2D grid, kernel gets `uint2 [[thread_position_in_grid]]`, returns `w * h` elements row by row |
| `run(args..., count: n, into: buffer)` | Writes into a reusable `MetalBuffer`, no allocation |
| `runAndWait(...)` | Same as `run`, but synchronous: blocks the calling thread until the GPU has finished |

## Arguments

All types conforming to `MetalArgument` can be passed:

| Swift | Metal | Passed as |
|-------|-------|-----------|
| `Float` | `float` | value (`constant float &`) |
| `Int32`, `UInt32` | `int`, `uint` | value |
| `Int16`, `UInt16`, `Int8`, `UInt8` | `short`, `ushort`, `char`, `uchar` | value |
| `Bool` | `bool` | value |
| `Float16` (Apple silicon only) | `half` | value |
| `SIMD2<Float>`, `SIMD3<Float>`, `SIMD4<Float>` | `float2`, `float3`, `float4` | value |
| `SIMD2<Int32>`, `SIMD4<UInt32>`, … | `int2`, `uint4`, … | value |
| `Double` | `float2` (Metal64 `f64`) | value, converted to (hi, lo) |
| `Complex<Double>` | `float4` (Metal64 `c64`) | value, converted to (re.hi, re.lo, im.hi, im.lo) |
| `Complex<Float>` | `float2` | value |
| Own struct with `extension MyStruct: MetalArgument {}` | `struct` with same layout | value |
| `[T]` | `device const T *` | copied into a buffer (`Double` / `Complex<Double>` converted) |
| `UnsafeBufferPointer<T>`, `UnsafeMutableBufferPointer<T>` | `device const T *` | copied into a buffer (`Double` / `Complex<Double>` converted) |
| `MetalBuffer<T>` | `device T *` | buffer, no copy |

Values up to 4 KB are passed with `setBytes`, larger values in a buffer.

### Own structs

A struct can be passed if it only contains plain values (numbers, SIMD vectors, tuples,
other plain structs). Classes, `String`, `Array` and other types with references are rejected at runtime.

```swift
struct ComputeParameters {
    var maxIterations: UInt32
    var bailout: Float
    var center: SIMD2<Float>
}
extension ComputeParameters: MetalArgument {}
```

```metal
struct ComputeParameters {
    uint   maxIterations;
    float  bailout;
    float2 center;
};
```

The layout of the Swift struct must match the Metal struct. Pitfalls:

- **`Int` is not supported.** Swift `Int` has 64 bit, Metal `int` 32 bit. Use `Int32` / `UInt32`,
  or `Int64` for Metal `long`. `Int` deliberately doesn't conform to `MetalArgument`, so passing it
  is a compile error. Arrays and buffers of `Int` or `UInt` are stopped at runtime.
- **No `Double` fields in structs.** Only `Double` arguments are converted, not `Double` fields
  inside a struct. Use `SIMD2<Float>` (Metal64 `Float2`) fields and convert with `Float2(value)`.
- **`SIMD3<Float>` / `float3` take 16 bytes**, not 12, in Swift and in Metal. Use them on both
  sides or use `packed_float3` in Metal together with three `Float` fields in Swift.
- Keep the fields in the same order on both sides. Swift doesn't reorder struct fields, so with the same order and types the alignment padding matches.

### Literals

Numeric literals are `Double` by default. A literal `1.5` is therefore passed as `float2` (f64),
not as `float`. Write `Float(1.5)` or `Int32(10)` when the kernel expects `float` or `int`.

## Double and Complex&lt;Double&gt;

Metal doesn't support `double`. MetalCompute converts `Double` and `Complex<Double>` arguments
to the 64 bit emulation format of [Metal64](https://github.com/zapccu/Metal64):

| Swift | Metal | Metal64 type | Layout |
|-------|-------|--------------|--------|
| `Double` | `float2` | `f64` | (hi, lo) |
| `Complex<Double>` | `float4` | `c64` | (re.hi, re.lo, im.hi, im.lo) |

`hi` is the value rounded to `Float`, `lo` the rounded rest. `hi + lo` has about 48 bit
(14 decimal digits) precision. Infinity and values beyond the `Float` range get `lo = 0`.

This works for single values, arrays and buffer pointers:

```swift
let center = Complex<Double>(-0.743643887037151, 0.131825904205330)
let points: [Complex<Double>] = ...

let result: [SIMD2<Float>] = try await kernel.run(points, center, Double(zoom), count: points.count)
```

```metal
#include "c64.h"

kernel void calculate(device const float4 *points [[buffer(0)]],
                      constant float4     &center [[buffer(1)]],
                      constant float2     &zoom   [[buffer(2)]],
                      device float2       *result [[buffer(3)]],
                      uint index [[thread_position_in_grid]])
{
    c64 c = c64(points[index]) * f64(zoom) + c64(center);
    ...
    result[index] = value.v;      // f64 result
}
```

Results are not converted back automatically: use `SIMD2<Float>` / `SIMD4<Float>` (Metal64
`Float2` / `Complex2`) as output type and convert with Metal64:

```swift
let doubles = result.map { Double($0) }                 // Float2 -> Double
let complexes = result.map { ComplexDouble($0) }        // Complex2 -> Complex<Double>
```

`MetalBuffer<Double>` and output type `Double` are not allowed (stopped at runtime), because their
memory would be passed 1:1. Use `MetalBuffer<SIMD2<Float>>` instead.

The swift-numerics dependency of MetalCompute only provides the `Complex` type, Metal64 isn't required.

## Reusing buffers

For repeated runs with the same size, e.g. redrawing a fractal while zooming, buffers can be
created once and reused. `MetalBuffer` lives in shared memory, the CPU reads and writes it directly.

```swift
let points = try metal.makeBuffer(count: width * height, of: SIMD4<Float>.self)
let output = try metal.makeBuffer(count: width * height, of: MandelbrotResultMtl.self)

// for every frame
points.copy(from: newPoints)                  // or write via points.pointer / points[i]
try await kernel.run(points, parameters, count: points.count, into: output)
let result = output.map { $0.mandelbrotResult }
```

Don't modify a buffer while a kernel using it is running.

## Example: Mandelbrot set with Metal64

[Metal64](https://github.com/zapccu/Metal64) provides 64 bit floating point emulation for Metal.
Its `Complex2` type (`SIMD4<Float>`, Metal `float4` / `c64`) can be passed directly.

```swift
import Metal64
import MetalCompute

extension ComputeParameters: MetalArgument {}

enum MandelbrotGPU {
    static let metal = try? MetalCompute()

    nonisolated static func calculateMetal(P: DynamicParameters<PT>) async -> CombinedResult {
        let cplx2Grid: LinearBuffer<Complex2> = P.dimensions.asDouble.mapScreenCoordinates(
            resolution: P.resolution, osSize: P.osSize)

        do {
            guard let metal else { throw MetalComputeError.noDevice }

            let result = try await metal.kernel("mandelbrot").run(
                cplx2Grid.pointer, P.computeParameters, P.potential,
                count: cplx2Grid.size,
                initialValue: MandelbrotResult.initialValue.mandelbrotResultMtl)

            return CombinedResult(
                Matrix<MandelbrotResult>((P.osImageHeight, P.osImageWidth), grid: result.map { $0.mandelbrotResult }),
                CalculationInfo(state: .completed)
            )
        }
        catch {
            return CombinedResult(nil, CalculationInfo(state: .error, message: "\(error)"))
        }
    }
}
```

```metal
#include "c64.h"

kernel void mandelbrot(device const float4      *points    [[buffer(0)]],
                       constant ComputeParameters &params  [[buffer(1)]],
                       constant float             &potential [[buffer(2)]],
                       device MandelbrotResultMtl *result  [[buffer(3)]],
                       uint index [[thread_position_in_grid]])
{
    c64 c = c64(points[index]);
    ...
}
```

## Errors

All functions throw `MetalComputeError` with a readable `description`, e.g. when the kernel
function doesn't exist, a buffer can't be allocated or the GPU reports an execution error.

## Migration from MetalCompute in Metal64

| Metal64 (old) | MetalCompute |
|---|---|
| `MetalCompute("fnc", count)` | `MetalCompute()` once, then `metal.kernel("fnc")` |
| `prepareComputeEncoder()` | not needed |
| `addBuffer(pointer, .inputBuffer)` / `addArray(array)` | pass `pointer` / `array` to `run` |
| `addStruct(value)` / `addValue(value)` | pass `value` to `run` (struct: `extension T: MetalArgument {}`) |
| `compute(initValue)` → `[T]?` | `try await run(..., count:, initialValue:)` → `[T]`, throws on errors |

The buffer order is the same as before: arguments in call order, result buffer last.

## Tests

The tests compile their kernels at runtime and run on any Mac with Metal: *Product → Test* in Xcode
or `swift test`.
