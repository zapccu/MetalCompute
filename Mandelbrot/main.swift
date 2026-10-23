//
//  main.swift
//
//  Demo for MetalCompute and Metal64
//
//  Created by Dirk Braner on 10.10.2026
//
//  Requires package Swift Numerics
//

import Foundation
import Metal
import ComplexModule
import Metal64
import MetalCompute

/// Mandelbrot calculation parameters
struct Parameters {
    let maxIter: Int
    let bailout: Double
    let div_2_logBailout: Double
}

/// Mandelbrot calculation parameters (Metal layout, see mandelbrot.h)
struct MetalParameters {
    let maxIter: Int32
    let bailout: Float
    let div_2_logBailout: Float2
}

/// Parameters are converted to the Metal layout when passed to the kernel
extension Parameters: MetalEncodable {
    func toMetal() -> MetalParameters {
        MetalParameters(maxIter: Int32(maxIter), bailout: Float(bailout), div_2_logBailout: div_2_logBailout.toMetal())
    }
}

/// Mandelbrot calculation result
struct Result {
    let nZ: Double
    let aZ: Double
    let logaZ: Double
    let iterations: Double
    let normalReal: Double
    let normalImag: Double
    let distance: Double
    let potential: Double
    let flag: Int
}

/// Mandelbrot calculation result (Metal layout, see mandelbrot.h)
struct MetalResult {
    let nZ: Float2
    let aZ: Float2
    let logaZ: Float2
    let iterations: Float2
    let normalReal: Float2
    let normalImag: Float2
    let distance: Float2
    let potential: Float2
    let flag: Int32
}

/// Results are converted from the Metal layout before run() returns
extension Result: MetalDecodable {
    init(fromMetal m: MetalResult) {
        self.init(nZ: Double(fromMetal: m.nZ), aZ: Double(fromMetal: m.aZ), logaZ: Double(fromMetal: m.logaZ),
                  iterations: Double(fromMetal: m.iterations),
                  normalReal: Double(fromMetal: m.normalReal), normalImag: Double(fromMetal: m.normalImag),
                  distance: Double(fromMetal: m.distance), potential: Double(fromMetal: m.potential),
                  flag: Int(m.flag))
    }
}


// Image dimensions
let width: Int = 1024
let height: Int = 1024
let count = width * height

// Pixel distance
let dx: Double = 3.0 / Double(width)
let dy: Double = 3.0 / Double(height)

// Initialize calculation parameters
let parameters = Parameters(maxIter: 500, bailout: 1e6, div_2_logBailout: 2.0 / log(2.0))

// Init an array with complex points to be calculated.
// Complex<Double> is converted to c64 (Metal float4) when passed to the kernel.
var C: [Complex<Double>] = Array(repeating: .zero, count: width * height)
for y in 0..<height {
    let y0: Double = -1.5 + dy * Double(y)
    for x in 0..<width {
        let x0: Double = -2.25 + dx * Double(x)
        C[y * width + x] = Complex(x0, y0)
    }
}

do {
    // Setup Metal and kernel
    let metal = try MetalCompute()
    let kernel = try metal.kernel("mandelbrot")
    
    // Calculate Mandelbrot set.
    // Arguments are checked against the kernel signature, results are converted to Result.
    let result: [Result] = try await kernel.run(C, parameters, count: C.count)

    // Print 3 results starting from x = 200, y = 200
    let offset = 200 * width + 200
    for i in 0..<3 {
        let r = result[i + offset]
        print("\(C[i + offset]) = nz: \(r.nZ), aZ: \(r.aZ), logaZ: \(r.logaZ), iterations: \(r.iterations), distance: \(r.distance)")
    }
}
catch {
    print(error)
}



