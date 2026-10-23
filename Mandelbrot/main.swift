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

/// Mandelbrot calculation parameters (Metal layout)
struct MetalParameters: MetalArgument {
    let maxIter: Int32
    let bailout: Float
    let div_2_logBailout: Float2
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

/// Mandelbrot calculation result (Metal layout)
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


// Image dimensions
let width: Int = 1024
let height: Int = 1024
let count = width * height

// Pixel distance
let dx: Double = 3.0 / Double(width)
let dy: Double = 3.0 / Double(height)

// Initialize calculation parameters
let parameters = Parameters(maxIter: 500, bailout: 1e6, div_2_logBailout: 2.0 / log(2.0))
let computeParameters = MetalParameters(maxIter: Int32(parameters.maxIter), bailout: Float(parameters.bailout), div_2_logBailout: Float2(2.0 / log(2.0)))

// Init an array with complex points to be calculated
var C: [Complex2] = Array(repeating: Complex2(0.0, 0.0), count: width * height)
for y in 0..<height {
    let y0: Double = -1.5 + dy * Double(y)
    for x in 0..<width {
        let x0: Double = -2.25 + dx * Double(x)
        C[y * width + x] = Complex2(x0, y0)
    }
}

do {
    // Setup Metal and kernel
    let metal = try MetalCompute()
    let kernel = try metal.kernel("mandelbrot")
    
    // Calculate Mandelbrot set
    let result: [MetalResult] = try await kernel.run(C, computeParameters, count: C.count)

    // Print 4 results starting from x = 200, y = 200
    let offset = 200 * width + 200
    for i in 0..<3 {
        print("\(Complex<Double>(C[i+offset])) = nz: \(Double(result[i+offset].nZ)), aZ: \(Double(result[i+offset].aZ)), logaZ: \(Double(result[i+offset].logaZ)), iterations: \(Double(result[i+offset].iterations)), distance: \(Double(result[i+offset].distance))")
    }
}
catch {
    print(error)
}



