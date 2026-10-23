//
//  mandelbrot.metal
//  MetalTest
//
//  Created by Dirk Braner on 09.08.26.
//

#include <metal_stdlib>
#include "c64.h"
#include "mandelbrot.h"

using namespace metal;


//
//  Calculate Mandelbrot set, using classes c64 and f64
//

// Mandelbrot iteration
MandelbrotResult iterate(c64 C, device const MandelbrotParameters *MP) {
    c64 Z;
    c64 D = 0.0;
    c64 normal;
    f64 logaZ = 0.0;
    f64 logZn = 0.0;
    f64 logRatio = 0.0;
    f64 nZ = 0.0, aZ = 0.0;
    f64 distance = 0.0;
    f64 potential = 0.0;
    f64 smoothIter = 0.0;
    f64 div_2_logBailout = f64(MP->div_2_logBailout);
    int i;
        
    for(i=0; i<=MP->maxIter; i++) {
        Z = sqr(Z) + C;
        
        nZ = norm(Z);
        if (nZ.v.x > MP->bailout) {
            // Distance calculation
            aZ = sqrt(nZ);
            logaZ = log(aZ);
            distance = (aZ * logaZ / abs(D) * 0.5);
            
            logRatio = logaZ * div_2_logBailout;
            smoothIter =  f64(1.0) - log(logRatio) * F64_1_LOG2 + f64(i);
            
            // Potential calculation
            logZn = log(nZ) / 2.0;
            potential = log(logZn / F64_LOG2) / F64_LOG2;

            normal = Z / D;
            return { nZ.v, aZ.v, logaZ.v, smoothIter.v, normal.real().v, normal.imaginary().v, distance.v, potential.v, R_BAILOUT };
        }
        
        D = (Z + Z) * D + 1.0;
    }
    
    return { nZ.v, aZ.v, logaZ.v, f64(MP->maxIter).v, normal.real().v, normal.imaginary().v, distance.v, potential.v, R_MAX_ITERATIONS };
}


// Kernel function for Mandelbrot calculation
kernel void mandelbrot(device const float4 *C,                          // Array with complex coordinates
                       device const MandelbrotParameters *MP,           // Compute parameters
                       device MandelbrotResult *resultArray,            // Calculation results
                       uint   index [[ thread_position_in_grid ]])
{
    // Result array contains a MandelbrotResult element for each point in array C
    resultArray[index] = iterate(c64(C[index]), MP);
}


