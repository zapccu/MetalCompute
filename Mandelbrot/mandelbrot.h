//
//  mandelbrot.h
//  MetalTest
//
//  Created by Dirk Braner on 09.08.26.
//

#ifndef __MANDELBROT_H
#define __MANDELBROT_H

// Iteration parameters
struct MandelbrotParameters {
    int32_t maxIter;
    float bailout;
    float2 div_2_logBailout;
};

#define R_DEFAULT_COLOR 0
#define R_MAX_ITERATIONS 1
#define R_BAILOUT 2
#define R_INSIDE 4

// Iteration results
struct MandelbrotResult {
    float2 nZ;
    float2 aZ;
    float2 logAz;
    float2 iterations;
    float2 normalReal;
    float2 normalImag;
    float2 distance;
    float2 potential;
    int32_t flag;
};


#endif

