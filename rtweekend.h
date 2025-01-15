#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

// __device__ inline float random_float() {
//     // Returns a random real in [0,1).
//     return std::rand() / (RAND_MAX + 1.0);
// }

// __device__ inline float random_float(float min, float max) {
//     // Returns a random real in [min,max).
//     return min + (max-min)*random_float();
// }

// Common Headers

#include "interval.h"
#include "color.h"
#include "ray.h"
#include "vec3.h"

#endif