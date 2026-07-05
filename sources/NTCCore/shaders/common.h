#pragma once

#include <metal_stdlib>
using namespace metal;

constant int SCALE = 1 << 18;

// a cheap piecewise approximation of GELU
inline float hard_gelu(float x) {
    return 0.5f * x * (1.0f + clamp(x * 0.5f, -1.0f, 1.0f));
}

// https://en.wikipedia.org/wiki/Product_rule
// derived from the product rule on out = 0.5 * x * (1 + c)
// where c = clamp(x/2, -1, 1):
//   d(out)/dx = 0.5 * (1 + c)       (derivative of 0.5*x, times (1+c))
//             + 0.5 * x * dc/dx     (0.5*x, times derivative of (1+c))
// dc/dx is 0.5 when x is in [-2, 2] (clamp is just x/2 there), and 0
// outside. The in_band flag picks the right case.

inline float hard_gelu_prime(float x) {
    float c = clamp(x * 0.5f, -1.0f, 1.0f);
    float in_band = (x >= -2.0f && x <= 2.0f) ? 1.0f : 0.0f;
    return 0.5f * (1.0f + c) + 0.5f * x * 0.5f * in_band;
}

inline void atomic_add_fixed(device atomic_int* slot, float val) {
    atomic_fetch_add_explicit(slot, int(rint(val * float(SCALE))), memory_order_relaxed);
}

inline void bilinear_sample(device const float* grid,
                            uint W, uint F,
                            int ix0, int iy0, float fx, float fy,
                            thread float* w,
                            thread uint*  c,
                            thread float* features) {
    w[0] = (1.0f - fx) * (1.0f - fy);  // w00
    w[1] =         fx  * (1.0f - fy);  // w01
    w[2] = (1.0f - fx) *         fy;   // w10
    w[3] =         fx  *         fy;   // w11

    c[0] = (uint(iy0)     * W + uint(ix0))     * F;
    c[1] = (uint(iy0)     * W + uint(ix0 + 1)) * F;
    c[2] = (uint(iy0 + 1) * W + uint(ix0))     * F;
    c[3] = (uint(iy0 + 1) * W + uint(ix0 + 1)) * F;

    for (uint i = 0; i < F; i++) {
        features[i] = w[0]*grid[c[0]+i] + w[1]*grid[c[1]+i]
                    + w[2]*grid[c[2]+i] + w[3]*grid[c[3]+i];
    }
}
