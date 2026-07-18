#pragma once

#include <metal_stdlib>
using namespace metal;

constant int SCALE = 1 << 24;

#define PE_WAVES 3
#define PE_DIM   (4 * PE_WAVES)

// Latent pyramid: K_GRIDS levels, adjacent pair (nm, nm+1) sampled per output mip.
// F_PER_GRID channels per level. MLP still sees 2 * F_PER_GRID grid features.
#define K_GRIDS     8
#define F_PER_GRID  8
#define MAX_LODS    13

struct SPDConstants {
    uint numWorkgroups;
    uint mipCount;
};

struct StepConstants {
    uint  kBatch;
    uint  pyramidOffsets[K_GRIDS + 1];
    uint  pyramidSizes  [K_GRIDS];
    uint  offsetW1;
    uint  offsetB1;
    uint  offsetW2;
    uint  offsetB2;
    uint  offsetW3;
    uint  offsetB3;
    uint  total;
    uint  bits;
    float q;
    float lo;
    float hi;
    uint  adamOffset; // for MLP-only fine-tune after fake quantization
    uint  inferLod;
    uint  neuralMipForLod[MAX_LODS];
};

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
    // 2^30 leaves 2x headroom below INT32_MAX per single write
    float scaled = clamp(val * float(SCALE), -1073741824.0f, 1073741824.0f);
    atomic_fetch_add_explicit(slot, int(rint(scaled)), memory_order_relaxed);
}

inline float hash01(uint seed, uint corner, uint i) {
    uint x = seed * 1664525u + corner * 1013904223u + i * 2654435761u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x) * (1.0f / 4294967296.0f);  // [0, 1)
}

inline void bilinear_sample(device const float* grid,
                            uint W, uint F,
                            int ix0, int iy0, float fx, float fy,
                            thread float* w,
                            thread uint*  c,
                            thread float* features,
                            float q,
                            uint  seed) {
    w[0] = (1.0f - fx) * (1.0f - fy);  // w00
    w[1] =         fx  * (1.0f - fy);  // w01
    w[2] = (1.0f - fx) *         fy;   // w10
    w[3] =         fx  *         fy;   // w11
    
    c[0] = (uint(iy0)     * W + uint(ix0))     * F;
    c[1] = (uint(iy0)     * W + uint(ix0 + 1)) * F;
    c[2] = (uint(iy0 + 1) * W + uint(ix0))     * F;
    c[3] = (uint(iy0 + 1) * W + uint(ix0 + 1)) * F;
    
    for (uint feat = 0; feat < F; feat++) {
        float acc = 0;
        for (uint corner = 0; corner < 4; corner++) {
            float g = grid[c[corner] + feat];
            if (q != 0.0f) {
                g += q * (hash01(seed, corner, feat) - 0.5f); // [-q/2, q/2)
            }
            acc += w[corner] * g;
        }
        features[feat] = acc;
    }
}

inline void pe_encode(float2 posf, thread float* pe) {
    for (uint i = 0; i < PE_WAVES; i++) {
        pe[4*i + 0] = fract(posf.x)         * 2.0 - 1.0; // saw x
        pe[4*i + 1] = fract(posf.y)         * 2.0 - 1.0; // saw y
        pe[4*i + 2] = fract(posf.x + 0.25)  * 2.0 - 1.0; // saw shift x
        pe[4*i + 3] = fract(posf.y + 0.25)  * 2.0 - 1.0; // saw shift y
        posf *= 2.0;
    }
}
    
inline void mlp_forward(device const float* params,
                        uint off_w1, uint off_b1,
                        uint off_w2, uint off_b2,
                        uint off_w3, uint off_b3,
                        uint fan_in, uint hidden, uint out_dim,
                        thread const float* features,
                        thread float* pre1, thread float* hid1,
                        thread float* pre2, thread float* hid2,
                        thread float* pred) {
    // Linear1 + hardGELU
    for (uint h = 0; h < hidden; h++) {
        float acc = params[off_b1 + h];
        for (uint i = 0; i < fan_in; i++) {
            acc += params[off_w1 + i * hidden + h] * features[i];
        }
        pre1[h] = acc;
        hid1[h] = hard_gelu(pre1[h]);
    }

    // Linear2 + hardGELU
    for (uint h = 0; h < hidden; h++) {
        float acc = params[off_b2 + h];
        for (uint i = 0; i < hidden; i++) {
            acc += params[off_w2 + i * hidden + h] * hid1[i];
        }
        pre2[h] = acc;
        hid2[h] = hard_gelu(pre2[h]);
    }

    // Linear3
    for (uint k = 0; k < out_dim; k++) {
        float acc = params[off_b3 + k];
        for (uint h = 0; h < hidden; h++) {
            acc += params[off_w3 + h * out_dim + k] * hid2[h];
        }
        pred[k] = acc;
    }
}
