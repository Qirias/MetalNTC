#pragma once

#include "../../NTCShared/include/ntc_constants.h"
using namespace metal;

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
    uint  kOut;
    uint  nSlices;
    uint  sliceChannels       [K_OUT_MAX];
    uint  sliceChannelOffsets [K_OUT_MAX];
    uint  srcW;
    uint  srcH;
    uint  mipCount;   // log2(srcW) + 1
    float posScale;   // srcW / 8; base frequency of the positional encoding
};

// a cheap piecewise approximation of GELU
inline float hard_gelu(float x) {
    return 0.5f * x * (1.0f + clamp(x * 0.5f, -1.0f, 1.0f));
}

inline half hard_gelu(half x) {
    return 0.5h * x * (1.0h + clamp(x * 0.5h, -1.0h, 1.0h));
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
                        thread half* pre1, thread half* hid1,
                        thread half* pre2, thread half* hid2,
                        thread half* pred) {
    half feat_h[F_IN];
    for (uint i = 0; i < fan_in; i++) {
        feat_h[i] = half(features[i]);
    }

    // Linear1 + hardGELU
    for (uint h = 0; h < hidden; h++) {
        half acc = half(params[off_b1 + h]);
        for (uint i = 0; i < fan_in; i++) {
            acc += half(params[off_w1 + h * fan_in + i]) * feat_h[i];
        }
        pre1[h] = acc;
        hid1[h] = hard_gelu(acc);
    }

    // Linear2 + hardGELU
    for (uint h = 0; h < hidden; h++) {
        half acc = half(params[off_b2 + h]);
        for (uint i = 0; i < hidden; i++) {
            acc += half(params[off_w2 + h * hidden + i]) * hid1[i];
        }
        pre2[h] = acc;
        hid2[h] = hard_gelu(acc);
    }

    // Linear3
    for (uint k = 0; k < out_dim; k++) {
        half acc = half(params[off_b3 + k]);
        for (uint h = 0; h < hidden; h++) {
            acc += half(params[off_w3 + k * hidden + h]) * hid2[h];
        }
        pred[k] = acc;
    }
}

inline void sample_latent_grid(texture2d_array<float> latents,
                               sampler                latentSampler,
                               float2 uv, uint mip, uint gridSize,
                               float scale, float bias,
                               thread float* out /* F_PER_GRID */) {
    float  gridRes  = float(gridSize);
    float2 sampleUV = (uv * (gridRes - 1.0f) + 0.5f) / gridRes;

    for (uint slice = 0; slice < F_PER_GRID / 4; slice++) {
        float4 sampled     = latents.sample(latentSampler, sampleUV, slice, level(float(mip)));
        float4 dequantized = sampled * scale + bias;
        out[slice * 4 + 0] = dequantized.x;   // .r
        out[slice * 4 + 1] = dequantized.y;   // .g
        out[slice * 4 + 2] = dequantized.z;   // .b
        out[slice * 4 + 3] = dequantized.w;   // .a
    }
}

inline void mlp_forward_h(device const half* mlp,
                          uint off_w1, uint off_b1,
                          uint off_w2, uint off_b2,
                          uint off_w3, uint off_b3,
                          uint fan_in, uint hidden, uint out_dim,
                          thread const float* features,
                          thread half* hid1, thread half* hid2,
                          thread half* pred) {
    
    // pack the padded input vector into aligned half4 groups
    half4 input4[F_IN / 4];
    for (uint group = 0; group < fan_in / 4; group++) {
        uint base = group * 4;
        input4[group] = half4(features[base + 0], features[base + 1],
                              features[base + 2], features[base + 3]);
    }

    // Linear1 + hardGELU
    // one output row at a time
    for (uint outNeuron = 0; outNeuron < hidden; outNeuron++) {
        device const half4* weightRow = (device const half4*)(mlp + off_w1 + outNeuron * fan_in);
        half4 rowAcc4 = half4(0.0h);
        for (uint group = 0; group < fan_in / 4; group++) {
            rowAcc4 += weightRow[group] * input4[group];
        }
        half rowSum = mlp[off_b1 + outNeuron] + rowAcc4.x + rowAcc4.y + rowAcc4.z + rowAcc4.w;
        hid1[outNeuron] = hard_gelu(rowSum);
    }

    // pack into half4
    half4 hidden4[K_HIDDEN / 4];
    for (uint group = 0; group < hidden / 4; group++) {
        uint base = group * 4;
        hidden4[group] = half4(hid1[base + 0], hid1[base + 1],
                               hid1[base + 2], hid1[base + 3]);
    }

    // Linear2 + hardGELU
    for (uint outNeuron = 0; outNeuron < hidden; outNeuron++) {
        device const half4* weightRow = (device const half4*)(mlp + off_w2 + outNeuron * hidden);
        half4 rowAcc4 = half4(0.0h);
        for (uint group = 0; group < hidden / 4; group++) {
            rowAcc4 += weightRow[group] * hidden4[group];
        }
        half rowSum = mlp[off_b2 + outNeuron] + rowAcc4.x + rowAcc4.y + rowAcc4.z + rowAcc4.w;
        hid2[outNeuron] = hard_gelu(rowSum);
    }

    // pack into half4
    for (uint group = 0; group < hidden / 4; group++) {
        uint base = group * 4;
        hidden4[group] = half4(hid2[base + 0], hid2[base + 1],
                               hid2[base + 2], hid2[base + 3]);
    }

    // Linear3
    for (uint outChannel = 0; outChannel < out_dim; outChannel++) {
        device const half4* weightRow = (device const half4*)(mlp + off_w3 + outChannel * hidden);
        half4 rowAcc4 = half4(0.0h);
        for (uint group = 0; group < hidden / 4; group++) {
            rowAcc4 += weightRow[group] * hidden4[group];
        }
        pred[outChannel] = mlp[off_b3 + outChannel] + rowAcc4.x + rowAcc4.y + rowAcc4.z + rowAcc4.w;
    }
}

inline void ntc_decode_quant(float2                   uv,
                             uint                     lod,
                             texture2d_array<float>   latents,
                             sampler                  latentSampler,
                             float                    gridScale,
                             float                    gridBias,
                             device const half*       mlp,
                             constant StepConstants&  consts,
                             thread half*             pred) {
    uint neural_mip = consts.neuralMipForLod[lod];
    uint g0_size    = consts.pyramidSizes[neural_mip];
    uint g1_size    = consts.pyramidSizes[neural_mip + 1];

    float features[F_IN];

    sample_latent_grid(latents, latentSampler, uv, neural_mip,     g0_size,
                       gridScale, gridBias, features);
    sample_latent_grid(latents, latentSampler, uv, neural_mip + 1, g1_size,
                       gridScale, gridBias, features + F_PER_GRID);

    float2 posf = uv * consts.posScale;
    pe_encode(posf, features + F_TOTAL);
    features[F_TOTAL + PE_DIM] = float(lod) / float(MAX_LODS - 1);
    for (uint i = F_IN_RAW; i < F_IN; i++) {
        features[i] = 0.0f;   // zero padded lanes
    }

    half hid1[K_HIDDEN];
    half hid2[K_HIDDEN];

    mlp_forward_h(mlp,
                  consts.offsetW1, consts.offsetB1,
                  consts.offsetW2, consts.offsetB2,
                  consts.offsetW3, consts.offsetB3,
                  F_IN, K_HIDDEN, consts.kOut,
                  features,
                  hid1, hid2, pred);
}

inline void ntc_decode(float2                    uv,
                       uint                      lod,
                       device const float*       params,
                       constant StepConstants&   consts,
                       thread half*              pred) {
    uint neural_mip = consts.neuralMipForLod[lod];
    uint g0_offset  = consts.pyramidOffsets[neural_mip];
    uint g1_offset  = consts.pyramidOffsets[neural_mip + 1];
    uint g0_size    = consts.pyramidSizes[neural_mip];
    uint g1_size    = consts.pyramidSizes[neural_mip + 1];

    float ix0   = uv.x * float(g0_size - 1);
    float iy0   = uv.y * float(g0_size - 1);
    int   ix0_0 = min(int(floor(ix0)), int(g0_size) - 2);
    int   iy0_0 = min(int(floor(iy0)), int(g0_size) - 2);
    float fx0   = ix0 - float(ix0_0);
    float fy0   = iy0 - float(iy0_0);

    float ix1   = uv.x * float(g1_size - 1);
    float iy1   = uv.y * float(g1_size - 1);
    int   ix0_1 = min(int(floor(ix1)), int(g1_size) - 2);
    int   iy0_1 = min(int(floor(iy1)), int(g1_size) - 2);
    float fx1   = ix1 - float(ix0_1);
    float fy1   = iy1 - float(iy0_1);

    float w0[4]; float w1[4];
    uint  c0[4]; uint  c1[4];
    float features[F_IN];

    bilinear_sample(params + g0_offset, g0_size, F_PER_GRID,
                    ix0_0, iy0_0, fx0, fy0,
                    w0, c0, features,
                    0.0f, 0u);
    bilinear_sample(params + g1_offset, g1_size, F_PER_GRID,
                    ix0_1, iy0_1, fx1, fy1,
                    w1, c1, features + F_PER_GRID,
                    0.0f, 0u);

    float2 posf = uv * consts.posScale;
    pe_encode(posf, features + F_TOTAL);
    features[F_TOTAL + PE_DIM] = float(lod) / float(MAX_LODS - 1);
    for (uint i = F_IN_RAW; i < F_IN; i++) {
        features[i] = 0.0f;   // zero padded lanes
    }

    half pre1[K_HIDDEN];
    half pre2[K_HIDDEN];
    half hid1[K_HIDDEN];
    half hid2[K_HIDDEN];
    mlp_forward(params,
                consts.offsetW1, consts.offsetB1,
                consts.offsetW2, consts.offsetB2,
                consts.offsetW3, consts.offsetB3,
                F_IN, K_HIDDEN, K_OUT_MAX,
                features,
                pre1, hid1, pre2, hid2, pred);
}
