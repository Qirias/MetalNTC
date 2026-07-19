#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define POS_SCALE (float(SRC_W) / 8.0f)

#define SAMPLE_X      0
#define SAMPLE_Y      1
#define SAMPLE_LOD    2
#define SAMPLE_LOSS   3
#define SAMPLE_STRIDE 4

kernel void grid_mlp_train(device               float*                                  params  [[buffer(0)]],
                           device               float*                                  samples [[buffer(1)]],
                                    constant    StepConstants&                          consts  [[buffer(2)]],
                                                texture2d_array<float, access::read>    pyramid [[texture(0)]],
                                                uint                                    gid     [[thread_position_in_grid]]) {
    if (gid >= consts.kBatch) return;

    device atomic_int* grads = reinterpret_cast<device atomic_int*>(params + consts.total);

    uint sample_base = gid * SAMPLE_STRIDE;
    int x = int(samples[sample_base + SAMPLE_X]);
    int y = int(samples[sample_base + SAMPLE_Y]);
    uint lod = uint(samples[sample_base + SAMPLE_LOD]);
    uint srcWL = max(uint(SRC_W) >> lod, 1u);
    uint srcHL = max(uint(SRC_H) >> lod, 1u);

    uint2 coord = uint2(uint(x), uint(y));
    float gt[K_OUT_MAX];
    for (uint k = 0; k < K_OUT_MAX; k++) {
        gt[k] = 0.0;
    }
    for (uint slice = 0; slice < consts.nSlices; slice++) {
        float4 val  = pyramid.read(coord, slice, lod);
        uint offset = consts.sliceChannelOffsets[slice];
        uint num_ch = consts.sliceChannels[slice];
        // only 3 and 1 channels supported
        if (num_ch == 3) {
            gt[offset + 0] = val.r;
            gt[offset + 1] = val.g;
            gt[offset + 2] = val.b;
        } else {
            gt[offset] = val.r;
        }
    }

    uint neural_mip = consts.neuralMipForLod[lod];
    
    uint g0_offset  = consts.pyramidOffsets[neural_mip];
    uint g1_offset  = consts.pyramidOffsets[neural_mip + 1];
    uint g0_size    = consts.pyramidSizes[neural_mip];
    uint g1_size    = consts.pyramidSizes[neural_mip + 1];

    float ix0 = float(x) * float(g0_size - 1) / float(srcWL - 1);;
    float iy0 = float(y) * float(g0_size - 1) / float(srcHL - 1);
    int ix0_0 = min(int(floor(ix0)), int(g0_size) - 2);
    int iy0_0 = min(int(floor(iy0)), int(g0_size) - 2);
    float fx0 = ix0 - float(ix0_0);
    float fy0 = iy0 - float(iy0_0);
    
    // G1
    float ix1 = float(x) * float(g1_size - 1) / float(srcWL - 1);;
    float iy1 = float(y) * float(g1_size - 1) / float(srcHL - 1);
    int ix0_1 = min(int(floor(ix1)), int(g1_size) - 2);
    int iy0_1 = min(int(floor(iy1)), int(g1_size) - 2);
    float fx1 = ix1 - float(ix0_1);
    float fy1 = iy1 - float(iy0_1);

    float w0[4];
    float w1[4];
    uint  c0[4];
    uint  c1[4];
    float features[F_IN];

    bilinear_sample(params + g0_offset, g0_size, F_PER_GRID, ix0_0, iy0_0, fx0, fy0, w0, c0, features               , consts.q, gid);
    bilinear_sample(params + g1_offset, g1_size, F_PER_GRID, ix0_1, iy0_1, fx1, fy1, w1, c1, features + F_PER_GRID  , consts.q, gid);

    float2 uv   = float2(float(x) / float(srcWL - 1), float(y) / float(srcHL - 1));
    float2 posf = uv * POS_SCALE;
    pe_encode(posf, features + F_TOTAL);

    features[F_TOTAL + PE_DIM] = float(lod) / float(MAX_LODS - 1);
    
    // ========
    // Forward
    // ========
    float pre1[K_HIDDEN];
    float pre2[K_HIDDEN];
    float hid1[K_HIDDEN];
    float hid2[K_HIDDEN];
    float pred[K_OUT_MAX];
    mlp_forward(params,
                consts.offsetW1, consts.offsetB1,
                consts.offsetW2, consts.offsetB2,
                consts.offsetW3, consts.offsetB3,
                F_IN, K_HIDDEN, K_OUT_MAX,
                features,
                pre1, hid1, pre2, hid2, pred);

    // Loss + d_pred
    float diff[K_OUT_MAX];
    float d_pred[K_OUT_MAX];
    float loss = 0;
    for (uint k = 0; k < consts.kOut; k++) {
        diff[k] = pred[k] - gt[k];
        loss += diff[k]*diff[k];
        d_pred[k] = 2 * diff[k] / float(consts.kOut) / float(consts.kBatch);
    }
    samples[sample_base + SAMPLE_LOSS] = loss / float(consts.kOut);

    // ========
    // Backward
    // ========
    // Linear3
    float d_hid2[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_hid2[h] = 0;
    }

    for (uint k = 0; k < consts.kOut; k++) {
        atomic_add_fixed(&grads[consts.offsetB3 + k], d_pred[k]);
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        for (uint k = 0; k < consts.kOut; k++) {
            atomic_add_fixed(&grads[consts.offsetW3 + h * K_OUT_MAX + k], d_pred[k] * hid2[h]);
            d_hid2[h] += params[consts.offsetW3 + h * K_OUT_MAX + k] * d_pred[k];
        }
    }
    
    // hardGELU
    float d_pre2[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_pre2[h] = d_hid2[h] * hard_gelu_prime(pre2[h]);
    }
    
    // Linear2
    float d_hid1[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_hid1[h] = 0;
    }

    for (uint k = 0; k < K_HIDDEN; k++) {
        atomic_add_fixed(&grads[consts.offsetB2 + k], d_pre2[k]);
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        for (uint k = 0; k < K_HIDDEN; k++) {
            atomic_add_fixed(&grads[consts.offsetW2 + h * K_HIDDEN + k], d_pre2[k] * hid1[h]);
            d_hid1[h] += params[consts.offsetW2 + h * K_HIDDEN + k] * d_pre2[k];
        }
    }

    // hardGELU
    float d_pre1[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_pre1[h] = d_hid1[h] * hard_gelu_prime(pre1[h]);
    }

    // Linear1
    float d_feats[F_IN];
    for (uint i = 0; i < F_IN; i++) {
        d_feats[i] = 0;
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        atomic_add_fixed(&grads[consts.offsetB1 + h], d_pre1[h]);
    }

    for (uint i = 0; i < F_IN; i++) {
        for (uint h = 0; h < K_HIDDEN; h++) {
            atomic_add_fixed(&grads[consts.offsetW1 + i * K_HIDDEN + h], d_pre1[h] * features[i]);
            d_feats[i] += params[consts.offsetW1 + i * K_HIDDEN + h] * d_pre1[h];
        }
    }

    for (uint i = 0; i < F_PER_GRID; i++) {
        for (uint corner = 0; corner < 4; corner++) {
            atomic_add_fixed(&grads[g0_offset + c0[corner] + i], w0[corner] * d_feats[i]);
        }
    }

    for (uint i = 0; i < F_PER_GRID; i++) {
        for (uint corner = 0; corner < 4; corner++) {
        atomic_add_fixed(&grads[g1_offset + c1[corner] + i], w1[corner] * d_feats[F_PER_GRID + i]);
        }
    }
}
