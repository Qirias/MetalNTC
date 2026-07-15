#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define GRID_H1  1024
#define GRID_W1  1024
#define GRID_F1  8
#define GRID1_TOTAL  (GRID_H1 * GRID_W1 * GRID_F1)

#define GRID_H2  512
#define GRID_W2  512
#define GRID_F2  8
#define GRID2_TOTAL (GRID_H2 * GRID_W2 * GRID_F2)

#define GRID_F_TOTAL (GRID_F1 + GRID_F2)

#define K_HIDDEN 64
#define K_OUT    3

#define SRC_W 4096
#define SRC_H 4096

#define F_IN      (GRID_F_TOTAL + PE_DIM)
#define POS_SCALE (float(SRC_W) / 8.0f)

#define SAMPLE_X      0
#define SAMPLE_Y      1
#define SAMPLE_LOSS   2
#define SAMPLE_STRIDE 3

kernel void grid_mlp_train(device               float*                          params  [[buffer(0)]],
                           device               float*                          samples [[buffer(1)]],
                                    constant    StepConstants&                  consts  [[buffer(2)]],
                                                texture2d<float, access::read>  pyramid [[texture(0)]],
                                                uint                            gid     [[thread_position_in_grid]]) {
    if (gid >= consts.kBatch) return;

    device atomic_int* grads = reinterpret_cast<device atomic_int*>(params + consts.total);

    uint sample_base = gid * SAMPLE_STRIDE;
    int x = int(samples[sample_base + SAMPLE_X]);
    int y = int(samples[sample_base + SAMPLE_Y]);

    float4 gt4 = pyramid.read(uint2(uint(x), uint(y)), 0);
    float gt[K_OUT];
    gt[0] = gt4.r;
    gt[1] = gt4.g;
    gt[2] = gt4.b;

    float ix1 = float(x) * float(GRID_W1 - 1) / float(SRC_W - 1);
    float iy1 = float(y) * float(GRID_H1 - 1) / float(SRC_H - 1);
    int iy0_1 = int(floor(iy1));
    int ix0_1 = int(floor(ix1));
    iy0_1 = min(iy0_1, GRID_H1 - 2);
    ix0_1 = min(ix0_1, GRID_W1 - 2);
    float fy1 = iy1 - float(iy0_1);
    float fx1 = ix1 - float(ix0_1);
    
    float ix2 = float(x) * float(GRID_W2 - 1) / float(SRC_W - 1);
    float iy2 = float(y) * float(GRID_H2 - 1) / float(SRC_H - 1);
    int iy0_2 = int(floor(iy2));
    int ix0_2 = int(floor(ix2));
    iy0_2 = min(iy0_2, GRID_H2 - 2);
    ix0_2 = min(ix0_2, GRID_W2 - 2);
    float fy2 = iy2 - float(iy0_2);
    float fx2 = ix2 - float(ix0_2);

    float w1[4];
    float w2[4];
    uint  c1[4];
    uint  c2[4];
    float features[F_IN];
    bilinear_sample(params                  , GRID_W1, GRID_F1, ix0_1, iy0_1, fx1, fy1, w1, c1, features          , consts.qPerGrid[0], gid);
    bilinear_sample(params + consts.offsetG2, GRID_W2, GRID_F2, ix0_2, iy0_2, fx2, fy2, w2, c2, features + GRID_F1, consts.qPerGrid[1], gid);

    float2 uv   = float2(float(x) / float(SRC_W - 1), float(y) / float(SRC_H - 1));
    float2 posf = uv * POS_SCALE;
    pe_encode(posf, features + GRID_F_TOTAL);
    
    // ========
    // Forward
    // ========
    float pre1[K_HIDDEN];
    float pre2[K_HIDDEN];
    float hid1[K_HIDDEN];
    float hid2[K_HIDDEN];
    float pred[K_OUT];
    mlp_forward(params,
                consts.offsetW1, consts.offsetB1,
                consts.offsetW2, consts.offsetB2,
                consts.offsetW3, consts.offsetB3,
                F_IN, K_HIDDEN, K_OUT,
                features,
                pre1, hid1, pre2, hid2, pred);

    // Loss + d_pred
    float diff[K_OUT];
    float d_pred[K_OUT];
    float loss = 0;
    for (uint k = 0; k < K_OUT; k++) {
        diff[k] = pred[k] - gt[k];
        loss += diff[k]*diff[k];
        d_pred[k] = 2 * diff[k] / float(K_OUT) / float(consts.kBatch);
    }
    samples[sample_base + SAMPLE_LOSS] = loss / float(K_OUT);

    // ========
    // Backward
    // ========
    // Linear3
    float d_hid2[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_hid2[h] = 0;
    }

    for (uint k = 0; k < K_OUT; k++) {
        atomic_add_fixed(&grads[consts.offsetB3 + k], d_pred[k]);
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        for (uint k = 0; k < K_OUT; k++) {
            atomic_add_fixed(&grads[consts.offsetW3 + h * K_OUT + k], d_pred[k] * hid2[h]);
            d_hid2[h] += params[consts.offsetW3 + h * K_OUT + k] * d_pred[k];
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

    for (uint i = 0; i < GRID_F1; i++) {
        for (uint corner = 0; corner < 4; corner++) {
            atomic_add_fixed(&grads[c1[corner] + i], w1[corner] * d_feats[i]);
        }
    }

    for (uint i = 0; i < GRID_F2; i++) {
        for (uint corner = 0; corner < 4; corner++) {
            atomic_add_fixed(&grads[consts.offsetG2 + c2[corner] + i], w2[corner] * d_feats[GRID_F1 + i]);
        }
    }
}
