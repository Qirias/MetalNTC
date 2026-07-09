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
#define GRID2_TOTAL  (GRID_H2 * GRID_W2 * GRID_F2)

#define GRID_F_TOTAL (GRID_F1 + GRID_F2)

#define K_HIDDEN 64
#define K_OUT    3
#define K_BATCH  1024

#define SRC_W 4096
#define SRC_H 4096

#define OFFSET_G1   0
#define OFFSET_G2   (GRID1_TOTAL)
#define OFFSET_W1   (OFFSET_G2 + GRID2_TOTAL)
#define OFFSET_B1   (OFFSET_W1 + GRID_F_TOTAL * K_HIDDEN)
#define OFFSET_W2   (OFFSET_B1 + K_HIDDEN)
#define OFFSET_B2   (OFFSET_W2 + K_HIDDEN * K_HIDDEN)
#define OFFSET_W3   (OFFSET_B2 + K_HIDDEN)
#define OFFSET_B3   (OFFSET_W3 + K_HIDDEN * K_OUT)
#define TOTAL       (OFFSET_B3 + K_OUT)

#define SAMPLE_X      0
#define SAMPLE_Y      1
#define SAMPLE_LOSS   2
#define SAMPLE_STRIDE 3

kernel void grid_mlp_train(device       float* params  [[buffer(0)]],
                           device       float* samples [[buffer(1)]],
                           device const float* source  [[buffer(2)]],
                                        uint   gid     [[thread_position_in_grid]]) {
    if (gid >= K_BATCH) return;

    device atomic_int* grads = reinterpret_cast<device atomic_int*>(params + TOTAL);

    uint sample_base = gid * SAMPLE_STRIDE;
    int x = int(samples[sample_base + SAMPLE_X]);
    int y = int(samples[sample_base + SAMPLE_Y]);

    uint src_base = (uint(y) * SRC_W + uint(x)) * K_OUT;
    float gt[K_OUT];
    for (uint ch = 0; ch < K_OUT; ch++) {
        gt[ch] = source[src_base + ch];
    }

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
    float features[GRID_F_TOTAL];
    bilinear_sample(params            , GRID_W1, GRID_F1, ix0_1, iy0_1, fx1, fy1, w1, c1, features);
    bilinear_sample(params + OFFSET_G2, GRID_W2, GRID_F2, ix0_2, iy0_2, fx2, fy2, w2, c2, features + GRID_F1);
    
    // ========
    // Forward
    // ========
    float pre1[K_HIDDEN];
    float pre2[K_HIDDEN];
    float hid1[K_HIDDEN];
    float hid2[K_HIDDEN];
    float pred[K_OUT];
    mlp_forward(params,
                OFFSET_W1, OFFSET_B1,
                OFFSET_W2, OFFSET_B2,
                OFFSET_W3, OFFSET_B3,
                GRID_F_TOTAL, K_HIDDEN, K_OUT,
                features,
                pre1, hid1, pre2, hid2, pred);

    // Loss + d_pred
    float diff[K_OUT];
    float d_pred[K_OUT];
    float loss = 0;
    for (uint k = 0; k < K_OUT; k++) {
        diff[k] = pred[k] - gt[k];
        loss += diff[k]*diff[k];
        d_pred[k] = 2 * diff[k] / float(K_OUT) / float(K_BATCH);
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
        atomic_add_fixed(&grads[OFFSET_B3 + k], d_pred[k]);
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        for (uint k = 0; k < K_OUT; k++) {
            atomic_add_fixed(&grads[OFFSET_W3 + h * K_OUT + k], d_pred[k] * hid2[h]);
            d_hid2[h] += params[OFFSET_W3 + h * K_OUT + k] * d_pred[k];
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
        atomic_add_fixed(&grads[OFFSET_B2 + k], d_pre2[k]);
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        for (uint k = 0; k < K_HIDDEN; k++) {
            atomic_add_fixed(&grads[OFFSET_W2 + h * K_HIDDEN + k], d_pre2[k] * hid1[h]);
            d_hid1[h] += params[OFFSET_W2 + h * K_HIDDEN + k] * d_pre2[k];
        }
    }

    // hardGELU
    float d_pre1[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_pre1[h] = d_hid1[h] * hard_gelu_prime(pre1[h]);
    }

    // Linear1
    float d_feats[GRID_F_TOTAL];
    for (uint i = 0; i < GRID_F_TOTAL; i++) {
        d_feats[i] = 0;
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        atomic_add_fixed(&grads[OFFSET_B1 + h], d_pre1[h]);
    }

    for (uint i = 0; i < GRID_F_TOTAL; i++) {
        for (uint h = 0; h < K_HIDDEN; h++) {
            atomic_add_fixed(&grads[OFFSET_W1 + i * K_HIDDEN + h], d_pre1[h] * features[i]);
            d_feats[i] += params[OFFSET_W1 + i * K_HIDDEN + h] * d_pre1[h];
        }
    }

    // bilinear
    for (uint i = 0; i < GRID_F1; i++) {
        for (uint corner = 0; corner < 4; corner++) {
            atomic_add_fixed(&grads[c1[corner] + i], w1[corner] * d_feats[i]);
        }
    }
    
    for (uint i = 0; i < GRID_F2; i++) {
        for (uint corner = 0; corner < 4; corner++) {
            atomic_add_fixed(&grads[OFFSET_G2 + c2[corner] + i], w2[corner] * d_feats[GRID_F1 + i]);
        }
    }
}
