#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define GRID_H  256
#define GRID_W  256
#define GRID_F  8
#define GRID_TOTAL  (GRID_H * GRID_W * GRID_F)

#define K_HIDDEN 32
#define K_OUT    3
#define K_BATCH  1024

#define SRC_W 4096
#define SRC_H 4096

#define OFFSET_W1   (GRID_TOTAL)
#define OFFSET_B1   (OFFSET_W1 + GRID_F * K_HIDDEN)
#define OFFSET_W2   (OFFSET_B1 + K_HIDDEN)
#define OFFSET_B2   (OFFSET_W2 + K_HIDDEN * K_OUT)
#define TOTAL       (OFFSET_B2 + K_OUT)

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

    float ix = float(x) * float(GRID_W - 1) / float(SRC_W - 1);
    float iy = float(y) * float(GRID_H - 1) / float(SRC_H - 1);
    int iy0 = int(floor(iy));
    int ix0 = int(floor(ix));
    iy0 = min(iy0, GRID_H - 2);
    ix0 = min(ix0, GRID_W - 2);
    float fy = iy - float(iy0);
    float fx = ix - float(ix0);

    float w[4];
    uint  c[4];
    float features[GRID_F];
    bilinear_sample(params, GRID_W, GRID_F, ix0, iy0, fx, fy, w, c, features);
    
    // ========
    // Forward
    // ========
    
    // Linear1 + hardGELU
    float pre[K_HIDDEN];
    float hid[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        float acc = params[OFFSET_B1 + h];
        for (uint i = 0; i < GRID_F; i++) {
            acc += params[OFFSET_W1 + i * K_HIDDEN + h] * features[i];
        }
        pre[h] = acc;
        hid[h] = hard_gelu(pre[h]);
    }

    // Linear2
    float pred[K_OUT];
    for (uint k = 0; k < K_OUT; k++) {
        float acc = params[OFFSET_B2 + k];
        for (uint h = 0; h < K_HIDDEN; h++) {
            acc += params[OFFSET_W2 + h * K_OUT + k] * hid[h];
        }
        pred[k] = acc;
    }

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
    // Linear2
    float d_hid[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_hid[h] = 0;
    }

    for (uint k = 0; k < K_OUT; k++) {
        atomic_add_fixed(&grads[OFFSET_B2 + k], d_pred[k]);
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        for (uint k = 0; k < K_OUT; k++) {
            atomic_add_fixed(&grads[OFFSET_W2 + h * K_OUT + k], d_pred[k] * hid[h]);
            d_hid[h] += params[OFFSET_W2 + h * K_OUT + k] * d_pred[k];
        }
    }

    // hardGELU
    float d_pre[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        d_pre[h] = d_hid[h] * hard_gelu_prime(pre[h]);
    }

    // Linear1
    float d_feats[GRID_F];
    for (uint i = 0; i < GRID_F; i++) {
        d_feats[i] = 0;
    }

    for (uint h = 0; h < K_HIDDEN; h++) {
        atomic_add_fixed(&grads[OFFSET_B1 + h], d_pre[h]);
    }

    for (uint i = 0; i < GRID_F; i++) {
        for (uint h = 0; h < K_HIDDEN; h++) {
            atomic_add_fixed(&grads[OFFSET_W1 + i * K_HIDDEN + h], d_pre[h] * features[i]);
            d_feats[i] += params[OFFSET_W1 + i * K_HIDDEN + h] * d_pre[h];
        }
    }

    // bilinear
    for (uint i = 0; i < GRID_F; i++) {
        for (uint corner = 0; corner < 4; corner++) {
            atomic_add_fixed(&grads[c[corner] + i], w[corner] * d_feats[i]);
        }
    }
}
