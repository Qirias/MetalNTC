#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define GRID_H  256
#define GRID_W  256
#define GRID_F  8
#define GRID_TOTAL  (GRID_H * GRID_W * GRID_F)

#define K_HIDDEN 32
#define K_OUT    3

#define OUT_W 4096
#define OUT_H 4096

#define OFFSET_W1   (GRID_TOTAL)
#define OFFSET_B1   (OFFSET_W1 + GRID_F * K_HIDDEN)
#define OFFSET_W2   (OFFSET_B1 + K_HIDDEN)
#define OFFSET_B2   (OFFSET_W2 + K_HIDDEN * K_HIDDEN)
#define OFFSET_W3   (OFFSET_B2 + K_HIDDEN)
#define OFFSET_B3   (OFFSET_W3 + K_HIDDEN * K_OUT)

kernel void grid_mlp_infer(device const float* params  [[buffer(0)]],
                           device       float* output  [[buffer(1)]],
                           uint2        gid            [[thread_position_in_grid]]) {
    if (gid.x >= OUT_W || gid.y >= OUT_H) return;

    int x = int(gid.x);
    int y = int(gid.y);
    uint out_base = (uint(y) * OUT_W + uint(x)) * K_OUT;

    float ix = float(x) * float(GRID_W - 1) / float(OUT_W - 1);
    float iy = float(y) * float(GRID_H - 1) / float(OUT_H - 1);
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
    float pre1[K_HIDDEN];
    float pre2[K_HIDDEN];
    float hid1[K_HIDDEN];
    float hid2[K_HIDDEN];
    for (uint h = 0; h < K_HIDDEN; h++) {
        float acc = params[OFFSET_B1 + h];
        for (uint i = 0; i < GRID_F; i++) {
            acc += params[OFFSET_W1 + i * K_HIDDEN + h] * features[i];
        }
        pre1[h] = acc;
        hid1[h] = hard_gelu(pre1[h]);
    }
    
    // Linear2
    for (uint h = 0; h < K_HIDDEN; h++) {
        float acc = params[OFFSET_B2 + h];
        for (uint i = 0; i < K_HIDDEN; i++) {
            acc += params[OFFSET_W2 + i * K_HIDDEN + h] * hid1[i];
        }
        pre2[h] = acc;
        hid2[h] = hard_gelu(pre2[h]);
    }

    // Linear3
    for (uint k = 0; k < K_OUT; k++) {
        float acc = params[OFFSET_B3 + k];
        for (uint h = 0; h < K_HIDDEN; h++) {
            acc += params[OFFSET_W3 + h * K_OUT + k] * hid2[h];
        }
        output[out_base + k] = acc;
    }
}
