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

#define OUT_W 4096
#define OUT_H 4096

#define OFFSET_G1   0
#define OFFSET_G2   (GRID1_TOTAL)
#define OFFSET_W1   (OFFSET_G2 + GRID2_TOTAL)
#define OFFSET_B1   (OFFSET_W1 + GRID_F_TOTAL * K_HIDDEN)
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

    float ix1 = float(x) * float(GRID_W1 - 1) / float(OUT_W - 1);
    float iy1 = float(y) * float(GRID_H1 - 1) / float(OUT_H - 1);
    int iy0_1 = int(floor(iy1));
    int ix0_1 = int(floor(ix1));
    iy0_1 = min(iy0_1, GRID_H1 - 2);
    ix0_1 = min(ix0_1, GRID_W1 - 2);
    float fy1 = iy1 - float(iy0_1);
    float fx1 = ix1 - float(ix0_1);

    float ix2 = float(x) * float(GRID_W2 - 1) / float(OUT_W - 1);
    float iy2 = float(y) * float(GRID_H2 - 1) / float(OUT_H - 1);
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

    for (uint k = 0; k < K_OUT; k++) {
        output[out_base + k] = pred[k];
    }
}
