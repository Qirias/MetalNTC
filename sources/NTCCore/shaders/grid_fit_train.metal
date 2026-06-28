#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define GRID_H  64
#define GRID_W  64
#define GRID_CH 3
#define GRID_TOTAL  (GRID_H * GRID_W * GRID_CH)

#define SRC_H 4096
#define SRC_W 4096

#define K_BATCH 1024

#define SAMPLE_Y      0
#define SAMPLE_X      1
#define SAMPLE_LOSS   2
#define SAMPLE_STRIDE 3

kernel void grid_fit_train(device       float* params  [[buffer(0)]],
                           device       float* samples [[buffer(1)]],
                           device const float* source  [[buffer(2)]],
                                        uint   gid     [[thread_position_in_grid]]) {
    if (gid >= K_BATCH) return;
    
    device atomic_int* grads = reinterpret_cast<device atomic_int*>(params);
    
    uint sample_base = gid * SAMPLE_STRIDE;
    int x = int(samples[sample_base + SAMPLE_X]);
    int y = int(samples[sample_base + SAMPLE_Y]);

    float gt[GRID_CH];
    uint src_base = (uint(y) * SRC_W + uint(x)) * GRID_CH;
    
    for (uint ch = 0; ch < GRID_CH; ch++) {
        gt[ch] = source[src_base + ch];
    }

    float iy = float(y) * float(GRID_H - 1) / float(SRC_H - 1);
    float ix = float(x) * float(GRID_W - 1) / float(SRC_W - 1);
    int   iy0 = int(floor(iy));
    int   ix0 = int(floor(ix));
    iy0 = min(iy0, GRID_H - 2);
    ix0 = min(ix0, GRID_W - 2);
    float fy = iy - float(iy0);
    float fx = ix - float(ix0);
    
    float w00 = (1 - fx) * (1 - fy);
    float w01 =      fx  * (1 - fy);
    float w10 = (1 - fx) *      fy;
    float w11 =      fx  *      fy;
    
    uint c00 = (uint(iy0)     * GRID_W + uint(ix0))     * GRID_CH;
    uint c01 = (uint(iy0)     * GRID_W + uint(ix0 + 1)) * GRID_CH;
    uint c10 = (uint(iy0 + 1) * GRID_W + uint(ix0))     * GRID_CH;
    uint c11 = (uint(iy0 + 1) * GRID_W + uint(ix0 + 1)) * GRID_CH;
    
    float pred[GRID_CH];
    for (uint ch = 0; ch < GRID_CH; ch++) {
        pred[ch] = w00*params[c00 + ch] + w01*params[c01 + ch] + w10*params[c10 + ch] + w11*params[c11 + ch];
    }
    
    float diff[GRID_CH];
    float loss = 0;
    for (uint ch = 0; ch < GRID_CH; ch++) {
        diff[ch] = pred[ch] - gt[ch];
        loss += diff[ch] * diff[ch];
    }
    samples[sample_base + SAMPLE_LOSS] = loss / float(GRID_CH);
    
    float d_pred[GRID_CH];
    for (uint ch = 0; ch < GRID_CH; ch++) {
        d_pred[ch] = 2.0f * diff[ch] / float(GRID_CH);
    }

    for (uint ch = 0; ch < GRID_CH; ch++) {
        atomic_add_fixed(&grads[GRID_TOTAL + c00 + ch], d_pred[ch] * w00);
        atomic_add_fixed(&grads[GRID_TOTAL + c01 + ch], d_pred[ch] * w01);
        atomic_add_fixed(&grads[GRID_TOTAL + c10 + ch], d_pred[ch] * w10);
        atomic_add_fixed(&grads[GRID_TOTAL + c11 + ch], d_pred[ch] * w11);
    }
}
