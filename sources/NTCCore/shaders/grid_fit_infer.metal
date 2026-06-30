#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define GRID_H  64
#define GRID_W  64
#define GRID_CH 3

#define OUT_H 4096
#define OUT_W 4096

kernel void grid_fit_infer(device const float* params  [[buffer(0)]],
                           device       float* output  [[buffer(1)]],
                           uint2        gid            [[thread_position_in_grid]]) {
    if (gid.x >= OUT_W || gid.y >= OUT_H) return;
        
    int x = gid.x;
    int y = gid.y;
    uint out_base = (y * OUT_W + x) * GRID_CH;
    
    float iy = float(y) * float(GRID_H - 1) / float(OUT_H - 1);
    float ix = float(x) * float(GRID_W - 1) / float(OUT_W - 1);
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
    
    for (uint ch = 0; ch < GRID_CH; ch++) {
        output[out_base + ch] = w00*params[c00 + ch] + w01*params[c01 + ch] + w10*params[c10 + ch] + w11*params[c11 + ch];
    }
}
