#include <metal_stdlib>
#include "shapes.h"
using namespace metal;

kernel void bilinear_sample_train(device const float* grid     [[buffer(0)]],   // 192 floats
                                  device float*       gridGrad [[buffer(1)]],   // 192 floats
                                  device float*       sample   [[buffer(2)]],   // stride 9: iy, ix, y[3], out[3], loss
                                  uint                gid      [[thread_position_in_grid]]) {
    if (gid != 0) return;
    
    float ix = sample[0];
    float iy = sample[1];
    
    float iy0 = floor(iy);
    float ix0 = floor(ix);
    float iy1 = iy0 + 1;
    float ix1 = ix0 + 1;
    
    float fract_y = iy - iy0;
    float fract_x = ix - ix0;
    
    float w00 = (1 - fract_x)   * (1 - fract_y);
    float w01 = fract_x         * (1 - fract_y);
    float w10 = (1 - fract_x)   * fract_y;
    float w11 = fract_x         * fract_y;
    
    uint base00 = (uint(iy0) * GRID_W + uint(ix0)) * GRID_CH;
    uint base01 = (uint(iy0) * GRID_W + uint(ix1)) * GRID_CH;
    uint base10 = (uint(iy1) * GRID_W + uint(ix0)) * GRID_CH;
    uint base11 = (uint(iy1) * GRID_W + uint(ix1)) * GRID_CH;
    
    float d_out[GRID_CH];
    float loss = 0;
    
    for (uint ch = 0; ch < GRID_CH; ch++) {
        float out = w00 * grid[base00 + ch] + w01 * grid[base01 + ch] +
                    w10 * grid[base10 + ch] + w11 * grid[base11 + ch];
        float y = sample[2 + ch];
        float diff = out - y;
        sample[5 + ch] = out;
        d_out[ch] = 2.0 * diff;
        loss += diff * diff;
    }
    sample[8] = loss;
    
    for (uint ch = 0; ch < GRID_CH; ch++) {
        gridGrad[base00 + ch] = w00 * d_out[ch];
        gridGrad[base01 + ch] = w01 * d_out[ch];
        gridGrad[base10 + ch] = w10 * d_out[ch];
        gridGrad[base11 + ch] = w11 * d_out[ch];
    }
}
