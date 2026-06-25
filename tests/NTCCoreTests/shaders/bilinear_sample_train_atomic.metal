#include <metal_stdlib>
#include "common.h"
#include "shapes.h"
using namespace metal;

struct SampleOffset {
    static constant int ix     = 0;
    static constant int iy     = 1;
    static constant int y      = 2; // GRID_CH floats
    static constant int out    = y + GRID_CH;   // 5
    static constant int loss   = out + GRID_CH; // 8
    static constant int stride = loss + 1;      // 9
};

constant uint GRID_TOTAL = GRID_H * GRID_W * GRID_CH;

kernel void bilinear_sample_train_atomic(
    device float* grid    [[buffer(0)]],   // 2 * GRID_TOTAL, [grid, grad]
    device float* samples [[buffer(1)]],   // K_QUERIES * SampleOffset::stride
    uint          gid     [[thread_position_in_grid]])
{
    if (gid >= K_QUERIES) return;

    device atomic_int* gradInt = reinterpret_cast<device atomic_int*>(grid) + GRID_TOTAL;

    uint s_base = gid * SampleOffset::stride;
    float ix = samples[s_base + SampleOffset::ix];
    float iy = samples[s_base + SampleOffset::iy];
    
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
        float y = samples[s_base + SampleOffset::y + ch];
        float diff = out - y;
        samples[s_base + SampleOffset::out + ch] = out;
        d_out[ch] = 2.0 * diff;
        loss += diff * diff;
    }
    samples[s_base + SampleOffset::loss] = loss;
    
    for (uint ch = 0; ch < GRID_CH; ch++) {
        grid[base00 + ch] = w00 * d_out[ch];
        grid[base01 + ch] = w01 * d_out[ch];
        grid[base10 + ch] = w10 * d_out[ch];
        grid[base11 + ch] = w11 * d_out[ch];
        
        atomic_add_fixed(&gradInt[base00 + ch], w00 * d_out[ch]);
        atomic_add_fixed(&gradInt[base01 + ch], w01 * d_out[ch]);
        atomic_add_fixed(&gradInt[base10 + ch], w10 * d_out[ch]);
        atomic_add_fixed(&gradInt[base11 + ch], w11 * d_out[ch]);
    }
}
