constant uint GRID_H  = 8;
constant uint GRID_W  = 8;
constant uint GRID_CH = 3;

kernel void bilinear_sample_forward(
    device const float* grid   [[buffer(0)]],
    device const float* query  [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]])
{
    if (gid != 0) return;
    
    float ix = query[0];
    float iy = query[1];
    
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
    
    for (uint ch = 0; ch < GRID_CH; ch++) {
        out[ch] = w00*grid[base00 + ch] + w01*grid[base01 + ch] + w10*grid[base10 + ch] + w11*grid[base11 + ch];
    }
}
