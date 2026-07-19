#include <metal_stdlib>
#include "common.h"
using namespace metal;

#define K_HIDDEN 64

#define OUT_W 4096
#define OUT_H 4096

#define F_TOTAL   (2 * F_PER_GRID)
#define F_IN      (F_TOTAL + PE_DIM + 1)
#define POS_SCALE (float(OUT_W) / 8.0f)

kernel void grid_mlp_infer(device   const       float*          params  [[buffer(0)]],
                           device               float*          output  [[buffer(1)]],
                                    constant    StepConstants&  consts  [[buffer(2)]],
                                                uint2           gid     [[thread_position_in_grid]]) {
    uint lod = consts.inferLod;
    uint outWL = max(uint(OUT_W) >> lod, 1u);
    uint outHL = max(uint(OUT_H) >> lod, 1u);
    if (gid.x >= outWL || gid.y >= outHL) return;

    int x = int(gid.x);
    int y = int(gid.y);
    uint out_base = (uint(y) * outWL + uint(x)) * consts.kOut;

    float denomX = float(max(int(outWL) - 1, 1));
    float denomY = float(max(int(outHL) - 1, 1));

    uint neural_mip = consts.neuralMipForLod[lod];
    uint g0_offset  = consts.pyramidOffsets[neural_mip];
    uint g1_offset  = consts.pyramidOffsets[neural_mip + 1];
    uint g0_size    = consts.pyramidSizes[neural_mip];
    uint g1_size    = consts.pyramidSizes[neural_mip + 1];

    // G0
    float ix0 = float(x) * float(g0_size - 1) / denomX;
    float iy0 = float(y) * float(g0_size - 1) / denomY;
    int ix0_0 = min(int(floor(ix0)), int(g0_size) - 2);
    int iy0_0 = min(int(floor(iy0)), int(g0_size) - 2);
    float fx0 = ix0 - float(ix0_0);
    float fy0 = iy0 - float(iy0_0);
    
    // G1
    float ix1 = float(x) * float(g1_size - 1) / denomX;
    float iy1 = float(y) * float(g1_size - 1) / denomY;
    int ix0_1 = min(int(floor(ix1)), int(g1_size) - 2);
    int iy0_1 = min(int(floor(iy1)), int(g1_size) - 2);
    float fx1 = ix1 - float(ix0_1);
    float fy1 = iy1 - float(iy0_1);

    float w0[4];
    float w1[4];
    uint  c0[4];
    uint  c1[4];
    float features[F_IN];

    bilinear_sample(params + g0_offset, g0_size, F_PER_GRID, ix0_0, iy0_0, fx0, fy0, w0, c0, features               , 0.0, gid.x);
    bilinear_sample(params + g1_offset, g1_size, F_PER_GRID, ix0_1, iy0_1, fx1, fy1, w1, c1, features + F_PER_GRID  , 0.0, gid.x);
    
    float2 uv   = float2(float(x) / denomX, float(y) / denomY);
    float2 posf = uv * POS_SCALE;
    pe_encode(posf, features + F_TOTAL);

    features[F_TOTAL + PE_DIM] = float(lod) / float(MAX_LODS - 1);
    // ========
    // Forward
    // ========
    float pre1[K_HIDDEN];
    float pre2[K_HIDDEN];
    float hid1[K_HIDDEN];
    float hid2[K_HIDDEN];
    float pred[K_OUT_MAX];
    mlp_forward(params,
                consts.offsetW1, consts.offsetB1,
                consts.offsetW2, consts.offsetB2,
                consts.offsetW3, consts.offsetB3,
                F_IN, K_HIDDEN, K_OUT_MAX,
                features,
                pre1, hid1, pre2, hid2, pred);

    for (uint k = 0; k < consts.kOut; k++) {
        output[out_base + k] = pred[k];
    }
}
