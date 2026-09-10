// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

/// This file only serves for testing and it is not part of the real-time inference.

#include <metal_stdlib>
#include "common.h"
using namespace metal;

kernel void grid_mlp_infer(device   const       float*          params  [[buffer(0)]],
                           device               float*          output  [[buffer(1)]],
                                    constant    StepConstants&  consts  [[buffer(2)]],
                                                uint2           gid     [[thread_position_in_grid]]) {
    uint lod   = consts.inferLod;
    uint outWL = max(consts.srcW >> lod, 1u);
    uint outHL = max(consts.srcH >> lod, 1u);
    if (gid.x >= outWL || gid.y >= outHL) return;

    float denomX = float(max(int(outWL) - 1, 1));
    float denomY = float(max(int(outHL) - 1, 1));
    float2 uv = float2(float(gid.x) / denomX, float(gid.y) / denomY);

    half pred[K_OUT_MAX];
    ntc_decode(uv, lod, params, consts, pred);

    uint out_base = (gid.y * outWL + gid.x) * consts.kOut;
    for (uint k = 0; k < consts.kOut; k++) {
        output[out_base + k] = float(pred[k]);
    }
}
