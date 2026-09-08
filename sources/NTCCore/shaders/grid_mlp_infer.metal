// MIT License
//
// Copyright (c) 2026 Kyriakos Gavras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
