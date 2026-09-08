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

#include <metal_stdlib>
#include "common.h"
using namespace metal;

constant float BETA1 = 0.9f;
constant float BETA2 = 0.999f;
constant float EPS   = 1e-8f;

struct AdamConstants {
    float lrGrid;
    float lrMlp;
    float bc1;
    float bc2;
};

kernel void adam_step(device            float*          params      [[buffer(0)]],
                      device            float*          m           [[buffer(1)]],
                      device            float*          v           [[buffer(2)]],
                             constant   AdamConstants&  adamConsts  [[buffer(3)]],
                             constant   StepConstants&  stepConsts  [[buffer(4)]],
                                        uint            gid         [[thread_position_in_grid]]) {

    uint slot = gid + stepConsts.adamOffset;
    if (slot >= stepConsts.total) return;

    device atomic_int* grad_slot = (device atomic_int*)params + slot + stepConsts.total;
    int   grad_fixed = atomic_exchange_explicit(grad_slot, 0, memory_order_relaxed);
    float grad       = float(grad_fixed) / float(SCALE);

    float m_new = BETA1 * m[slot] + (1.0f - BETA1) * grad;
    float v_new = BETA2 * v[slot] + (1.0f - BETA2) * grad * grad;
    m[slot] = m_new;
    v[slot] = v_new;

    float m_hat = m_new / adamConsts.bc1;
    float v_hat = v_new / adamConsts.bc2;
    float lr = slot < stepConsts.pyramidOffsets[K_GRIDS] ? adamConsts.lrGrid : adamConsts.lrMlp;
    params[slot] -= lr * m_hat / (sqrt(v_hat) + EPS);

    if (stepConsts.bits != 0u && slot < stepConsts.pyramidOffsets[K_GRIDS]) {
        params[slot] = clamp(params[slot], stepConsts.lo, stepConsts.hi);
    }
}
