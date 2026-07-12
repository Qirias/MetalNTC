#include <metal_stdlib>
#include "common.h"
using namespace metal;

constant float BETA1 = 0.9f;
constant float BETA2 = 0.999f;
constant float EPS   = 1e-8f;

struct AdamConstants {
    float lr;
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
    params[slot] -= adamConsts.lr * m_hat / (sqrt(v_hat) + EPS);

    uint grid_idx = UINT_MAX;
    if (slot < stepConsts.offsetG2) {
        grid_idx = 0;
    } else if (slot < stepConsts.offsetW1) {
        grid_idx = 1;
    }
    
    // if it is a grid and quant is enabled
    if (grid_idx != UINT_MAX && stepConsts.bitsPerGrid[grid_idx] != 0u) {
        params[slot] = clamp(params[slot], stepConsts.loPerGrid[grid_idx], stepConsts.hiPerGrid[grid_idx]);
    }
}
