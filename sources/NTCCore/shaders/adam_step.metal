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
    if (gid >= stepConsts.total) return;

    device atomic_int* grad_slot = (device atomic_int*)params + gid + stepConsts.total;
    int   grad_fixed = atomic_exchange_explicit(grad_slot, 0, memory_order_relaxed);
    float grad       = float(grad_fixed) / float(SCALE);

    float m_new = BETA1 * m[gid] + (1.0f - BETA1) * grad;
    float v_new = BETA2 * v[gid] + (1.0f - BETA2) * grad * grad;
    m[gid] = m_new;
    v[gid] = v_new;

    float m_hat = m_new / adamConsts.bc1;
    float v_hat = v_new / adamConsts.bc2;
    params[gid] -= adamConsts.lr * m_hat / (sqrt(v_hat) + EPS);
}
