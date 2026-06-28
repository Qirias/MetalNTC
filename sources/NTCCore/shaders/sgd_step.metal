#include <metal_stdlib>
#include "common.h"
using namespace metal;

kernel void sgd_step(device         float* params  [[buffer(0)]],
                     device         float* lr      [[buffer(1)]],
                     device const   uint*  nFloats [[buffer(2)]],
                            uint   gid     [[thread_position_in_grid]]) {
    if (gid >= nFloats[0]) return;
    
    device atomic_int* grad_slot = (device atomic_int*)params + gid + nFloats[0];
    // get grad value and zero the grad
    int grad_fixed = atomic_exchange_explicit(grad_slot, 0, memory_order_relaxed);
    float grad = float(grad_fixed) / float(SCALE);
    params[gid] -= lr[0] * grad;
}
