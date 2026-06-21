#include <metal_stdlib>
#include "common.h"
using namespace metal;

constant uint K_IN     = 1;
constant uint K_HIDDEN = 4;
constant uint K_OUT    = 2;
constant uint K_BATCH  = 64;

struct ParamsOffset {
    static constant int w1 = 0;
    static constant int b1 = w1 + K_IN * K_HIDDEN;
    static constant int w2 = b1 + K_HIDDEN;
    static constant int b2 = w2 + K_HIDDEN * K_OUT;
    static constant int floatTotal = b2 + K_OUT;
    
    static constant int dW1 = floatTotal;
    static constant int db1 = dW1 + K_IN * K_HIDDEN;
    static constant int dW2 = db1 + K_HIDDEN;
    static constant int db2 = dW2 + K_HIDDEN * K_OUT;
    static constant int total = db2 + K_OUT;
};

kernel void sgd_step(device float* params  [[buffer(0)]],
                     device float* lr      [[buffer(1)]],
                            uint   gid     [[thread_position_in_grid]]) {
    if (gid >= ParamsOffset::floatTotal) return;
    
    device atomic_int* grad_slot = (device atomic_int*)params + gid + ParamsOffset::floatTotal;
    // get grad value and zero the grad
    int grad_fixed = atomic_exchange_explicit(grad_slot, 0, memory_order_relaxed);
    float grad = float(grad_fixed) / float(SCALE);
    params[gid] -= lr[0] * grad;
}
