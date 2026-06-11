#include <metal_stdlib>
using namespace metal;

struct BatchParams {
    float w;
    float b;
    uint  n;
    uint  _pad;
};


constant int SCALE = 1 << 14;

kernel void scalar_backprop_batch(device atomic_int*    accum  [[buffer(0)]],   // [dw_sum, db_sum, loss_sum]
                                  device const float*   x      [[buffer(1)]],
                                  device const float*   y      [[buffer(2)]],
                                  constant BatchParams& params [[buffer(3)]],
                                  uint                  tid    [[thread_position_in_grid]]) {
    if (tid >= params.n) return;

    float xi    = x[tid];
    float yi    = y[tid];
    float pred  = params.w * xi + params.b;
    float diff  = pred - yi;
    float loss  = diff * diff;
    float dpred = 2.0f * diff;
    float dw    = dpred * xi;
    float db    = dpred;

    int dw_q   = int(rint(dw   * float(SCALE)));
    int db_q   = int(rint(db   * float(SCALE)));
    int loss_q = int(rint(loss * float(SCALE)));

    atomic_fetch_add_explicit(&accum[0], dw_q,   memory_order_relaxed);
    atomic_fetch_add_explicit(&accum[1], db_q,   memory_order_relaxed);
    atomic_fetch_add_explicit(&accum[2], loss_q, memory_order_relaxed);
}
