#include <metal_stdlib>
using namespace metal;
#include "common.h"

struct ScalarIO {
    float w;    // weight
    float b;    // bias
    float x;    // input
    float y;    // target
    float loss;
    float dw;
    float db;
};

// Forward pass:
//   pred = w*x + b
//   out  = hard_gelu(pred)
//   loss = (out - y)^2

// https://en.wikipedia.org/wiki/Chain_rule
// Backward pass:
//   d(loss)/d(out)  = 2 * (out - y)
//   d(out)/d(pred)  = hard_gelu_prime(pred)
//   d(loss)/d(pred) = d(loss)/d(out) * d(out)/d(pred)
//   d(loss)/d(w)    = d(loss)/d(pred) * x
//   d(loss)/d(b)    = d(loss)/d(pred)
kernel void scalar_backprop_hardgelu(device ScalarIO&   io  [[buffer(0)]],
                                            uint        tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    float pred = io.w * io.x + io.b;
    float out  = hard_gelu(pred);
    float diff = out - io.y;
    io.loss = diff * diff;

    float dout  = 2.0f * diff;
    float dpred = dout * hard_gelu_prime(pred);
    io.dw = dpred * io.x;
    io.db = dpred;
}
