#include <metal_stdlib>
using namespace metal;

struct ScalarIO {
    float w;    // weight
    float b;    // bias
    float x;    // input
    float y;    // target
    float loss;
    float dw;
    float db;
};

// a cheap piecewise approximation of GELU
inline float hard_gelu(float x) {
    return 0.5f * x * (1.0f + clamp(x * 0.5f, -1.0f, 1.0f));
}

// https://en.wikipedia.org/wiki/Product_rule
// derived from the product rule on out = 0.5 * x * (1 + c)
// where c = clamp(x/2, -1, 1):
//   d(out)/dx = 0.5 * (1 + c)       (derivative of 0.5*x, times (1+c))
//             + 0.5 * x * dc/dx     (0.5*x, times derivative of (1+c))
// dc/dx is 0.5 when x is in [-2, 2] (clamp is just x/2 there), and 0
// outside. The in_band flag picks the right case.

inline float hard_gelu_prime(float x) {
    float c = clamp(x * 0.5f, -1.0f, 1.0f);
    float in_band = (x >= -2.0f && x <= 2.0f) ? 1.0f : 0.0f;
    return 0.5f * (1.0f + c) + 0.5f * x * 0.5f * in_band;
}

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
kernel void scalar_backprop_hardgelu(device ScalarIO& io [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]]) {
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
