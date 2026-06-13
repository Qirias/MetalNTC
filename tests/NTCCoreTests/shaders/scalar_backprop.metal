#include <metal_stdlib>
using namespace metal;

struct ScalarIO {
    float w; // weight
    float b; // bias
    float x; // input
    float y; // target
    float loss;
    float dw;
    float db;
};

kernel void scalar_backprop(device ScalarIO& io [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    float pred = io.w * io.x + io.b; // forward pred
    float diff = pred - io.y; // error
    io.loss = diff * diff;
    float dpred = 2.0f * diff; // derivative w.r.t the prediction
    io.dw = dpred * io.x; // derivative w.r.t the weight
    io.db = dpred; // derivative w.r.t the bias
}
