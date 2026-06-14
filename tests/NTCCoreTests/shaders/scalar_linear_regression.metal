#include <metal_stdlib>
using namespace metal;

constant uint kInDim = 2;
constant uint kOutDim = 3;

struct LinRegIO {
    float W[kInDim*kOutDim];
    float b[kOutDim];
    float x[kInDim];
    float y[kOutDim];
    float loss;
    float y_pred[kOutDim];
    float dW[kInDim*kOutDim];
    float db[kOutDim];
};

kernel void scalar_linear_regression(device LinRegIO& io [[buffer(0)]],
                                     uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= kOutDim)
        return;
    
    float acc = io.b[tid];
    for (uint i = 0; i < kInDim; i++) {
        acc += io.W[i * kOutDim + tid] * io.x[i];
    }
    io.y_pred[tid] = acc;
    float diff = acc - io.y[tid];
    
    threadgroup float partial[kOutDim];
    partial[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint k = 0; k < kOutDim; ++k) {
            sum += partial[k];
        }
        io.loss = sum / float(kOutDim);
    }
    
    float dy = 2.0f * diff / float(kOutDim);
    io.db[tid] = dy;
    for (uint i = 0; i < kInDim; i++) {
        io.dW[i * kOutDim + tid] = dy * io.x[i];
    }
}
