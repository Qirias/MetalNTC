#include <metal_stdlib>
#include "common.h"
#include "shapes.h"
using namespace metal;

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

struct SampleOffset {
    static constant int x   = 0;
    static constant int y   = x + K_IN;
    static constant int out = y + K_OUT;
    static constant int loss = out + K_OUT;
    static constant int stride = loss + 1;
};

kernel void sin_mlp_train(device float* params  [[buffer(0)]],
                          device float* samples [[buffer(1)]],
                                 uint   gid     [[thread_position_in_grid]]) {
    
    if (gid >= K_BATCH) return;

    float pre[K_HIDDEN];   // preactivations of hidden layer
    float h[K_HIDDEN];     // hidden activations
    float out[K_OUT];      // predictions
    
    device atomic_int* grads = reinterpret_cast<device atomic_int*>(params);
    uint s_base = gid * SampleOffset::stride;
    
    // forward
    // linear1 + activation
    for (uint j = 0; j < K_HIDDEN; j++) {
        float acc = params[ParamsOffset::b1 + j];
        for (uint i = 0; i < K_IN; i++) {
            acc += params[ParamsOffset::w1 + i * K_HIDDEN + j] * samples[s_base + SampleOffset::x + i];
        }
        pre[j] = acc;
        h[j] = hard_gelu(pre[j]);
    }

    // linear2
    for (uint k = 0; k < K_OUT; k++) {
        float acc = params[ParamsOffset::b2 + k];
        for (uint j = 0; j < K_HIDDEN; j++) {
            acc += params[ParamsOffset::w2 + j * K_OUT + k] * h[j];
        }
        out[k] = acc;
        samples[s_base + SampleOffset::out + k] = acc;
    }

    // MSE loss
    float loss = 0;
    for (uint k = 0; k < K_OUT; k++) {
        float diff = out[k] - samples[s_base + SampleOffset::y + k];
        loss += diff * diff;
    }
    samples[s_base + SampleOffset::loss] = loss / float(K_OUT);


    // backward
    // first gradient from the loss and everything below is chain ruling to this one
    float d_out[K_OUT];
    for (uint k = 0; k < K_OUT; k++) {
        d_out[k] = 2.0 * (out[k] - samples[s_base + SampleOffset::y + k]) / float(K_OUT) / float(K_BATCH);
    }

    // linear2 backward
    // parameter gradients
    for (uint k = 0; k < K_OUT; k++) {
        atomic_add_fixed(&grads[ParamsOffset::db2 + k], d_out[k]);
    }
    for (uint j = 0; j < K_HIDDEN; j++) {
        for (uint k = 0; k < K_OUT; k++) {
            atomic_add_fixed(&grads[ParamsOffset::dW2 + j * K_OUT + k], d_out[k] * h[j]);
        }
    }
    
    // out used h to be built and out is used for the loss
    // out[k] = b2[k] + W2[0,k]*h[0] + W2[1,k]*h[1] + ... + W2[j,k]*h[j] + ...
    // differentiate that with respect to h[j] and everything that doesn't contain h[j] vanishes
    // the only thing surviving is the term W2[j,k]*h[j]
    // its derivative w.r.t h[j] is W2[j,k] so d_out[k]/d_h[j] = W2[j,k]
    float d_h[K_HIDDEN];
    for (uint j = 0; j < K_HIDDEN; j++) {
        float acc = 0;
        // h[j] has K_OUT separate ways of affecting the loss (see linear2 of forward)
        for (uint k = 0; k < K_OUT; k++) {
            acc += params[ParamsOffset::w2 + j * K_OUT + k] * d_out[k];
        }
        d_h[j] = acc;
    }

    float d_pre[K_HIDDEN];
    for (uint j = 0; j < K_HIDDEN; j++) {
        d_pre[j] = d_h[j] * hard_gelu_prime(pre[j]);
    }

    // linear1 backward
    for (uint j = 0; j < K_HIDDEN; j++) {
        atomic_add_fixed(&grads[ParamsOffset::db1 + j], d_pre[j]);
    }
    for (uint i = 0; i < K_IN; i++) {
        float xi = samples[s_base + SampleOffset::x + i];
        for (uint j = 0; j < K_HIDDEN; j++) {
            atomic_add_fixed(&grads[ParamsOffset::dW1 + i * K_HIDDEN + j], d_pre[j] * xi);
        }
    }
}
