#include <metal_stdlib>
#include "common.h"
#include "shapes.h"
using namespace metal;

struct Offset {
    static constant int w1    = 0;
    static constant int b1    = w1   + K_IN * K_HIDDEN;
    static constant int w2    = b1   + K_HIDDEN;
    static constant int b2    = w2   + K_HIDDEN * K_OUT;
    static constant int x     = b2   + K_OUT;
    static constant int y     = x    + K_IN;
    static constant int out   = y    + K_OUT;
    static constant int loss  = out  + K_OUT;
    static constant int dW1   = loss + 1;
    static constant int db1   = dW1  + K_IN * K_HIDDEN;
    static constant int dW2   = db1  + K_HIDDEN;
    static constant int db2   = dW2  + K_HIDDEN * K_OUT;
    static constant int total = db2  + K_OUT;
};

 kernel void sin_mlp_forward_backward(device float* io  [[buffer(0)]],
                                             uint   tid [[thread_position_in_threadgroup]]) {

     if (tid != 0) return;

     float pre[K_HIDDEN];   // preactivations of hidden layer
     float h[K_HIDDEN];     // hidden activations
     float out[K_OUT];      // predictions

     // forward
     // linear1 + activation
     for (uint j = 0; j < K_HIDDEN; j++) {
         float acc = io[Offset::b1 + j];
         for (uint i = 0; i < K_IN; i++) {
             acc += io[Offset::w1 + i * K_HIDDEN + j] * io[Offset::x + i];
         }
         pre[j] = acc;
         h[j] = hard_gelu(pre[j]);
     }

     // linear2
     for (uint k = 0; k < K_OUT; k++) {
         float acc = io[Offset::b2 + k];
         for (uint j = 0; j < K_HIDDEN; j++) {
             acc += io[Offset::w2 + j * K_OUT + k] * h[j];
         }
         out[k] = acc;
         io[Offset::out + k] = acc;
     }

     // MSE loss
     float loss = 0;
     for (uint k = 0; k < K_OUT; k++) {
         float diff = out[k] - io[Offset::y + k];
         loss += diff * diff;
     }
     io[Offset::loss] = loss / float(K_OUT);


     // backward
     // first gradient from the loss and everything below is chain ruling to this one
     float d_out[K_OUT];
     for (uint k = 0; k < K_OUT; k++) {
         d_out[k] = 2.0 * (out[k] - io[Offset::y + k]) / float(K_OUT);
     }

     // linear2 backward
     // parameter gradients
     for (uint k = 0; k < K_OUT; k++) {
         io[Offset::db2 + k] = d_out[k];
     }
     for (uint j = 0; j < K_HIDDEN; j++) {
         for (uint k = 0; k < K_OUT; k++) {
             io[Offset::dW2 + j * K_OUT + k] = d_out[k] * h[j];
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
             acc += io[Offset::w2 + j * K_OUT + k] * d_out[k];
         }
         d_h[j] = acc;
     }

     float d_pre[K_HIDDEN];
     for (uint j = 0; j < K_HIDDEN; j++) {
         d_pre[j] = d_h[j] * hard_gelu_prime(pre[j]);
     }

     // linear1 backward
     for (uint j = 0; j < K_HIDDEN; j++) {
         io[Offset::db1 + j] = d_pre[j];
     }
     for (uint i = 0; i < K_IN; i++) {
         float xi = io[Offset::x + i];
         for (uint j = 0; j < K_HIDDEN; j++) {
             io[Offset::dW1 + i * K_HIDDEN + j] = d_pre[j] * xi;
         }
     }
 }
