// MIT License
//
// Copyright (c) 2026 Kyriakos Gavras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import Foundation

// edit these together with common.h, never alone
struct AdamConstants {
    var lrGrid: Float
    var lrMlp:  Float
    var bc1:    Float
    var bc2:    Float
}

struct StepConstants {
    var kBatch:             UInt32
    var pyramidOffsets:     (UInt32, UInt32, UInt32, UInt32,
                             UInt32, UInt32, UInt32, UInt32,
                             UInt32)
    var pyramidSizes:       (UInt32, UInt32, UInt32, UInt32,
                             UInt32, UInt32, UInt32, UInt32)
    var offsetW1:           UInt32
    var offsetB1:           UInt32
    var offsetW2:           UInt32
    var offsetB2:           UInt32
    var offsetW3:           UInt32
    var offsetB3:           UInt32
    var total:              UInt32
    var bits:               UInt32
    var q:                  Float
    var lo:                 Float
    var hi:                 Float
    var adamOffset:         UInt32
    var inferLod:           UInt32
    // max mips 13 for 4k textures
    var neuralMipsForLod:   (UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32, UInt32, UInt32,
                            UInt32)
    var kOut:                UInt32
    var nSlices:             UInt32
    // K_OUT_MAX slice channels and offsets
    var sliceChannels:       (UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32)
    var sliceChannelOffsets: (UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32,
                              UInt32, UInt32, UInt32, UInt32)
    var srcW:                UInt32
    var srcH:                UInt32
    var mipCount:            UInt32
    var posScale:            Float
}
