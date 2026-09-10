// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

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
