#pragma once
#include <stdint.h>

// .ntc file layout (little-endian, no padding beyond what fields imply)
//
//   NTCHeader                                      (64 bytes)
//   uint32_t pyramidSizes    [header.kGrids]
//   uint32_t neuralMipsForLod[header.mipCount]
//   NTCSlot  slots           [header.nSlots]       (32 bytes each)
//   uint8_t  grid            nInts = sum(pyramidSizes[i]^2) * fPerGrid quantized
//                            ints, one per grid feature. Each is offset-binary
//                            in [0, 2^bits):
//                              stored = clamp(round(f/q) + offset, 0, 2^bits-1)
//                              decode = (stored - offset) * q     offset = 1<<(bits-1)
//                            q = quantScale, bits = quantBits.
//                            bits is always 4, so the ints are packed two per
//                            byte (even index low, odd high) and on disk this
//                            is ceil(nInts/2) bytes.
//   uint16_t mlp             [mlpFloatCount]       raw IEEE 754 binary16 bits
//                                                  (only when header.mlpDType == NTC_MLP_DTYPE_FP16)
//
// mlpFloatCount is derived from the header architecture fields:
//   F_IN  = roundUp(2 * fPerGrid + 4 * peWaves + 1, 16)   // padded input dim
//   count = F_IN * kHidden + kHidden
//         + kHidden * kHidden + kHidden
//         + kHidden * kOutMax + kOutMax

// file signature: the first 4 bytes of every .ntc file. Little-endian bytes
// spell "NTC1" in ASCII. Verify this before touching anything else in the file
static const uint32_t NTC_SIGNATURE       = 0x3143544EU;   // 'N','T','C','1' little-endian
static const uint32_t NTC_VERSION         = 1U;
static const int      NTC_SEMANTIC_LEN    = 16;
static const int      NTC_SWIZZLE_LEN     = 8;

static const uint32_t NTC_MLP_DTYPE_FP32  = 0U;
static const uint32_t NTC_MLP_DTYPE_FP16  = 1U;
static const uint32_t NTC_MLP_DTYPE_INT8  = 2U;   // reserved

typedef struct NTCHeader {
    uint32_t signature;      // NTC_SIGNATURE ("NTC1")
    uint32_t version;        // NTC_VERSION
    uint32_t srcW;
    uint32_t srcH;
    uint32_t mipCount;       // log2(srcW) + 1
    uint32_t kGrids;
    uint32_t fPerGrid;
    uint32_t kHidden;
    uint32_t kOutMax;        // channel slots reserved in MLP output
    uint32_t kOut;           // sum of slot channels actually used
    uint32_t nSlots;
    uint32_t peWaves;
    float    quantScale;     // q; 0 means grid is stored raw (unused for now)
    uint32_t quantBits;      // always 4; readNTC rejects anything else
    uint32_t mlpDType;       // NTC_MLP_DTYPE_*
    uint32_t flags;          // reserved, zero
} NTCHeader;

typedef struct NTCSlot {
    char     semantic     [16];  // null-padded ASCII, e.g. "Albedo"
    char     swizzle      [8];   // null-padded ASCII, e.g. "RGB"
    uint8_t  channels;           // 1 or 3
    uint8_t  channelOffset;      // base index into kOut
    uint8_t  sliceIndex;         // source-pyramid slice
    uint8_t  isSRGB;             // 0 or 1
    uint32_t reserved;           // zero
} NTCSlot;

_Static_assert(sizeof(NTCHeader) == 64, "NTCHeader must be 64 bytes");
_Static_assert(sizeof(NTCSlot)   == 32, "NTCSlot must be 32 bytes");
