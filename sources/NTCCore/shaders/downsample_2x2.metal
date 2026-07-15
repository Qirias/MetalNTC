#include <metal_stdlib>
#include "common.h"
using namespace metal;

inline float4 reduce2x2(float4 a, float4 b, float4 c, float4 d) {
    return (a + b + c + d) * 0.25f;
}

kernel void downsample_2x2(texture2d<float, access::read>       srcTexture      [[texture(0)]],
                           texture2d<float, access::read_write> pyramid         [[texture(1)]],
                           device atomic_uint&                  globalCounter   [[buffer(0)]],
                           constant SPDConstants&               consts          [[buffer(1)]],
                           uint2                                lid             [[thread_position_in_threadgroup]],
                           uint2                                tgid            [[threadgroup_position_in_grid]],
                           uint                                 lid_flat        [[thread_index_in_threadgroup]]) {
    threadgroup float4 lds[32][32];

    uint2 srcSize   = uint2(srcTexture.get_width(), srcTexture.get_height());
    uint2 baseCoord = tgid * 64 + lid * 4;

    // load 4x4 from source, copy to mip 0, reduce to 2x2 for mip 1

    // row 0
    float4 s00 = srcTexture.read(min(baseCoord + uint2(0, 0), srcSize - 1));
    float4 s10 = srcTexture.read(min(baseCoord + uint2(1, 0), srcSize - 1));
    float4 s20 = srcTexture.read(min(baseCoord + uint2(2, 0), srcSize - 1));
    float4 s30 = srcTexture.read(min(baseCoord + uint2(3, 0), srcSize - 1));

    // row 1
    float4 s01 = srcTexture.read(min(baseCoord + uint2(0, 1), srcSize - 1));
    float4 s11 = srcTexture.read(min(baseCoord + uint2(1, 1), srcSize - 1));
    float4 s21 = srcTexture.read(min(baseCoord + uint2(2, 1), srcSize - 1));
    float4 s31 = srcTexture.read(min(baseCoord + uint2(3, 1), srcSize - 1));

    // row 2
    float4 s02 = srcTexture.read(min(baseCoord + uint2(0, 2), srcSize - 1));
    float4 s12 = srcTexture.read(min(baseCoord + uint2(1, 2), srcSize - 1));
    float4 s22 = srcTexture.read(min(baseCoord + uint2(2, 2), srcSize - 1));
    float4 s32 = srcTexture.read(min(baseCoord + uint2(3, 2), srcSize - 1));

    // row 3
    float4 s03 = srcTexture.read(min(baseCoord + uint2(0, 3), srcSize - 1));
    float4 s13 = srcTexture.read(min(baseCoord + uint2(1, 3), srcSize - 1));
    float4 s23 = srcTexture.read(min(baseCoord + uint2(2, 3), srcSize - 1));
    float4 s33 = srcTexture.read(min(baseCoord + uint2(3, 3), srcSize - 1));

    // write 4x4 block to mip 0 to get a full copy
    if (all(baseCoord + uint2(0, 0) < srcSize)) pyramid.write(s00, baseCoord + uint2(0, 0), 0);
    if (all(baseCoord + uint2(1, 0) < srcSize)) pyramid.write(s10, baseCoord + uint2(1, 0), 0);
    if (all(baseCoord + uint2(2, 0) < srcSize)) pyramid.write(s20, baseCoord + uint2(2, 0), 0);
    if (all(baseCoord + uint2(3, 0) < srcSize)) pyramid.write(s30, baseCoord + uint2(3, 0), 0);

    if (all(baseCoord + uint2(0, 1) < srcSize)) pyramid.write(s01, baseCoord + uint2(0, 1), 0);
    if (all(baseCoord + uint2(1, 1) < srcSize)) pyramid.write(s11, baseCoord + uint2(1, 1), 0);
    if (all(baseCoord + uint2(2, 1) < srcSize)) pyramid.write(s21, baseCoord + uint2(2, 1), 0);
    if (all(baseCoord + uint2(3, 1) < srcSize)) pyramid.write(s31, baseCoord + uint2(3, 1), 0);

    if (all(baseCoord + uint2(0, 2) < srcSize)) pyramid.write(s02, baseCoord + uint2(0, 2), 0);
    if (all(baseCoord + uint2(1, 2) < srcSize)) pyramid.write(s12, baseCoord + uint2(1, 2), 0);
    if (all(baseCoord + uint2(2, 2) < srcSize)) pyramid.write(s22, baseCoord + uint2(2, 2), 0);
    if (all(baseCoord + uint2(3, 2) < srcSize)) pyramid.write(s32, baseCoord + uint2(3, 2), 0);

    if (all(baseCoord + uint2(0, 3) < srcSize)) pyramid.write(s03, baseCoord + uint2(0, 3), 0);
    if (all(baseCoord + uint2(1, 3) < srcSize)) pyramid.write(s13, baseCoord + uint2(1, 3), 0);
    if (all(baseCoord + uint2(2, 3) < srcSize)) pyramid.write(s23, baseCoord + uint2(2, 3), 0);
    if (all(baseCoord + uint2(3, 3) < srcSize)) pyramid.write(s33, baseCoord + uint2(3, 3), 0);

    // reduce 4x4 to 2x2 for mip 1
    float4 r0 = reduce2x2(s00, s10, s01, s11);
    float4 r1 = reduce2x2(s20, s30, s21, s31);
    float4 r2 = reduce2x2(s02, s12, s03, s13);
    float4 r3 = reduce2x2(s22, s32, s23, s33);

    uint2 ldsBase = lid * 2;
    lds[ldsBase.y + 0][ldsBase.x + 0] = r0;
    lds[ldsBase.y + 0][ldsBase.x + 1] = r1;
    lds[ldsBase.y + 1][ldsBase.x + 0] = r2;
    lds[ldsBase.y + 1][ldsBase.x + 1] = r3;

    // mip 1: 64x64 to 32x32
    uint2 mip1Base = tgid * 32 + ldsBase;
    uint2 mip1Size = srcSize >> 1;
    if (all(mip1Base + uint2(0, 0) < mip1Size)) pyramid.write(r0, mip1Base + uint2(0, 0), 1);
    if (all(mip1Base + uint2(1, 0) < mip1Size)) pyramid.write(r1, mip1Base + uint2(1, 0), 1);
    if (all(mip1Base + uint2(0, 1) < mip1Size)) pyramid.write(r2, mip1Base + uint2(0, 1), 1);
    if (all(mip1Base + uint2(1, 1) < mip1Size)) pyramid.write(r3, mip1Base + uint2(1, 1), 1);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mip 2: 32x32 to 16x16
    float4 reduced = reduce2x2(lds[ldsBase.y + 0][ldsBase.x + 0],
                               lds[ldsBase.y + 0][ldsBase.x + 1],
                               lds[ldsBase.y + 1][ldsBase.x + 0],
                               lds[ldsBase.y + 1][ldsBase.x + 1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    lds[lid.y][lid.x] = reduced;

    uint2 mip2Coord = tgid * 16 + lid;
    uint2 mip2Size  = srcSize >> 2;
    if (all(mip2Coord < mip2Size)) {
        pyramid.write(reduced, mip2Coord, 2);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mip 3: 16x16 to 8x8
    if (all(lid < uint2(8))) {
        reduced = reduce2x2(lds[ldsBase.y + 0][ldsBase.x + 0],
                            lds[ldsBase.y + 0][ldsBase.x + 1],
                            lds[ldsBase.y + 1][ldsBase.x + 0],
                            lds[ldsBase.y + 1][ldsBase.x + 1]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        lds[lid.y][lid.x] = reduced;
        
        uint2 mip3Size  = max(srcSize >> 3, uint2(1));
        uint2 mip3Coord = tgid * 8 + lid;
        if (all(mip3Coord < mip3Size)) {
            pyramid.write(reduced, mip3Coord, 3);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mip 4: 8x8 to 4x4
    if (all(lid < uint2(4))) {
        reduced = reduce2x2(lds[ldsBase.y + 0][ldsBase.x + 0],
                            lds[ldsBase.y + 0][ldsBase.x + 1],
                            lds[ldsBase.y + 1][ldsBase.x + 0],
                            lds[ldsBase.y + 1][ldsBase.x + 1]);
    
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lds[lid.y][lid.x] = reduced;
        
        uint2 mip4Coord = tgid * 4 + lid;
        uint2 mip4Size  = srcSize >> 4;
        if (all(mip4Coord < mip4Size)) {
            pyramid.write(reduced, mip4Coord, 4);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mip 5: 4x4 to 2x2
    if (all(lid < uint2(2))) {
        reduced = reduce2x2(lds[ldsBase.y + 0][ldsBase.x + 0],
                            lds[ldsBase.y + 0][ldsBase.x + 1],
                            lds[ldsBase.y + 1][ldsBase.x + 0],
                            lds[ldsBase.y + 1][ldsBase.x + 1]);
    
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lds[lid.y][lid.x] = reduced;
        
        uint2 mip5Coord = tgid * 2 + lid;
        uint2 mip5Size  = srcSize >> 5;
        if (all(mip5Coord < mip5Size)) {
            pyramid.write(reduced, mip5Coord, 5);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mip 6: 2x2 to 1x1
    if (lid_flat == 0) {
        reduced = reduce2x2(lds[0][0], lds[0][1], lds[1][0], lds[1][1]);
        
        uint2 mip6Coord = tgid;
        uint2 mip6Size  = srcSize >> 6;
        if (all(mip6Coord < mip6Size)) {
            pyramid.write(reduced, mip6Coord, 6);
        }
    }

    // last threadgroup computes remaining mips
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_texture | mem_flags::mem_threadgroup);

    if (lid_flat == 0) {
        uint completed = atomic_fetch_add_explicit(&globalCounter, 1, memory_order_relaxed) + 1;

        if (completed == consts.numWorkgroups) {
            atomic_store_explicit(&globalCounter, 0, memory_order_relaxed);

            pyramid.fence();

            for (uint mip = 7; mip < consts.mipCount; mip++) {
                uint2 dstSize    = max(srcSize >> mip,        uint2(1));
                uint2 srcMipSize = max(srcSize >> (mip - 1), uint2(1));
                
                for (uint y = 0; y < dstSize.y; y++) {
                    for (uint x = 0; x < dstSize.x; x++) {
                        uint2 readBase = uint2(x, y) * 2;
                        float4 t0 = pyramid.read(min(readBase + uint2(0, 0), srcMipSize - 1), mip - 1);
                        float4 t1 = pyramid.read(min(readBase + uint2(1, 0), srcMipSize - 1), mip - 1);
                        float4 t2 = pyramid.read(min(readBase + uint2(0, 1), srcMipSize - 1), mip - 1);
                        float4 t3 = pyramid.read(min(readBase + uint2(1, 1), srcMipSize - 1), mip - 1);
                        pyramid.write(reduce2x2(t0, t1, t2, t3), uint2(x, y), mip);
                    }
                }
                
                pyramid.fence();
            }
        }
    }
}
