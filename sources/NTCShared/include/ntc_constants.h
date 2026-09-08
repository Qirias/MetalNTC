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

#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>

#define K_HIDDEN     64
#define K_OUT_MAX    16
#define K_GRIDS      8
#define F_PER_GRID   16
#define MAX_LODS     13
#define PE_WAVES     3
#define PE_DIM       (4 * PE_WAVES)
#define F_TOTAL      (2 * F_PER_GRID)
#define F_IN_RAW     (F_TOTAL + PE_DIM + 1)
#define F_IN         (((F_IN_RAW + 15) / 16) * 16)

constant int SCALE = 1 << 24;

#else

#import <Foundation/Foundation.h>

static const NSInteger K_HIDDEN    = 64;
static const NSInteger K_OUT_MAX   = 16;
static const NSInteger K_GRIDS     = 8;
static const NSInteger F_PER_GRID  = 16;
static const NSInteger MAX_LODS    = 13;
static const NSInteger PE_WAVES    = 3;
static const NSInteger PE_DIM      = 4 * PE_WAVES;
static const NSInteger F_TOTAL     = 2 * F_PER_GRID;
static const NSInteger F_IN_RAW    = F_TOTAL + PE_DIM + 1;
static const NSInteger F_IN        = ((F_IN_RAW + 15) / 16) * 16;

#endif
