import Foundation
import NTCCore
import NTCShared

struct PyramidPreset {
    let sizes:  [Int]      // K_GRIDS grid resolutions, finest first
    let mipMap: [UInt32]   // lod -> index of the first grid in its pair
}

let MIP_MAPS: [Int: [UInt32]] = [
    4096: [0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6],
    2048: [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6],
    1024: [0, 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6],
]

func pyramidPreset(srcW: Int, quality: Quality) -> PyramidPreset? {
    guard let mipMap = MIP_MAPS[srcW] else {
        return nil
    }
    let base  = srcW / quality.gridScale
    let sizes = (0..<K_GRIDS).map { grid in
        max(base >> grid, 1)
    }
    return PyramidPreset(sizes: sizes, mipMap: mipMap)
}

let K_BATCH = 4096

let BITS: UInt32 = 4

let IMPORTANCE_WEIGHTS: [String: Float] = [
    "Albedo":       2.0,
    "Normal":       1.0,
    "Roughness":    0.35,
    "Metalness":    0.35,
    "Occlusion":    0.35,
    "Displacement": 0.35,
    "AlphaMask":    0.35,
]

func fake_quant(bits: UInt32) -> (q: Float, lo: Float, hi: Float) {
    let N = Float(1 << bits)
    let q = 1.0 / N
    let lo = -(N - 1) / 2 * q
    let hi =  N / 2 * q
    return (q, lo, hi)
}

let SAMPLE_X      = 0
let SAMPLE_Y      = 1
let SAMPLE_LOD    = 2
let SAMPLE_LOSS   = 3
let SAMPLE_STRIDE = 4

let nSteps  = 10000
let logEvery = 100
let BETA1: Float = 0.9
let BETA2: Float = 0.999
let LR_GRID_MAX: Float = 0.01
let LR_MLP_MAX:  Float = 0.005
let UNIFORM_LOD_FRACTION: Float = 0.05

// cosine annealing to 0 across [0, total)
func cosineLr(step: Int, total: Int, lrMax: Float) -> Float {
    let denom = Float(max(total - 1, 1))
    let t = Float(step) / denom
    return lrMax * 0.5 * (1.0 + cosf(.pi * t))
}

// choose randomly a level proportionally to the mip level's area by sampling
// from an exponential distribution. To mitigate undersampling of low resolution
// mip levels, 5% of the batches sample their LOD from a uniform distribution
// of the entire range of the mip chain
func sampleBatchLod(lodMax: Int) -> Int {
    if Float.random(in: 0..<1) < UNIFORM_LOD_FRACTION {
        return Int.random(in: 0..<lodMax)
    }
    let x = Float.random(in: Float.leastNormalMagnitude..<1.0)
    let lod = Int(floor(-logf(x) / logf(4.0)))
    return min(max(lod, 0), lodMax - 1)
}
