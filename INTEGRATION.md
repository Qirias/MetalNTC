# Integrating MetalNTC into another engine

Contract for consuming a `.ntc` in any Metal renderer — Swift **or**
metal-cpp/C++. This is a living document: append findings as the real
integration (Anthos) surfaces them.

## What you integrate

The **runtime decode**, per pixel: given a `.ntc`, a `uv`, and a mip `lod`,
reconstruct a `Material` (albedo, normal, roughness, metalness, occlusion,
emissive). The trainer (`NTCTrainerCLI`) produces the `.ntc`; your renderer
consumes it. You do **not** need any of the training code at runtime.

## Reusable pieces (include or copy)

| File | Role | Portable to |
|---|---|---|
| `sources/NTCShared/include/ntc_constants.h` | Architecture constants (`K_HIDDEN`, `F_PER_GRID`, `F_IN`, …). | C, C++, MSL |
| `sources/NTCShared/include/ntc_file_format.h` | `.ntc` binary layout as C structs (`NTCHeader`, `NTCSlot`). Parse with these. | C, C++ |
| `sources/NTCCore/shaders/common.h` | The MSL decode: `sample_latent_grid`, `mlp_forward_h`, `ntc_decode_quant`, `StepConstants`. `#include` it in your fragment shader. | MSL |

Reference consumer (the parts to mirror in your engine):
- **Shader:** `sources/NTCRenderer/shaders/mesh.metal` — `sample_material`, the
  fragment bindings, `select_lod` (LOD + stochastic filtering).
- **Host loader:** `sources/NTCRenderer/renderer.swift` — `buildNTCBuffers`,
  `buildLatentTexture`, `buildStepConstantsBuffer`, `MaterialLayout(slots:)`.

## Host side: parse `.ntc` → 5 GPU resources

The `.ntc` is: `NTCHeader` (64 B) → `pyramidSizes[kGrids]` →
`neuralMipsForLod[mipCount]` → `NTCSlot[nSlots]` (32 B each) → `grid` (quantized
feature ints) → `mlp` (fp16). Full layout in `ntc_file_format.h`. From it, build:

| Resource | How | Reference |
|---|---|---|
| **Latent texture** | `texture2d_array`, `abgr4Unorm` (4-bit) or `rgba8Unorm` (8-bit). `arrayLength = fPerGrid/4`, `mipmapLevelCount = kGrids`. Grid `m` → mip `m` (size `pyramidSizes[m]`); features `4·slice+c` → channel `c` of slice. | `buildLatentTexture` |
| **MLP buffer** | Raw fp16, layout `[W1 (F_IN·kHidden)][B1][W2 (kHidden²)][B2][W3 (kHidden·kOutMax)][B3]`. Weights are **output-major**: `w[out·n_in + in]`. | `buildNTCBuffers` |
| **StepConstants** | The `struct StepConstants` in `common.h`, filled field-by-field (order is load-bearing). MLP offsets re-baselined to 0. Decode reads: `pyramidSizes`, `neuralMipForLod`, `offsetW1..B3`, `kOut`, `posScale`, `srcW/srcH/mipCount`. | `buildStepConstantsBuffer` |
| **MaterialLayout** | 6 `int`s (albedo, normal, roughness, metalness, occlusion, emissive) = base channel index into the MLP output, or `-1` if that semantic is absent. Derived from the slot table. | `MaterialLayout(slots:)` |
| **gridDequant** | `float2(scale, bias)`, `scale = (2^bits − 1)·quantScale`, `bias = −2^(bits−1)·quantScale`. Applied to the sampled latent before the MLP. | `loadNTCResource` |

## Shader side

`#include "common.h"`, then per fragment:

```metal
constexpr sampler latentSampler(coord::normalized, address::clamp_to_edge,
                                filter::linear, mip_filter::linear);

half pred[K_OUT_MAX];
ntc_decode_quant(uv, lod, latents, latentSampler,
                 gridDequant.x, gridDequant.y, mlp, consts, pred);
// map channels with MaterialLayout; pred[layout.albedo..+3], etc.
```

`ntc_decode_quant` samples two latent grids by HW bilinear, dequantizes, adds the
positional encoding + LOD, and runs the 3-layer fp16 MLP. `sample_material` in
`mesh.metal` shows the channel mapping and neutral defaults.

**Fragment bindings used by the reference (`mesh_fs`):**

| Slot | Binding |
|---|---|
| `texture(0)` | latent texture2d_array |
| `buffer(1)` | mlp (`device const half*`) |
| `buffer(2)` | StepConstants |
| `buffer(3)` | MaterialLayout |
| `buffer(4)` | gridDequant (`float2`) |
| `buffer(5)` | frameIndex (`uint`, for stochastic LOD) |

`select_lod` derives a continuous LOD from `dfdx/dfdy(uv)` and stochastically
picks the bracketing mip (needs `frameIndex` + a temporal resolve to denoise).
For a first bring-up you can skip STF and pass `lod = round(continuousLod)`.

## Swift vs metal-cpp / C++

- `ntc_constants.h` and `ntc_file_format.h` are already plain C — `#include`
  them directly from a metal-cpp engine and parse the file with `NTCHeader` /
  `NTCSlot`. No Swift needed.
- The decode `.metal` (`common.h`) is standard MSL — include it in your engine's
  shaders unchanged.
- Only the **host loader** (~200 lines in `renderer.swift`) needs porting to
  `MTL::` calls. It is mechanical: `makeTexture`/`replace`, `makeBuffer`, and the
  byte-cursor fill of `StepConstants`.
- Put `StepConstants` and `MaterialLayout` in a shared C header so Swift, C++,
  and MSL agree on the layout (the reference declares them in `common.h` for MSL
  and mirrors them in Swift).

## Bindless

Per material, the NTC resources are: latent texture + mlp buffer + consts buffer
(larger, heap/argument-buffer resident, indexed by material id) and gridDequant +
MaterialLayout (tiny, inline them in your material struct). This is a natural fit
for a `materialIndex → NTC resource` argument buffer.

## Metal 3 vs Metal 4

The runtime decode is **plain MSL and runs on Metal 3** — texture sampling, fp16
math, a buffer read. Nothing in the decode path needs Metal 4. Only the *trainer*
uses the Metal 4 command/queue API. So a Metal 3 engine can consume `.ntc` today;
the Metal 5 tensor path (`matmul2d`) would only swap the MLP leaf later.

## Gotchas

- **abgr4 channel order (M1):** `.r` is the *high* nibble `[12,16)`, `.a` the low
  `[0,4)`. Pack feature `4·slice+0` into the high nibble. Verify packed formats
  with a 1×1 probe; the docs had this backwards.
- **Align-corners remap:** the trainer's bilinear is align-corners; the HW
  sampler is texel-center. `sample_latent_grid` feeds `uv' = (uv·(N−1)+0.5)/N` to
  reproduce it exactly. Keep that remap or the reconstruction shifts.
- **fp16 MLP, output-major weights** `w[out·n_in + in]`; **zero the padding
  lanes** `features[F_IN_RAW..F_IN)` before the MLP (padding weights are nonzero).
- **posScale = srcW / 8** — base frequency of the positional encoding.
- **neuralMipForLod** maps a render LOD → the base grid index of its grid pair;
  pad the tail out to `MAX_LODS` (with the finest entry) when filling the uniform.
- **sRGB:** decoded albedo/emissive are sRGB-*encoded* values (the trainer learned
  them directly from sRGB PNGs; `slot.isSRGB` flags which). Linearize before
  lighting, or write to a non-sRGB target if passing through.
- **StepConstants field order is load-bearing** — it is filled by a raw byte
  cursor, so any struct edit must be mirrored in the fill code.

## Findings log

Append notes here as MetalNTC is integrated into Anthos (metal-cpp, bindless,
Metal 3 → 4). Nothing yet.
