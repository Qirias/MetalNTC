# Integrating MetalNTC into another engine

Contract for consuming a `.ntc` in any Metal renderer — Swift **or**
metal-cpp/C++. This is a living document: append findings as real integrations
surface them. The section at the end collects what a completed port into a
production renderer actually cost.

## What you integrate

The **runtime decode**, per pixel: given a `.ntc`, a `uv`, and a mip `lod`,
reconstruct a `Material` (albedo, normal, roughness, metalness, occlusion,
emissive). The trainer (`NTCTrainer`) produces the `.ntc`; your renderer
consumes it. You do **not** need any of the training code at runtime.

There are two decode paths, and they need different renderer architectures:

| | where it runs | needs |
|---|---|---|
| **fragment** (`mlp_forward_h`) | inside your G-buffer / forward fragment shader | Metal 3, any Apple GPU |
| **tensor ops** (`mlp_forward_tensor_ops`) | a compute pass between the G-buffer and lighting | `-std=metal4.0`, M5-class GPU |

Start with the fragment path — it drops into an existing renderer with no
structural change. Move to tensor ops when you want the speed, and read
[Tensor ops path](#tensor-ops-path-m5-the-compute-pass-in-the-middle) first,
because it dictates where the decode lives.

## Reusable pieces (include or copy)

| File | Role | Portable to |
|---|---|---|
| `sources/NTCShared/include/ntc_constants.h` | Architecture constants (`K_HIDDEN`, `F_PER_GRID`, `F_IN`, …). | C, C++, MSL |
| `sources/NTCShared/include/ntc_file_format.h` | `.ntc` binary layout as C structs (`NTCHeader`, `NTCSlot`). Parse with these. | C, C++ |
| `sources/NTCCore/shaders/common.h` | The MSL decode. Fragment path: `sample_latent_grid`, `mlp_forward_h`, `ntc_decode_quant`. Tensor path: `ntc_build_features` (per pixel) + `mlp_forward_tensor_ops` (batched), behind `NTC_TENSOR_OPS`. Plus `StepConstants`. | MSL |

Reference consumer (the parts to mirror in your engine):
- **Shader, fragment path:** `sources/NTCRenderer/shaders/mesh.metal` —
  `sample_material`, the fragment bindings, `select_lod` (LOD + stochastic
  filtering).
- **Shader, tensor path:** same file — `gbuffer_fs` (writes the decode inputs)
  and `ntc_decode_shade` (the compute pass). `benchmark_decode` is a
  full-screen twin of `benchmark_fs` for A/B timing.
- **Host loader:** `sources/NTCRenderer/renderer.swift` — `buildNTCBuffers`,
  `buildLatentTexture`, `buildStepConstantsBuffer`, `MaterialLayout(slots:)`,
  and `makeSlotBuffer` for the bindless slots.

Both paths are behind one switch, `NTC_TENSOR_OPS` in `common.h`. With it off the
tensor entry points do not exist, so the host can select the fallback by checking
whether `makeFunction` returned nil — one source of truth, nothing to keep in
sync.

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

### Byte layout in detail

Little-endian, no padding beyond what the fields imply. Check
`header.signature == NTC_SIGNATURE` (`0x3143544E`, "NTC1") and
`header.version == NTC_VERSION` before reading anything else.

| # | Contents | Size |
|---|---|---|
| 1 | `NTCHeader` | 64 B (static-asserted) |
| 2 | `uint32_t pyramidSizes[header.kGrids]` | 4·kGrids |
| 3 | `uint32_t neuralMipsForLod[header.mipCount]` | 4·mipCount |
| 4 | `NTCSlot slots[header.nSlots]` | 32 B each (static-asserted) |
| 5 | grid, quantized feature ints | see below |
| 6 | `uint16_t mlp[mlpFloatCount]` | raw binary16 bits, when `mlpDType == NTC_MLP_DTYPE_FP16` |

**`NTCHeader`**, all `uint32_t` except where noted, in order: `signature`,
`version`, `srcW`, `srcH`, `mipCount` (= log2(srcW)+1), `kGrids`, `fPerGrid`,
`kHidden`, `kOutMax` (channel slots reserved in the MLP output), `kOut` (slot
channels actually used), `nSlots`, `peWaves`, `float quantScale`, `quantBits`,
`mlpDType`, `flags`.

Take the architecture from the header rather than assuming the compile-time
constants in `ntc_constants.h`. Those (`K_HIDDEN 64`, `K_GRIDS 8`,
`F_PER_GRID 16`, `K_OUT_MAX 16`, `MAX_LODS 13`, `PE_WAVES 3`) size your shader's
fixed-length arrays, but the file states what it was actually trained with.

**`NTCSlot`** (32 B): `char semantic[16]` null-padded ASCII (e.g. `"Albedo"`),
`char swizzle[8]` (e.g. `"RGB"`), `uint8_t channels` (1 or 3),
`uint8_t channelOffset` (base index into `kOut` — this is what `MaterialLayout`
is built from), `uint8_t sliceIndex`, `uint8_t isSRGB`, `uint32_t reserved`.

The slot table is the only place semantics are named, and the names are free-form
ASCII. Match on `semantic` to build your layout; unrecognised semantics are still
decoded, just unshaded.

**Grid**: `nInts = sum(pyramidSizes[i]^2) * fPerGrid` values, offset-binary in
`[0, 2^bits)`:

```
stored = clamp(round(f/q) + offset, 0, 2^bits - 1)
decode = (stored - offset) * q      offset = 1 << (bits-1),  q = quantScale
```

The format carries `quantBits` (4 or 8), but the trainer currently always writes
**4-bit** (`BITS` in `NTCTrainer/config.swift`). At 4 bits **two ints pack into
one byte, the even index in the low 4 bits `[0,4)` and the odd index in the high
`[4,8)`**, so the grid occupies `ceil(nInts/2)` bytes on disk; at 8 bits it is one
int per byte. Branch on `header.quantBits` rather than assuming 4, since that also
selects the upload format (`abgr4Unorm` vs `rgba8Unorm`). This on-disk packing is a
separate question from the `abgr4` channel order you upload it in — see Gotchas,
they do not agree.

**`mlpFloatCount`** is derived from the header, not stored:

```
F_IN  = roundUp(2*fPerGrid + 4*peWaves + 1, 16)
count = F_IN*kHidden + kHidden + kHidden*kHidden + kHidden + kHidden*kOutMax + kOutMax
```

### StepConstants

Declared in `NTCCore/shaders/common.h` and shared with the trainer, so it carries
training-only fields the decode never reads (`kBatch`, `total`, `lo`, `hi`,
`adamOffset`, `inferLod`, `bits`, `q`). They still occupy their slots — write the
whole struct in declaration order.

The decode reads only `pyramidSizes`, `neuralMipForLod`, `offsetW1`…`offsetB3`,
`kOut`, `posScale`, `srcW`, `srcH`, `mipCount`. Re-baseline the MLP offsets to 0,
since they are absolute within the trainer's own buffer.

## Shader side (fragment path)

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

For the fragment fallback this is a convenience. For the tensor ops path it is a
**requirement** — see below.

## Metal 3 vs Metal 4

The **fallback decode** is plain MSL and runs on Metal 3 — texture sampling, fp16
math, a buffer read. Nothing in it needs Metal 4.

The **tensor ops decode** needs `-std=metal4.0` for the shading language and
M5-class hardware for the matrix unit, but it does *not* need the Metal 4 host
API: the reference renderer drives it with `MTLCommandQueue` /
`MTLComputeCommandEncoder` and gets the full speedup. Only the shading-language
version matters.

## Tensor ops path (M5+): the compute pass in the middle

On M5-class hardware the MLP can run on the matrix unit instead of the ALU. It
is substantially faster, and close enough to what the hardware will do with a
well-fed GEMM that the MLP stops being the thing worth optimising.

The cost is structural: it cannot live in a fragment shader, so it changes where
the decode sits in your frame. That is what this section is about.

### The one thing that decides your architecture

**`execution_thread` does not reach the matrix unit.** At `m = 1` there is no
matrix to feed and `matmul2d` degrades to a per-lane loop — slower than the
hand-written `half4` GEMV it would replace. The speedup only exists when you
batch pixels into a real matrix with `execution_simdgroups<4>`.

And every scope except `execution_thread` accepts operands in **device or
threadgroup memory only** (`dv_`/`tg_` in the impl symbol names; no `th_`
variant). A fragment shader has no threadgroup address space. Apple states it
outright in `MPPTensorOpsMatMul2d.h`:

> *"Fragment shaders only support this execution scope"* — on `execution_thread`.

So the decode **cannot** stay in your G-buffer fragment shader. It has to become
a compute pass. That is not a performance trade-off, it is an address-space
constraint.

### Where it goes

```
geometry / G-buffer pass  →  NTC decode (compute)  →  your lighting / IBL pass
```

Your G-buffer fragment shader stops decoding. For neural pixels it writes the
*decode inputs* instead, and leaves the material factors in the usual targets.
The compute pass decodes `TILE_SIZE` (64) pixels per matmul and patches the
G-buffer in place, so everything downstream sees an ordinary G-buffer and needs
no changes.

Dispatch it as an 8×8 pixel tile per threadgroup with **128 threads**
(`execution_simdgroups<4>`, four simdgroups). Note 128 threads for 64 pixels:
threads past `TILE_SIZE` own no pixel but must still reach every matmul, because
the cooperative tensor is distributed across all their registers.

**What the G-buffer must carry for a neural pixel:**

| value | why it cannot be recomputed in compute |
|---|---|
| `uv` | interpolated; needs full float (fp16 bands visibly at 2048 texels) |
| `lod` | `select_lod` needs `dfdx/dfdy`, which compute does not have |
| `materialIndex` | to reach the right weights, factors and channel layout |
| world tangent + sign | to rebuild the TBN for the decoded normal |

That is more than fits in a spare channel. **Metal allows 8 colour attachments**
— if your G-buffer already uses 7, you get exactly one more. One `RGBA32Float`
is enough if you pack:

```
.xy  uv
.z   materialIndex | lod<<20 | tangentSign<<24   (all-ones = not neural)
.w   world tangent, octahedral 16:16
```

Clear `.z` to the "not neural" sentinel so background and non-neural pixels fall
out of the same test.

**Your G-buffer formats may not be writable from compute.** `MTLReadWriteTextureTier`
covers `r32*` at tier 1 and adds `rgba32*`, `rgba16*`, `rgba8unorm`, `r16*`,
`r8*` at tier 2. **No sRGB format is in either tier**, and G-buffers commonly
store albedo and emissive as `rgba8unorm_srgb`. Declaring one `access::write` or
`access::read_write` does not fail loudly — the writes simply never land, and you
get a G-buffer holding whatever the fragment shader left there. If your neural
materials render flat and untextured while everything else looks right, check
this first.

The fix is a linear view of the same storage: create the texture with
`MTLTextureUsagePixelFormatView`, make an `rgba8unorm` view for the compute pass
to write, and keep reading through the sRGB texture. The view bypasses the
hardware sRGB encode, so the shader has to apply it:

```metal
albedo.write(float4(linear_to_srgb(rgb), alpha), pixel);
```

Prefer taking material factors from your material struct over reading them back
out of the G-buffer — fewer reads, and fewer textures that need a writable view
at all.

**The cost to weigh:** that target is written then read every frame, which a
single-pass forward renderer never paid. At high resolution it is a real share
of the frame. A **tile shader** avoids it entirely — a kernel dispatched inside
the render pass has threadgroup memory *and* imageblock access, so the inputs
never leave tile memory. That is the better shape on Apple silicon and the one
lossless win still on the table; the compute pass is the simpler first step.

### Bindless is not optional here

A threadgroup shares one matmul, so it can only decode one material at a time.
The obvious response — dispatch once per material — **does not scale**:

> A draw call is bounded by its geometry. A dispatch is not; it covers exactly
> the grid you ask for.

Dispatching per material over the screen costs a full-screen scan *each*. With a
handful of materials that is invisible; with a real scene's worth it dominates
the frame, and almost every threadgroup launched does nothing but read the
G-buffer and early-out. The decode itself is entirely innocent, which makes it a
miserable thing to profile.

**Dispatch once, and let each threadgroup loop over only the materials its own
tile uses** — one or two in practice:

```metal
while (true) {
    if (tid == 0) atomic_store_explicit(&nextMaterial, NO_MATERIAL, ...);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pending) atomic_fetch_min_explicit(&nextMaterial, myMaterial, ...);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // uniform across the threadgroup, which it MUST be
    uint materialIndex = atomic_load_explicit(&nextMaterial, ...);
    if (materialIndex == NO_MATERIAL) break;

    device const NTCMaterialSlot& slot = slots[materialIndex];
    ... build features, matmul, write the pixels that match, retire them ...
}
```

Each pass retires at least one material, so it terminates. Cost follows screen
coverage instead of material count, and scales to any number of materials.

This requires every material's resources to be reachable without rebinding — a
buffer of slots holding `MTLResourceID` / `gpuAddress` per material:

```metal
struct NTCMaterialSlot {
    texture2d_array<float>      latents;
    device const half*          mlp;
    device const StepConstants* consts;
    MaterialLayout              layout;
    float2                      gridDequant;
};
```

Fill it host-side from `texture.gpuResourceID` and `buffer.gpuAddress`, and
`useResource` each underlying resource on the encoder — the slot buffer reaches
them by address, so residency is not inferred.

The fallback path does **not** need this. It binds per draw call, and the
rasteriser bounds each draw to its own triangles.

## Gotchas

- **abgr4 channel order:** in the 16-bit texel `.r` is the *most significant* 4
  bits `[12,16)` and `.a` the least `[0,4)`. Pack feature `4·slice+0` into the
  MSBs. Verify packed formats with a 1×1 probe; the docs had this backwards.
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

## Integration notes

Failure modes and design decisions that are not visible from the API surface.

### Before anything: producing a `.ntc`

**Build in Xcode.** A plain `swift build` copies the `.metal` sources into the
`NTCCore` resource bundle but does **not** compile them, so the
`default.metallib` that `MetalContext(bundle:)` opens is never produced and the
trainer aborts with *no default library was found*. Xcode compiles the shaders
as part of building the package; the SwiftPM command line does not. This blocks
step zero, before any integration work starts.

The consequence for an integrator is worth stating plainly: **do not wire
`swift build` into your engine's build system** expecting a runnable trainer.
Producing a `.ntc` is an offline step you do in Xcode, and the engine consumes
the file afterwards. Anything that shells out to SwiftPM has to compile and
install the metallib itself, which is a moving part with no upside.

Related: use the **Release** configuration. The image loader's per-pixel
`UInt8 → Float` and swizzle loops are ~490× slower unoptimised, which turns a
2048² decode from 80 ms into 6.8 s and a few hundred textures into minutes.
`NTC_DEBUG` (`Package.swift`) is independent of the optimisation level, so the
PSNR tables and atlas dumps survive a Release build.

### `.ntc` carries no alpha

Nothing in the pipeline preserves an alpha channel, and it is dropped twice, in
two independent places:

- `NTCTrainer/manifest_gen.swift` maps `baseColorTexture` as
  `["Albedo": "RGB"]`. The `A` channel is never given a semantic, so the trainer
  is never asked to learn it.
- `NTCAssets/image_loader.swift` decodes with `CGImageAlphaInfo.noneSkipLast`,
  which retains no alpha at all — the source is composited over a zeroed buffer.

Two consequences for an integrator:

1. **Cutout / alpha-masked materials still need their source base-colour
   texture**, purely for its alpha. Decode albedo from the `.ntc`, sample the
   original texture for the mask. Budget for those textures staying resident.
2. Because the trainer composites over black, transparent regions decode to
   black rather than to the source colour. **When scoring PSNR against the
   source, multiply the reference by its own alpha first.** Skipping that scores
   far below the true figure and sends you hunting a decode bug that does not
   exist.

Supporting alpha means changing both layers — a `premultipliedLast`/`last`
decode plus an `"Alpha": "A"` mapping — and retraining.

### Finding the `.ntc`: the filename carries the quality

The trainer writes **one `.ntc` per quality profile**, and tags the filename with
it (`Quality.ntcFileName`, `NTCCore/quality.swift`):

```
<material name>_<quality>.ntc      e.g. Foliage_Leaves_high.ntc
```

`<quality>` is one of `low`, `medium`, `high`, `veryHigh`. The material name is
the glTF material name with every character outside `[A-Za-z0-9-_.]` replaced by
`_`, and an unnamed material falls back to `material_<index>` — mirror
`ManifestGen.safeName` exactly or nothing resolves.

Getting this wrong fails **silently**: a missing `.ntc` is also the legitimate
"this material is not neural" signal, so the loader falls back to textures and
the scene renders correctly from the source images. Log the miss during bring-up,
or you will spend a while wondering why nothing changed.

**The profile only changes storage, never decode cost.** `Quality.gridScale`
divides the source width to size the finest latent grid (`veryHigh` ÷3, `high`
÷4, `medium` ÷6, `low` ÷8); the eight grids, the 16 features per texel, `F_IN`
and the MLP shape are identical across all four. Same sample count, same MAC
count, same frame time. So an engine can mix profiles per material — `veryHigh`
on hero assets, `low` on background — with no performance variance, and can
change profile without re-tuning anything in the shader.

### Not every material can be trained

Two conditions take a material out of the neural path, and both are decided
while training, so your loader has to cope with a `.ntc` simply not existing.

**Unsupported source width.** `MIP_MAPS` (`NTCTrainer/config.swift:10`) is
keyed by the **exact** source width: 4096, 2048, 1024, 512. Any other width
throws `no mip map for WxH; add one to MIP_MAPS`. A material whose only texture
is a small solid-colour swatch cannot be compressed at all.

**Mixed texture resolutions inside one material.** Every slot of a model feeds
one texture array and one latent grid, so the whole set has to agree on
resolution. A model mixing, say, a 2048² roughness map with 512² everything else
throws `mixed texture sizes: ...` (`NTCTrainer/manifest.swift`) and is skipped.
This is common in real assets — Amazon Lumberyard Bistro has 380 textures at 512²
and a single 2048² outlier, which is enough to disqualify the material that uses
it.

The trainer skips rather than resamples, because either direction is a quality
decision the tool should not make silently. If you want those materials on the
neural path, resample in `loadImages()` before the size check, and pick the
direction deliberately:

- **Upscale the small ones to the largest slot.** Preserves the detailed maps and
  — because latent storage is set by grid resolution and is *flat in channel
  count* — costs **nothing** in `.ntc` size. An upscaled 512² roughness map adds
  no bytes; it just cannot carry detail it never had. This is the right default.
- **Downscale everything to the smallest slot.** Shrinks the `.ntc`, but throws
  away real detail in the albedo and normal maps to accommodate whichever slot
  happened to be lowest-resolution. Rarely what you want.

Until then such materials stay on traditional texture sampling.

Both are thrown, not trapped, so a batch of hundreds of models keeps going and
reports what it skipped at the end.

So your material system needs a **per-material fallback to the conventional
texture path**, selected at load time. Treat "is neural" as a property of the
individual material, never an assumption about the asset set.

### "Has a `.ntc`" is not "is decoded"

The decode lives in whichever shaders you added it to. If your renderer routes
some materials to a *different* shading path — a forward or OIT pass for blended
materials, for instance — those materials will have no decode branch and will
render from their factors alone, untextured.

Resolve it at load time: if a material's shading path has no decode branch, do not
resolve its `.ntc`, and let it keep its source textures. Deciding this per shading
path is much cheaper than discovering it as a shading bug later.

### Do not decode in passes that need one channel

The decode is a per-pixel MLP: two `texture2d_array` samples plus
`F_IN*kHidden + kHidden^2 + kHidden*kOutMax` fp16 MACs per invocation. That is
affordable once per shaded pixel and wasteful anywhere else.

Any pass that historically sampled a material texture for a *single* channel must
not run the decode. The common case is a depth pre-pass or a shadow pass alpha
testing against base-colour alpha: full material reconstruction to obtain one
scalar, and the `.ntc` does not carry alpha at all (see below), so the decode
cannot answer that question even at full cost.

Options, in order of preference:

1. Keep the source texture resident for exactly the channel that pass needs and
   sample it there. For alpha testing this is mandatory, not an optimisation.
2. Return early for materials where the pass has nothing to do — an opaque
   material needs no alpha test, so it should exit before any sampling.
3. Bake the value into the material struct when it is constant per material.

Rule: decode once, in the pass that produces shading (G-buffer or forward), and
let every other pass read a cheaper source. If your renderer has a depth
pre-pass, this also means the decode runs only on surviving fragments, which is
where the arrangement pays for itself.

### Skipping texture uploads makes other passes reachable

The payoff for neural materials is *not* uploading the source textures. That
quietly changes the preconditions of every other pass that sampled them.

A depth pre-pass or shadow pass that samples the base-colour texture for alpha
testing typically does so before checking the alpha mode. Once most materials no
longer have a base-colour texture, that unguarded sample becomes an out-of-bounds
read on a real code path. Audit every pass that touches material textures, and
have each one return early unless it is genuinely a masked material.

### Growing the material struct is a build-system hazard

This is the expensive one, and it does not look like a build problem.

You will add the six layout ints, the two dequant floats and a resource index to
whatever material struct your CPU and shaders already share. That changes the
struct's **size**. If only one side rebuilds, CPU and GPU disagree on the array
stride, and:

- `materials[0]` stays correct — index 0 is at offset 0 under either stride.
  **Every other material reads garbage.**
- Fields land at plausible offsets, so values are wrong but not obviously
  corrupt: an `alphaMode` that decodes to the wrong enum, a texture index
  pointing into another material's textures.
- Every CPU-side check passes. Log the struct: correct. Inspect the buffer:
  correct. The corruption exists only in how the shader indexes it.

The usual cause is a build rule that compiles each `.metal` with a dependency on
only that `.metal`, not on the headers it includes. Editing the shared header
rebuilds your C++ and leaves the shader binaries stale. **Make your shader
compilation depend on the headers**, and after any change to a shared struct,
confirm the shader objects are newer than the header before debugging anything
else.

Related: use a **signed** type for the six layout fields. `-1` is the absent
sentinel and does not coexist with the `0xFFFFFFFF` sentinel unsigned texture
indices normally use.

### Which semantics need a presence test

The neutral fallbacks are not equally neutral.

- **Normal needs no test.** The fallback is `(0.5, 0.5, 1)`, unpacking to
  `(0, 0, 1)`. If your TBN places the interpolated vertex normal in its third
  column — the usual construction — then `TBN * (0,0,1)` *is* that normal, so
  decoding unconditionally is already the identity when the `.ntc` has no normal
  map. A `hasNormal` flag is dead weight. It does rely on `0 * tangent == 0`, so
  it holds only if your tangents cannot be NaN.
- **Emissive needs one.** Its fallback is `0`, which is not the same as "no
  emissive data, so use the material's emissive factor". Without an explicit
  `layout.emissive >= 0` test, every emissive-factor-only material goes black.
- Roughness (1), metalness (0) and occlusion (1) are scalars whose fallbacks are
  genuinely neutral.

Test `layout.<semantic> >= 0` at the point of use rather than carrying parallel
booleans through your material struct.

### UV wrapping does not survive — you must `fract` it yourself

A `.ntc` is trained over `uv` in `[0,1]` and the latent sampler is
`clamp_to_edge`, so `address::repeat` no longer means anything. **Any uv outside
`[0,1]` gets pinned to the texture border**, which on a tiled surface is not a
subtle artefact — pavement and walls come out unrecognisable while unwrapped
geometry beside them looks perfect. That split is the signature of this bug.

You have to wrap it yourself:

```metal
uv = fract(uv);
```

**Order matters.** The LOD needs the *unwrapped* derivatives; the decode needs
the *wrapped* uv. Fract before `select_lod` and you corrupt the LOD at every tile
seam:

```metal
uint lod = select_lod(uvdx, uvdy, consts);   // unwrapped
... decode(fract(uv), lod, ...)              // wrapped
```

Where it goes differs by path:

| path | where |
|---|---|
| fragment | at the decode call, after the LOD is taken |
| tensor ops | in the G-buffer fragment shader, on the uv you write out — the compute pass has no derivatives, so the LOD is already resolved by then |

Two things to know beyond getting it working. The latent grid does not match
across the seam the way a repeating texture did, so expect a visible
discontinuity on heavily tiled surfaces. And **the usual demo assets cannot catch
this** — a model whose UVs are already in `[0,1]` looks correct either way. Test
any uv change on something tiled.

### Bring-up tooling worth writing first

Three small harnesses that repay their cost immediately:

- A **probe** that parses a `.ntc` and dumps the header, slot table and the fully
  populated `StepConstants`. `StepConstants` is filled by a byte cursor, so
  reading it back is the only way to catch layout drift.
- A **PSNR harness** that decodes through *your engine's* shader path, not the
  reference one, and scores against the source image. This is what distinguishes
  "the decode is wrong" from "the material plumbing is wrong". Mind the alpha
  caveat above.
- A **1×1 probe** for packed formats, to confirm which 4 bits of an `abgr4`
  texel each channel occupies on your hardware rather than trusting documentation.
