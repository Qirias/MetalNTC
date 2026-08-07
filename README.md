# MetalNTC

Metal 4 implementation of Neural Texture Compression (NTC): compress a set of
material textures into feature grids + a small MLP, decode per pixel at render
time.

- **Trainer** — turns textures into a `.ntc` file.
- **Renderer** — optional submodule ([MetalNTC-Renderer](https://github.com/Qirias/MetalNTC-Renderer)) that displays a glTF shaded from its `.ntc`. Holds the demo assets, so a bare clone stays small.

## Install

```sh
# core + trainer only
git clone https://github.com/Qirias/MetalNTC.git

# also get the renderer + demo assets
git clone --recurse-submodules https://github.com/Qirias/MetalNTC.git
# (or in an existing clone) git submodule update --init sources/NTCRenderer
```

Open the package in **Xcode** and build. Xcode compiles the Metal shaders into
`default.metallib`; a plain `swift build` does not, so use Xcode.

Requires macOS 26 + Xcode 26 (Metal 4) on Apple silicon.

## Run

- **NTCTrainerCLI** — run it; a file picker opens to choose a `.gltf`,
  `manifest.json`, or texture folder. It writes `<material>.ntc` beside the input.
  (You can also pass a path as the first argument, or set `INPUT_OVERRIDE` in
  `sources/NTCTrainerCLI/main.swift`.)
- **NTCRenderer** — run it (needs the submodule); opens a window and renders the
  demo model from its `.ntc`.

## Integrate into your engine

See [INTEGRATION.md](INTEGRATION.md).
