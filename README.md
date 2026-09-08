# MetalNTC

Metal 4 implementation of Neural Texture Compression (NTC): compress a set of
material textures into feature grids + a small MLP and decode per pixel at render
time.

- **Trainer** will produce a `.ntc` file for each mesh in the `.gltf`.
- **Renderer** is an optional submodule ([MetalNTC-Renderer](https://github.com/Qirias/MetalNTC-Renderer)) that displays a glTF shaded from its `.ntc`.

## Install

```sh
git clone https://github.com/Qirias/MetalNTC.git

# also get the renderer + demo assets
git clone --recurse-submodules https://github.com/Qirias/MetalNTC.git
# (or in an existing clone) git submodule update --init sources/NTCRenderer
```

```sh
cd MetalNTC
open Package.swift
```

Then pick a scheme (`NTCTrainer` or `NTCRenderer`) and build.
Requires macOS 26 + Xcode 26 (Metal 4) on Apple silicon.

### Debug or release

Product -> Scheme -> Edit Scheme...-> **Run** -> **Info** -> Build Configuration.

Prefer **Release**.

## Run

- **NTCTrainer** opens a file picker to choose a `.gltf`,
  `manifest.json`, or texture folder. Then opens the quality setting selection. It writes `<material>_<quality>.ntc` beside
  the input. (Set `INPUT_OVERRIDE` in `sources/NTCTrainer/main.swift` to skip
  the picker.)
- **NTCRenderer** opens a window and renders the demo model from its `.ntc`.

## Integrate into your engine

See [LLM.md](LLM.md).


## License

**[MIT License](https://github.com/Qirias/MetalNTC/blob/main/LICENSE)**, free to use with attribution.