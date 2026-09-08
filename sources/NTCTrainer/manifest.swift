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
import NTCAssets

/// One manifest.json describes a whole glTF: a list of models (one per
/// submesh/material). One .ntc is trained per model, so a crash mid-training
/// only loses the model in flight, never the whole set.
public struct Manifest: Codable {
    public struct Entry: Codable {
        public let fileName: String
        public let isSRGB: Bool?
        /// semantic -> swizzle, e.g. {"Roughness": "G", "Metalness": "B"}.
        /// One file may feed several semantics; glTF packs roughness and
        /// metalness into a single metalRoughness texture.
        public let semantics: [String: String]
    }
    /// One trainable unit: the texture slots of a single glTF material.
    /// `name` becomes the .ntc basename; the quality is appended, so the file is
    /// `<name>_<quality>.ntc` (see Quality.ntcFileName).
    public struct Model: Codable {
        public let name: String
        public let textures: [Entry]
    }
    public let models: [Model]

    /// Load a manifest.json: `{ "models": [ { "name", "textures" } ] }`.
    /// A malformed file throws out of JSONDecoder with the offending key path.
    public static func load(from url: URL) throws -> Manifest {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Manifest.self, from: data)
    }
}

/// TODO: add verticalFlip
public struct TextureSlot {
    public let fileName: String
    public let semantic: String
    public let swizzle: String
    public let channels: Int
    public let isSRGB: Bool
    public let sliceIndex: Int
    public let channelOffset: Int
}

public struct TextureSet {
    public let name: String        // model name -> <name>_<quality>.ntc
    public let slots: [TextureSlot]
    public let kOut: Int
    public let manifestDir: URL

    /// Build the texture set for a single model (submesh/material). `dir` is the
    /// glTF's directory; entry file names resolve against it (they may include a
    /// `textures/` subpath).
    public init(model: Manifest.Model, dir: URL) {
        self.name = model.name
        self.manifestDir = dir

        var built: [TextureSlot] = []
        var offset = 0
        for entry in model.textures {
            precondition(!entry.semantics.isEmpty, "\(entry.fileName): no semantics declared")
            // Dictionary order is not stable across runs, so sort by semantic
            // name. Channel offsets are baked into the .ntc, and a run-to-run
            // reshuffle would silently change what each output channel means.
            for semantic in entry.semantics.keys.sorted() {
                let swizzle  = entry.semantics[semantic]!
                let channels = swizzle.count
                precondition(channels == 1 || channels == 3,
                             "\(entry.fileName): swizzle '\(swizzle)' has \(channels) channels; only 1 or 3 supported")
                built.append(TextureSlot(fileName: entry.fileName,
                                         semantic: semantic,
                                         swizzle: swizzle,
                                         channels: channels,
                                         isSRGB: entry.isSRGB ?? false,
                                         sliceIndex: built.count,
                                         channelOffset: offset))
                offset += channels
            }
        }
        self.slots = built
        self.kOut = offset
    }

    /// Decodes each distinct file once and extracts every slot that reads from
    /// it. Source dimensions come from the images themselves -- all slots must
    /// agree, since they share one texture array.
    public func loadImages() throws -> (images: [LoadedImage], width: Int, height: Int) {
        var decoded: [String: RGBAImage] = [:]
        var images: [LoadedImage] = []

        for s in slots {
            let rgba: RGBAImage
            if let cached = decoded[s.fileName] {
                rgba = cached
            } else {
                rgba = try decode_rgba(at: manifestDir.appendingPathComponent(s.fileName))
                decoded[s.fileName] = rgba
            }
            images.append(try extract_swizzle(rgba, swizzle: s.swizzle))
        }

        guard let first = images.first, let firstSlot = slots.first else {
            preconditionFailure("model '\(name)' declares no textures")
        }
        
        // could be handled by upscaling the low resolution textures to match
        for (s, img) in zip(slots, images) {
            if img.width != first.width || img.height != first.height {
                throw TrainerError.msg("mixed texture sizes: \(s.fileName) is \(img.width)x\(img.height) but \(firstSlot.fileName) is \(first.width)x\(first.height); NTC needs one resolution per model")
            }
        }
        precondition(first.width == first.height, "non-square sources not supported (got \(first.width)x\(first.height))")
        precondition(first.width > 0 && (first.width & (first.width - 1)) == 0, "source width \(first.width) is not a power of two")

        return (images, first.width, first.height)
    }
}
