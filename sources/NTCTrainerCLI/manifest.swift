import Foundation
import NTCAssets

public struct Manifest: Codable {
    public struct Entry: Codable {
        public let fileName: String
        public let isSRGB: Bool?
        /// semantic -> swizzle, e.g. {"Roughness": "G", "Metalness": "B"}.
        /// One file may feed several semantics; glTF packs roughness and
        /// metalness into a single metalRoughness texture.
        public let semantics: [String: String]
    }
    public let textures: [Entry]
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
    public let slots: [TextureSlot]
    public let kOut: Int
    public let manifestDir: URL

    public init(manifestURL: URL) throws {
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        self.manifestDir = manifestURL.deletingLastPathComponent()

        var built: [TextureSlot] = []
        var offset = 0
        for entry in manifest.textures {
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

        guard let first = images.first else {
            preconditionFailure("manifest declares no textures")
        }
        for (s, img) in zip(slots, images) {
            precondition(img.width == first.width && img.height == first.height,
                         "\(s.fileName): \(img.width)x\(img.height) does not match \(first.width)x\(first.height)")
        }
        precondition(first.width == first.height, "non-square sources not supported (got \(first.width)x\(first.height))")
        precondition(first.width > 0 && (first.width & (first.width - 1)) == 0, "source width \(first.width) is not a power of two")

        return (images, first.width, first.height)
    }
}
