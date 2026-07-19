import Foundation
import NTCAssets

public struct Manifest: Decodable {
    public struct Entry: Decodable {
        public let fileName: String
        public let isSRGB: Bool?
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
        for (i, entry) in manifest.textures.enumerated() {
            precondition(entry.semantics.count == 1,
                         "\(entry.fileName): bootstrap loader requires exactly one semantic per texture (got \(entry.semantics.count))")
            let (semantic, swizzle) = entry.semantics.first!
            let channels = swizzle.count
            precondition(channels == 1 || channels == 3,
                         "\(entry.fileName): swizzle '\(swizzle)' has \(channels) channels; image_loader supports 1 or 3")
            built.append(TextureSlot(fileName: entry.fileName,
                                     semantic: semantic,
                                     swizzle: swizzle,
                                     channels: channels,
                                     isSRGB: entry.isSRGB ?? false,
                                     sliceIndex: i,
                                     channelOffset: offset))
            offset += channels
        }
        self.slots = built
        self.kOut = offset
    }

    public func loadImages(width: Int, height: Int) throws -> [LoadedImage] {
        return try slots.map { s in
            let url = manifestDir.appendingPathComponent(s.fileName)
            let img = try load_image(at: url, channels: s.channels)
            precondition(img.width == width && img.height == height,
                         "\(s.fileName): expected \(width)x\(height), got \(img.width)x\(img.height)")
            return img
        }
    }
}
