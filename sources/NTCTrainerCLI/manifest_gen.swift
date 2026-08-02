import Foundation

/// The mapping is mechanical (glTF PBR metallic-roughness -> NTC semantics):
///   baseColorTexture         -> Albedo    RGB (sRGB)
///   normalTexture            -> Normal    RGB
///   metallicRoughnessTexture -> Roughness G  + Metalness B
///   occlusionTexture         -> Occlusion R
///   emissiveTexture          -> Emissive  RGB (sRGB)
enum ManifestGen {

    // Only the fields we map. glTF references a texture by index; a texture
    // points at an image `source`; the image carries the `uri` (the file name).
    private struct GLTFDoc: Decodable {
        struct TexRef:   Decodable {
            let index: Int
        }
        struct PBR:      Decodable {
            let baseColorTexture: TexRef?
            let metallicRoughnessTexture: TexRef?
        }
        struct Material: Decodable {
            let pbrMetallicRoughness: PBR?
            let normalTexture:    TexRef?
            let occlusionTexture: TexRef?
            let emissiveTexture:  TexRef?
        }
        struct Texture: Decodable {
            let source: Int
        }
        struct Image:   Decodable {
            let uri: String
        }
        let materials: [Material]?
        let textures:  [Texture]?
        let images:    [Image]?
    }

    enum Err: Error, CustomStringConvertible {
        case msg(String)
        var description: String {
            switch self {
                case .msg(let m): return m
            }
        }
    }

    /// Reads the glTF material block, writes `manifest.json` beside the .gltf,
    /// and returns the URL of the written file.
    @discardableResult
    static func generate(fromGLTF gltfURL: URL) throws -> URL {
        let data = try Data(contentsOf: gltfURL)
        let doc  = try JSONDecoder().decode(GLTFDoc.self, from: data)

        guard let materials = doc.materials, let material = materials.first else {
            throw Err.msg("glTF \(gltfURL.lastPathComponent) declares no materials")
        }
        if materials.count > 1 {
            FileHandle.standardError.write(Data(
                "warning: glTF has \(materials.count) materials; using the first (renderer supports one)\n".utf8))
        }

        let textures = doc.textures ?? []
        let images   = doc.images   ?? []

        // texture index -> image file name, or nil if the ref/source is missing.
        func fileName(_ ref: GLTFDoc.TexRef?) -> String? {
            guard let ref, ref.index >= 0, ref.index < textures.count else { return nil }
            let src = textures[ref.index].source
            guard src >= 0, src < images.count else { return nil }
            let uri = images[src].uri
            return uri.removingPercentEncoding ?? uri
        }

        var entries: [Manifest.Entry] = []
        func add(_ ref: GLTFDoc.TexRef?, _ semantics: [String: String], srgb: Bool) {
            guard let f = fileName(ref) else { return }
            entries.append(.init(fileName: f, isSRGB: srgb ? true : nil, semantics: semantics))
        }

        add(material.pbrMetallicRoughness?.baseColorTexture,         ["Albedo": "RGB"],                    srgb: true)
        add(material.normalTexture,                                  ["Normal": "RGB"],                    srgb: false)
        add(material.pbrMetallicRoughness?.metallicRoughnessTexture, ["Roughness": "G", "Metalness": "B"], srgb: false)
        add(material.occlusionTexture,                               ["Occlusion": "R"],                   srgb: false)
        add(material.emissiveTexture,                                ["Emissive": "RGB"],                  srgb: true)

        guard !entries.isEmpty else {
            throw Err.msg("glTF material referenced no usable textures")
        }

        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let json = try enc.encode(Manifest(textures: entries))

        let outURL = gltfURL.deletingLastPathComponent().appendingPathComponent("manifest.json")
        try json.write(to: outURL)
        return outURL
    }
}
