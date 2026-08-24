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
            let name: String?
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

    /// Reads every material in the glTF, writes ONE `manifest.json` beside the
    /// .gltf (a list of models, one per material), and returns the decoded
    /// Manifest. One .ntc is trained per model.
    @discardableResult
    static func generate(fromGLTF gltfURL: URL) throws -> Manifest {
        let data = try Data(contentsOf: gltfURL)
        let doc  = try JSONDecoder().decode(GLTFDoc.self, from: data)

        guard let materials = doc.materials, !materials.isEmpty else {
            throw TrainerError.msg("glTF \(gltfURL.lastPathComponent) declares no materials")
        }

        let textures = doc.textures ?? []
        let images   = doc.images   ?? []

        // texture index -> image file name; nil means the material does not
        // declare this texture at all; an out-of-range index is a corrupt glTF
        // and traps rather than silently dropping the texture
        func fileName(_ ref: GLTFDoc.TexRef?) -> String? {
            guard let ref else {
                return nil
            }
            let uri = images[textures[ref.index].source].uri
            return uri.removingPercentEncoding ?? uri
        }

        var models: [Manifest.Model] = []
        for (i, material) in materials.enumerated() {
            var entries: [Manifest.Entry] = []
            func add(_ ref: GLTFDoc.TexRef?, _ semantics: [String: String], srgb: Bool) {
                guard let file = fileName(ref) else {
                    return
                }
                entries.append(Manifest.Entry(fileName: file,
                                              isSRGB:   srgb ? true : nil,
                                              semantics: semantics))
            }

            add(material.pbrMetallicRoughness?.baseColorTexture,         ["Albedo": "RGB"],                    srgb: true)
            add(material.normalTexture,                                  ["Normal": "RGB"],                    srgb: false)
            add(material.pbrMetallicRoughness?.metallicRoughnessTexture, ["Roughness": "G", "Metalness": "B"], srgb: false)
            add(material.occlusionTexture,                               ["Occlusion": "R"],                   srgb: false)
            add(material.emissiveTexture,                                ["Emissive": "RGB"],                  srgb: true)

            guard !entries.isEmpty else {
                FileHandle.standardError.write(Data(
                    "warning: material \(i) (\(material.name ?? "unnamed")) has no usable textures; skipped\n".utf8))
                continue
            }
            models.append(Manifest.Model(name: safeName(material.name, index: i),
                                         textures: entries))
        }

        guard !models.isEmpty else {
            throw TrainerError.msg("glTF \(gltfURL.lastPathComponent) has no material with usable textures")
        }

        let manifest = Manifest(models: models)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let json = try enc.encode(manifest)

        let outURL = gltfURL.deletingLastPathComponent().appendingPathComponent("manifest.json")
        try json.write(to: outURL)
        return manifest
    }

    /// Make a material name safe to use as a filename; fall back to material_<i>.
    private static func safeName(_ raw: String?, index: Int) -> String {
        guard let raw, !raw.isEmpty else {
            return "material_\(index)"
        }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let cleaned = String(raw.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "_"
        })
        if cleaned.isEmpty {
            return "material_\(index)"
        }
        return cleaned
    }
}
