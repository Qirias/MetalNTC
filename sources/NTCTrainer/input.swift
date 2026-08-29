import Foundation
import AppKit
import NTCCore

enum TrainerError: Error, CustomStringConvertible {
    case msg(String)

    var description: String {
        switch self {
            case .msg(let text):
                return text
        }
    }
}

let DEFAULT_BROWSE_DIR = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()   // sources/NTCTrainer
    .deletingLastPathComponent()   // sources
    .appendingPathComponent("NTCRenderer/assets/models")

@MainActor
func pickInput() -> URL {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)

    let panel = NSOpenPanel()
    panel.title                   = "Select a model to compress"
    panel.message                 = "Choose a .gltf, a manifest.json, or a texture directory"
    panel.prompt                  = "Select Quality"
    panel.canChooseFiles          = true
    panel.canChooseDirectories    = true
    panel.allowsMultipleSelection = false
    if FileManager.default.fileExists(atPath: DEFAULT_BROWSE_DIR.path) {
        panel.directoryURL = DEFAULT_BROWSE_DIR
    }

    app.activate(ignoringOtherApps: true)
    guard panel.runModal() == .OK, let url = panel.url else {
        print("cancelled")
        exit(0)
    }
    return url
}

@MainActor
func pickQuality() -> Quality {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)

    let popup = NSPopUpButton(frame: NSRect(x: 0, y: 0, width: 220, height: 25))
    popup.addItems(withTitles: Quality.allCases.map(\.rawValue))
    popup.selectItem(at: Quality.allCases.firstIndex(of: .high)!)

    let alert = NSAlert()
    alert.messageText   = "Compression quality"
    alert.accessoryView = popup
    alert.addButton(withTitle: "Compress")
    alert.addButton(withTitle: "Cancel")

    app.activate(ignoringOtherApps: true)
    guard alert.runModal() == .alertFirstButtonReturn else {
        print("cancelled")
        exit(0)
    }
    return Quality.allCases[popup.indexOfSelectedItem]
}

func loadManifest(at url: URL) throws -> (manifest: Manifest, dir: URL) {
    var isDirectory: ObjCBool = false
    FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)

    if !isDirectory.boolValue {
        switch url.pathExtension.lowercased() {
            case "gltf":
                return (try ManifestGen.generate(fromGLTF: url), url.deletingLastPathComponent())
            case "json":
                return (try Manifest.load(from: url), url.deletingLastPathComponent())
            default:
                throw TrainerError.msg("\(url.lastPathComponent) is neither a .gltf nor a manifest.json")
        }
    }

    let manifestURL = url.appendingPathComponent("manifest.json")
    if FileManager.default.fileExists(atPath: manifestURL.path) {
        return (try Manifest.load(from: manifestURL), url)
    }

    let entries = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
    let gltfs   = entries.filter { $0.pathExtension.lowercased() == "gltf" }
    guard let gltfURL = gltfs.sorted(by: { $0.path < $1.path }).first else {
        throw TrainerError.msg("\(url.lastPathComponent) holds neither a manifest.json nor a .gltf")
    }
    return (try ManifestGen.generate(fromGLTF: gltfURL), url)
}
