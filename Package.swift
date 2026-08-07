// swift-tools-version: 6.0
import PackageDescription
import Foundation

// The renderer + its demo assets (models, hdr) are an optional git submodule at
// sources/NTCRenderer. Build the NTCRenderer target only when the submodule is
// actually checked out; a bare clone that skipped it is just the NTC core +
// trainer and still builds. This is why the heavy assets do not live in the main
// clone -- see the submodule at github.com/Qirias/MetalNTC-Renderer.
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let rendererPresent = FileManager.default.fileExists(atPath: packageDir + "/sources/NTCRenderer/renderer.swift")

var products: [Product] = [
    .library(name: "NTCCore",   targets: ["NTCCore"]),
    .library(name: "NTCAssets", targets: ["NTCAssets"]),
    .executable(name: "NTCTrainerCLI", targets: ["NTCTrainerCLI"]),
]

var targets: [Target] = [
    .target(
        name: "NTCShared",
        path: "sources/NTCShared",
        publicHeadersPath: "include"
    ),
    // SwiftPM refuses to mix C++ and Swift sources in one target
    .target(
        name: "AAPLMath",
        path: "sources/AAPLMath",
        publicHeadersPath: "include"
    ),
    .target(
        name: "NTCCore",
        dependencies: ["NTCShared"],
        path: "sources/NTCCore",
        resources: [.process("shaders")],
    ),
    .target(
        name: "NTCAssets",
        dependencies: ["NTCCore"],
        path: "sources/NTCAssets"
    ),
    .executableTarget(
        name: "NTCTrainerCLI",
        dependencies: ["NTCAssets", "NTCCore", "NTCShared"],
        path: "sources/NTCTrainerCLI"
    ),
    .testTarget(
        name: "NTCCoreTests",
        dependencies: ["NTCCore"],
        path: "tests/NTCCoreTests",
        resources: [.process("shaders")]
    ),
]

if rendererPresent {
    products.append(.executable(name: "NTCRenderer", targets: ["NTCRenderer"]))
    targets.append(.executableTarget(
        name: "NTCRenderer",
        dependencies: ["NTCCore", "NTCShared", "AAPLMath"],
        path: "sources/NTCRenderer",
        exclude: ["assets", "README.md"],
        resources: [.process("shaders")],
        swiftSettings: [.interoperabilityMode(.Cxx)]
    ))
}

let package = Package(
    name: "NTCMetalKit",
    platforms: [
        .macOS("26.0")
    ],
    products: products,
    targets: targets
)
