// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NTCMetalKit",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "NTCCore",   targets: ["NTCCore"]),
        .library(name: "NTCAssets", targets: ["NTCAssets"]),
        .executable(name: "NTCTrainerCLI", targets: ["NTCTrainerCLI"]),
    ],
    targets: [
        .target(
            name: "NTCCore",
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
            dependencies: ["NTCAssets", "NTCCore"],
            path: "sources/NTCTrainerCLI"
        ),
        .testTarget(
            name: "NTCCoreTests",
            dependencies: ["NTCCore"],
            path: "tests/NTCCoreTests",
            resources: [.process("shaders")]
        ),
    ]
)
