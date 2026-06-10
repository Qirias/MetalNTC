// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NTCMetalKit",
    platforms: [
        .macOS("26.0")
    ],
    products: [
        .library(name: "NTCCore",      targets: ["NTCCore"]),
        .library(name: "NTCAssets",    targets: ["NTCAssets"]),
        .library(name: "NTCTrainer",   targets: ["NTCTrainer"]),
        .library(name: "NTCInference", targets: ["NTCInference"]),
        .executable(name: "NTCTrainerCLI",   targets: ["NTCTrainerCLI"]),
        .executable(name: "NTCDemoRenderer", targets: ["NTCDemoRenderer"]),
    ],
    targets: [
        .target(
            name: "NTCCore",
            path: "sources/NTCCore",
            resources: [.process("shaders")]
        ),
        .target(
            name: "NTCAssets",
            dependencies: ["NTCCore"],
            path: "sources/NTCAssets"
        ),
        .target(
            name: "NTCTrainer",
            dependencies: ["NTCCore"],
            path: "sources/NTCTrainer"
        ),
        .target(
            name: "NTCInference",
            dependencies: ["NTCCore"],
            path: "sources/NTCInference"
        ),
        .executableTarget(
            name: "NTCTrainerCLI",
            dependencies: ["NTCTrainer", "NTCAssets"],
            path: "sources/NTCTrainerCLI"
        ),
        .executableTarget(
            name: "NTCDemoRenderer",
            dependencies: ["NTCInference", "NTCAssets"],
            path: "sources/NTCDemoRenderer"
        ),
        .testTarget(
            name: "NTCTrainerTests",
            dependencies: ["NTCTrainer", "NTCAssets"],
            path: "tests/NTCTrainerTests"
        ),
        .testTarget(
            name: "NTCInferenceTests",
            dependencies: ["NTCInference", "NTCAssets"],
            path: "tests/NTCInferenceTests"
        ),
        .testTarget(
            name: "NTCCoreTests",
            dependencies: ["NTCCore"],
            path: "tests/NTCCoreTests"
        ),
    ]
)
