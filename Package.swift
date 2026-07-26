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
        .executable(name: "NTCRenderer",   targets: ["NTCRenderer"]),
    ],
    targets: [
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
        .executableTarget(
            name: "NTCRenderer",
            dependencies: ["NTCCore", "NTCShared", "AAPLMath"],
            path: "sources/NTCRenderer",
            resources: [.process("shaders")],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "NTCCoreTests",
            dependencies: ["NTCCore"],
            path: "tests/NTCCoreTests",
            resources: [.process("shaders")]
        ),
    ]
)
