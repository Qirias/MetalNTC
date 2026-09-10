// swift-tools-version: 6.0

// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import PackageDescription
import Foundation

let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let rendererPresent = FileManager.default.fileExists(atPath: packageDir + "/sources/NTCRenderer/renderer.swift")

/// After flipping "debugEnabled" run ``swift build -c release --product NTCTrainer``
let debugEnabled = true
let trainerSwiftSettings: [SwiftSetting] = debugEnabled ? [.define("NTC_DEBUG")] : []

var products: [Product] = [
    .library(name: "NTCCore",   targets: ["NTCCore"]),
    .library(name: "NTCAssets", targets: ["NTCAssets"]),
    .executable(name: "NTCTrainer", targets: ["NTCTrainer"]),
]

var targets: [Target] = [
    .target(
        name: "NTCShared",
        path: "sources/NTCShared",
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
        name: "NTCTrainer",
        dependencies: ["NTCAssets", "NTCCore", "NTCShared"],
        path: "sources/NTCTrainer",
        swiftSettings: trainerSwiftSettings
    ),
]

if rendererPresent {
    products.append(.executable(name: "NTCRenderer", targets: ["NTCRenderer"]))
    // SwiftPM refuses to mix C++ and Swift sources in one target, so Apple's
    // math utilities get their own
    targets.append(.target(
        name: "AAPLMath",
        path: "sources/NTCRenderer/AAPLMath",
        publicHeadersPath: "include"
    ))
    targets.append(.executableTarget(
        name: "NTCRenderer",
        dependencies: ["NTCCore", "NTCShared", "AAPLMath"],
        path: "sources/NTCRenderer",
        // AAPLMath sits under this path but is its own target; without the
        // exclude SwiftPM sweeps its .cpp into this Swift target and refuses
        exclude: ["assets", "README.md", "AAPLMath"],
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
