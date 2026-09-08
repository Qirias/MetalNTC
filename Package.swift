// swift-tools-version: 6.0

// MIT License
//
// Copyright (c) 2026 Kyriakos Gavras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
