// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import Foundation

/// SwiftPM synthesises `Bundle.module` on every target that declares
/// `resources:`, but that symbol is internal-scope. Modules that
/// `import NTCCore` cannot reach it directly. This namespace lifts it
/// over the module boundary so the trainer / inference executables can
/// load shaders from NTCCore's `default.metallib`:
///
///     let ctx = try MetalContext(bundle: NTCCoreResources.bundle)
///
public enum NTCCoreResources {
    public static let bundle: Bundle = .module
}
