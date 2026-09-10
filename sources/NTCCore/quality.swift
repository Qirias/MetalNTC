// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import Foundation

public enum Quality: String, CaseIterable, Sendable {
    case low, medium, high, veryHigh

    public var gridScale: Int {
        switch self {
            case .veryHigh: return 3
            case .high:     return 4
            case .medium:   return 6
            case .low:      return 8
        }
    }

    public func ntcFileName(base: String) -> String {
        "\(base)_\(rawValue).ntc"
    }
}
