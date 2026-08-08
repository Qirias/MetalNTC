import Foundation

/// Compression quality. The only knob is `gridScale`;  grid 0 is srcW/gridScale wide and each grid
/// halves from there, so the latent rate falls off as 1/gridScale^2:
///
///     bits/pixel = 4/3 * F_PER_GRID * BITS / gridScale^2
///
/// F_PER_GRID and BITS are the same for every setting on purpose, so a quality
/// change never touches the shader and every .ntc stays loadable by one
/// metallib. Rates for a 2048^2 9-channel set (BCn-in-VRAM baseline 19.6 MB):
///
///     setting   gridScale   bpp    bpp/channel   .ntc      vs BCn
///     veryHigh      3       9.48      1.05       4.75 MB    4.1x
///     high          4       5.33      0.59       2.68 MB    7.3x
///     medium        6       2.37      0.26       1.20 MB   16.3x
///     low           8       1.33      0.15       0.68 MB   28.7x
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
