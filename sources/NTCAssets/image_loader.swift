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

import Foundation
import CoreGraphics
import ImageIO

public struct LoadedImage {
    public let pixels: [Float]
    public let height: Int
    public let width: Int
    public let channels: Int

    public init(pixels: [Float], height: Int, width: Int, channels: Int) {
        self.pixels = pixels
        self.height = height
        self.width = width
        self.channels = channels
    }
}

enum AssertError : Error {
    case fileNotFound(URL)
    case decodeFailed(URL)
    case encodeFailed(URL)
    case unsupportedChannelCount(Int)
    case badSwizzle(String)
    case contextCreationFailed
}

/// Decoded RGBA8, normalized to [0,1]. Always 4 channels regardless of how many
/// the file actually has -- CoreGraphics expands grayscale to R=G=B for us, so a
/// swizzle of "R" reads correctly off both a gray AO map and an RGB packed map.
public struct RGBAImage {
    public let pixels: [Float]   // width * height * 4
    public let width:  Int
    public let height: Int
}

public func decode_rgba(at url: URL) throws -> RGBAImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw AssertError.fileNotFound(url)
    }
    guard let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw AssertError.decodeFailed(url)
    }

    let width  = img.width
    let height = img.height

    let bytesPerRow = width * 4
    var raw = [UInt8](repeating: 0, count: bytesPerRow * height)

    try raw.withUnsafeMutableBytes { buf in
        guard let ctx = CGContext(
            data: buf.baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw AssertError.contextCreationFailed
        }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
    }

    var pixels = [Float](repeating: 0, count: width * height * 4)
    let invScale: Float = 1.0 / 255.0
    for i in 0 ..< (width * height * 4) {
        pixels[i] = Float(raw[i]) * invScale
    }

    return RGBAImage(pixels: pixels, width: width, height: height)
}

/// Maps a swizzle character to its index in an RGBA pixel.
func swizzle_index(_ c: Character) throws -> Int {
    switch c {
    case "R", "r": return 0
    case "G", "g": return 1
    case "B", "b": return 2
    case "A", "a": return 3
    default: throw AssertError.badSwizzle(String(c))
    }
}

/// Pulls the named channels out of an already-decoded RGBA image. Lets one file
/// on disk feed several semantics -- glTF packs roughness in G and metalness in
/// B of a single metalRoughness texture, so both come from one decode.
public func extract_swizzle(_ rgba: RGBAImage, swizzle: String) throws -> LoadedImage {
    let channels = swizzle.count
    guard channels == 1 || channels == 3 else {
        throw AssertError.unsupportedChannelCount(channels)
    }
    let idx = try swizzle.map { try swizzle_index($0) }

    let n = rgba.width * rgba.height
    var pixels = [Float](repeating: 0, count: n * channels)
    for i in 0 ..< n {
        let src = i * 4
        let dst = i * channels
        for c in 0 ..< channels {
            pixels[dst + c] = rgba.pixels[src + idx[c]]
        }
    }

    return LoadedImage(pixels: pixels,
                       height: rgba.height,
                       width: rgba.width,
                       channels: channels)
}

public func load_image(at url: URL, swizzle: String) throws -> LoadedImage {
    return try extract_swizzle(try decode_rgba(at: url), swizzle: swizzle)
}
