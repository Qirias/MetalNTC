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
    case contextCreationFailed
}

public func load_image(at url: URL, channels: Int) throws -> LoadedImage {
    guard channels == 1 || channels == 3 else {
        throw AssertError.unsupportedChannelCount(channels)
    }
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        throw AssertError.fileNotFound(url)
    }
    guard let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw AssertError.decodeFailed(url)
    }
    
    let width  = img.width
    let height = img.height

    let colorSpace: CGColorSpace
    let bitmapInfo: UInt32
    let bytesPerPixel: Int
    if channels == 3 {
        colorSpace    = CGColorSpaceCreateDeviceRGB()
        bitmapInfo    = CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        bytesPerPixel = 4
    } else {
        colorSpace    = CGColorSpaceCreateDeviceGray()
        bitmapInfo    = CGImageAlphaInfo.none.rawValue
        bytesPerPixel = 1
    }

    let bytesPerRow = width * bytesPerPixel
    var raw = [UInt8](repeating: 0, count: bytesPerRow * height)

    try raw.withUnsafeMutableBytes { buf in
        guard let ctx = CGContext(
            data: buf.baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw AssertError.contextCreationFailed
        }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    
    var pixels = [Float](repeating: 0, count: width * height * channels)
    let invScale: Float = 1.0 / 255.0
    
    if channels == 3 {
        for i in 0 ..< height * width {
            let src = i * 4
            let dst = i * 3
            pixels[dst    ] = Float(raw[src    ]) * invScale
            pixels[dst + 1] = Float(raw[src + 1]) * invScale
            pixels[dst + 2] = Float(raw[src + 2]) * invScale
        }
    } else {
        for i in 0 ..< height * width {
            pixels[i] = Float(raw[i]) * invScale
        }
    }
    
    return LoadedImage(pixels: pixels,
                       height: height,
                       width: width,
                       channels: channels)
}
