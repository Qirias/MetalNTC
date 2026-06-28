import Foundation
import CoreGraphics
import ImageIO

public func save_image(_ img: LoadedImage, to url: URL) throws {
    guard img.channels == 1 || img.channels == 3 else {
        throw AssertError.unsupportedChannelCount(img.channels)
    }

    let width    = img.width
    let height   = img.height
    let channels = img.channels

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
    if channels == 3 {
        for i in 0 ..< height * width {
            let src = i * 3
            let dst = i * 4
            for c in 0 ..< 3 {
                let scaled  = img.pixels[src + c] * 255.0 + 0.5
                let clamped = max(0.0, min(255.0, scaled))
                raw[dst + c] = UInt8(clamped)
            }
        }
    } else {
        for i in 0 ..< height * width {
            let scaled  = img.pixels[i] * 255.0 + 0.5
            let clamped = max(0.0, min(255.0, scaled))
            raw[i] = UInt8(clamped)
        }
    }

    let outImg: CGImage = try raw.withUnsafeMutableBytes { buf in
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
        guard let cg = ctx.makeImage() else {
            throw AssertError.encodeFailed(url)
        }
        return cg
    }

    guard let dst = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
        throw AssertError.encodeFailed(url)
    }
    
    CGImageDestinationAddImage(dst, outImg, nil)
    guard CGImageDestinationFinalize(dst) else {
        throw AssertError.encodeFailed(url)
    }
}
