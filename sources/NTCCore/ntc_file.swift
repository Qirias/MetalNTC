// MetalNTC — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import Foundation
import NTCShared

public struct NTCSlotInfo {
    public var semantic:      String
    public var swizzle:       String
    public var channels:      Int
    public var channelOffset: Int
    public var sliceIndex:    Int
    public var isSRGB:        Bool

    public init(semantic: String, swizzle: String, channels: Int,
                channelOffset: Int, sliceIndex: Int, isSRGB: Bool) {
        self.semantic      = semantic
        self.swizzle       = swizzle
        self.channels      = channels
        self.channelOffset = channelOffset
        self.sliceIndex    = sliceIndex
        self.isSRGB        = isSRGB
    }
}

public struct NTCFile {
    public var header:           NTCHeader
    public var pyramidSizes:     [UInt32]
    public var neuralMipsForLod: [UInt32]
    public var slots:            [NTCSlotInfo]
    public var grid:             [UInt8]    // offset-binary codes, one per byte
    public var mlp:              [UInt16]   // raw binary16 bits

    public var gridTexels: Int { pyramidSizes.reduce(0) { $0 + Int($1) * Int($1) } }

    public static func mlpFloatCount(fPerGrid: Int, peWaves: Int,
                                     kHidden: Int, kOutMax: Int) -> Int {
        // must match F_IN in ntc_constants.h
        let fInRaw = 2 * fPerGrid + 4 * peWaves + 1
        let fIn = ((fInRaw + 15) / 16) * 16
        return fIn * kHidden + kHidden
             + kHidden * kHidden + kHidden
             + kHidden * kOutMax + kOutMax
    }
}

public enum NTCFileError: Error {
    case badSignature(UInt32)
    case badVersion(UInt32)
    case truncated
    case unsupportedMlpDType(UInt32)
    case unsupportedQuantBits(UInt32)
}

// MARK: Packing

/// Build an NTCFile from a pre-quantized grid.
/// - `gridBytes`: offset-binary ints, one byte per grid feature, valued in
///   [0, 2^quantBits). With `off = 1 << (quantBits-1)` the layout on disk is
///   `stored = clamp(round(f / q) + off, 0, 2^quantBits - 1)`. The renderer
///   repacks these ints into a texture2d_array for hardware filtering.
/// - `mlpFloats` points to `NTCFile.mlpFloatCount(...)` fp32 values.
public func packNTC(srcW: Int, srcH: Int, mipCount: Int,
                    kGrids: Int, fPerGrid: Int,
                    kHidden: Int, kOutMax: Int, kOut: Int,
                    peWaves: Int,
                    quantScale: Float, quantBits: Int,
                    pyramidSizes: [Int],
                    neuralMipsForLod: [UInt32],
                    slots: [NTCSlotInfo],
                    gridBytes: [UInt8],
                    mlpFloats: UnsafePointer<Float>) -> NTCFile {
    precondition(pyramidSizes.count == kGrids, "pyramidSizes count \(pyramidSizes.count) != kGrids \(kGrids)")
    precondition(neuralMipsForLod.count == mipCount, "neuralMipsForLod count \(neuralMipsForLod.count) != mipCount \(mipCount)")
    precondition(slots.count <= kOutMax, "slot count \(slots.count) exceeds kOutMax \(kOutMax)")

    let texels = pyramidSizes.reduce(0) { $0 + $1 * $1 }
    let expectedGridBytes = texels * fPerGrid
    precondition(gridBytes.count == expectedGridBytes, "gridBytes count \(gridBytes.count) != expected \(expectedGridBytes)")

    var header = NTCHeader()
    header.signature  = NTC_SIGNATURE
    header.version    = NTC_VERSION
    header.srcW       = UInt32(srcW)
    header.srcH       = UInt32(srcH)
    header.mipCount   = UInt32(mipCount)
    header.kGrids     = UInt32(kGrids)
    header.fPerGrid   = UInt32(fPerGrid)
    header.kHidden    = UInt32(kHidden)
    header.kOutMax    = UInt32(kOutMax)
    header.kOut       = UInt32(kOut)
    header.nSlots     = UInt32(slots.count)
    header.peWaves    = UInt32(peWaves)
    header.quantScale = quantScale
    header.quantBits  = UInt32(quantBits)
    header.mlpDType   = NTC_MLP_DTYPE_FP16
    header.flags      = 0

    // MLP: fp32 -> fp16 raw bits
    let mlpCount = NTCFile.mlpFloatCount(fPerGrid: fPerGrid, peWaves: peWaves,
                                         kHidden: kHidden, kOutMax: kOutMax)
    var mlp = [UInt16](repeating: 0, count: mlpCount)
    for i in 0..<mlpCount {
        mlp[i] = Float16(mlpFloats[i]).bitPattern
    }

    return NTCFile(header:           header,
                   pyramidSizes:     pyramidSizes.map { UInt32($0) },
                   neuralMipsForLod: neuralMipsForLod,
                   slots:            slots,
                   grid:             gridBytes,
                   mlp:              mlp)
}

// MARK: Write

/// writes the file and returns how many bytes
@discardableResult
public func writeNTC(_ file: NTCFile, to url: URL) throws -> Int {
    var data = Data()

    // header
    var header = file.header
    withUnsafeBytes(of: &header) {
        data.append(contentsOf: $0)
    }

    // pyramidSizes
    appendArray(file.pyramidSizes, to: &data)

    // neuralMipsForLod
    appendArray(file.neuralMipsForLod, to: &data)

    // slots
    for s in file.slots {
        var rec = NTCSlot()
        writeFixedString(s.semantic, into: &rec.semantic)
        writeFixedString(s.swizzle,  into: &rec.swizzle)
        rec.channels      = UInt8(s.channels)
        rec.channelOffset = UInt8(s.channelOffset)
        rec.sliceIndex    = UInt8(s.sliceIndex)
        rec.isSRGB        = s.isSRGB ? 1 : 0
        rec.reserved      = 0
        withUnsafeBytes(of: &rec) {
            data.append(contentsOf: $0)
        }
    }

    // grid: two 4-bit codes per byte. In-memory `grid` is one code per byte
    data.append(contentsOf: packGrid(file.grid))

    // mlp
    appendArray(file.mlp, to: &data)

    try data.write(to: url, options: .atomic)
    return data.count
}

// MARK: Read

public func readNTC(from url: URL) throws -> NTCFile {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    return try readNTC(from: data)
}

public func readNTC(from data: Data) throws -> NTCFile {
    var cur = 0
    let header: NTCHeader = try readValue(from: data, at: &cur)
    guard header.signature == NTC_SIGNATURE else {
        throw NTCFileError.badSignature(header.signature)
    }
    guard header.version == NTC_VERSION else {
        throw NTCFileError.badVersion(header.version)
    }
    guard header.mlpDType == NTC_MLP_DTYPE_FP16 else {
        throw NTCFileError.unsupportedMlpDType(header.mlpDType)
    }
    
    guard header.quantBits == 4 else {
        throw NTCFileError.unsupportedQuantBits(header.quantBits)
    }

    let pyramidSizes = try readArray(from: data, at: &cur,
                                     count: Int(header.kGrids), as: UInt32.self)
    let neuralMipsForLod = try readArray(from: data, at: &cur,
                                         count: Int(header.mipCount), as: UInt32.self)

    var slots: [NTCSlotInfo] = []
    slots.reserveCapacity(Int(header.nSlots))
    for _ in 0..<Int(header.nSlots) {
        let rec: NTCSlot = try readValue(from: data, at: &cur)
        slots.append(NTCSlotInfo(
            semantic:      readFixedString(rec.semantic),
            swizzle:       readFixedString(rec.swizzle),
            channels:      Int(rec.channels),
            channelOffset: Int(rec.channelOffset),
            sliceIndex:    Int(rec.sliceIndex),
            isSRGB:        rec.isSRGB != 0
        ))
    }

    let texels    = pyramidSizes.reduce(0) { $0 + Int($1) * Int($1) }
    let gridCount = texels * Int(header.fPerGrid)   // number of codes
    let diskBytes = (gridCount + 1) / 2             // two codes per byte

    let packed = try readArray(from: data, at: &cur, count: diskBytes, as: UInt8.self)
    let grid   = unpackGrid(packed, count: gridCount)

    let mlpCount = NTCFile.mlpFloatCount(fPerGrid: Int(header.fPerGrid),
                                         peWaves:  Int(header.peWaves),
                                         kHidden:  Int(header.kHidden),
                                         kOutMax:  Int(header.kOutMax))
    let mlp = try readArray(from: data, at: &cur, count: mlpCount, as: UInt16.self)

    return NTCFile(header:           header,
                   pyramidSizes:     pyramidSizes,
                   neuralMipsForLod: neuralMipsForLod,
                   slots:            slots,
                   grid:             grid,
                   mlp:              mlp)
}

// MARK: Helpers

private func readValue<T>(from data: Data, at cursor: inout Int) throws -> T {
    let size = MemoryLayout<T>.size
    guard cursor + size <= data.count else { throw NTCFileError.truncated }
    let v: T = data.withUnsafeBytes { raw in
        raw.baseAddress!.advanced(by: cursor).loadUnaligned(as: T.self)
    }
    cursor += size
    return v
}

private func readArray<T>(from data: Data, at cursor: inout Int,
                          count: Int, as _: T.Type) throws -> [T] {
    let size = MemoryLayout<T>.size * count
    guard cursor + size <= data.count else { throw NTCFileError.truncated }
    let arr: [T] = data.withUnsafeBytes { raw -> [T] in
        let p = raw.baseAddress!.advanced(by: cursor)
        return Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: T.self),
                                         count: count))
    }
    cursor += size
    return arr
}

/// Pack grid codes for 4-bit storage: two codes per byte, codes[2k] in the low
/// 4 bits [0,4) and codes[2k+1] in the high 4 bits [4,8).
/// A trailing odd code (can't happen for even fPerGrid) leaves the high 4 bits 0.
func packGrid(_ codes: [UInt8]) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: (codes.count + 1) / 2)
    var i = 0
    while i < codes.count {
        let lo = codes[i] & 0x0F
        var hi: UInt8 = 0
        if i + 1 < codes.count {
            hi = codes[i + 1] & 0x0F
        }
        out[i / 2] = lo | (hi << 4)
        i += 2
    }
    return out
}

/// Inverse of `packGrid`: expand `count` codes back to one byte each.
func unpackGrid(_ packed: [UInt8], count: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: count)
    for k in 0..<count {
        let byte = packed[k / 2]
        if k % 2 == 0 {
            out[k] = byte & 0x0F
        } else {
            out[k] = byte >> 4
        }
    }
    return out
}

private func appendArray<T>(_ arr: [T], to data: inout Data) {
    arr.withUnsafeBufferPointer { buf in
        let raw = UnsafeRawBufferPointer(buf)
        data.append(raw.baseAddress!.assumingMemoryBound(to: UInt8.self),
                    count: raw.count)
    }
}

private func writeFixedString<T>(_ s: String, into field: inout T) {
    let bytes = Array(s.utf8)
    withUnsafeMutableBytes(of: &field) { raw in
        for i in 0..<raw.count {
            raw[i] = i < bytes.count ? bytes[i] : 0
        }
    }
}

private func readFixedString<T>(_ field: T) -> String {
    return withUnsafeBytes(of: field) { raw in
        var n = 0
        while n < raw.count, raw[n] != 0 {
            n += 1
        }
        let bytes = Array(raw.prefix(n))
        return String(decoding: bytes, as: UTF8.self)
    }
}
