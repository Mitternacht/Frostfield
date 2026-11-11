import Foundation
@testable import FrostfieldSwiftUI
import Metal
import QuartzCore
import Testing

@MainActor
struct DrawableResolutionTests {
    typealias Resolution = FrostfieldSceneView.DrawableResolution

    @Test("Fixed and display resolution apply to one Metal layer without resizing it")
    func drawableResolution() throws {
        let layer = CAMetalLayer()
        let logical = CGSize(width: 80.5, height: 40.25)
        layer.frame = CGRect(origin: .zero, size: logical)
        let fixed = try Resolution(pixelsPerPoint: 0.75)
        let nativePixels = CGSize(width: 161, height: 80.5)

        let scaled = try fixed.apply(
            to: layer, logicalSize: logical, backingSize: nativePixels, displayScale: 2
        ).get()
        #expect(scaled == CGSize(width: 61, height: 31))
        #expect(layer.drawableSize == scaled)
        #expect(layer.bounds.size == logical)
        #expect(layer.contentsScale == 2)

        let sameScale = try fixed.apply(
            to: layer, logicalSize: logical,
            backingSize: CGSize(width: 241.5, height: 120.75), displayScale: 3
        ).get()
        #expect(sameScale == scaled)
        #expect(layer.drawableSize == scaled)
        #expect(layer.contentsScale == 3)

        let native = try Resolution.display.apply(
            to: layer, logicalSize: logical, backingSize: nativePixels, displayScale: 2
        ).get()
        #expect(native == CGSize(width: 161, height: 81))
        #expect(layer.drawableSize == native)
        #expect(layer.bounds.size == logical)
    }

    @Test("Invalid ratios and oversized targets fail before changing the layer")
    func invalidAndEmptySizes() throws {
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: 0) }
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: -1) }
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: .infinity) }
        #expect(throws: Resolution.Failure.pixelRatio) { try Resolution(pixelsPerPoint: .nan) }

        let layer = CAMetalLayer()
        layer.contentsScale = 2
        layer.drawableSize = CGSize(width: 64, height: 32)
        let large = try Resolution(pixelsPerPoint: 8)
        #expect(large.apply(
            to: layer, logicalSize: CGSize(width: 2_049, height: 300), backingSize: .zero, displayScale: 3
        ) == .failure(.size))
        #expect(try large.apply(
            to: layer, logicalSize: .zero, backingSize: .zero, displayScale: 3
        ).get() == nil)
        #expect(layer.drawableSize == CGSize(width: 64, height: 32))
        #expect(layer.contentsScale == 2)
    }

    @Test("An acquired Metal texture has the requested pixel size",
          .enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
    func acquiredTextureDimensions() throws {
        let layer = CAMetalLayer()
        layer.device = try #require(MTLCreateSystemDefaultDevice())
        let logical = CGSize(width: 96, height: 48)
        layer.frame = CGRect(origin: .zero, size: logical)
        let resolution = try Resolution(pixelsPerPoint: 0.5)
        _ = try resolution.apply(
            to: layer, logicalSize: logical, backingSize: CGSize(width: 192, height: 96), displayScale: 2
        ).get()
        let texture = try #require(layer.nextDrawable()?.texture)
        #expect(texture.width == 48)
        #expect(texture.height == 24)
    }
}
