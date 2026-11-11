import simd

struct ArchivePrimitive: Codable {
    struct Fade: Codable {
        var distance: Float
        var contrast: Float

        init(_ fade: SoftParticleDepth.Fade) {
            distance = fade.distance
            contrast = fade.contrast
        }

        func runtime() throws -> SoftParticleDepth.Fade {
            try SoftParticleDepth.Fade(distance: distance, contrast: contrast)
        }
    }

    enum Blend: String, Codable {
        case opaque, alpha, additive

        init(_ blend: PrimitiveBlending) {
            self = switch blend {
            case .opaque: .opaque
            case .alpha: .alpha
            case .additive: .additive
            }
        }

        var runtime: PrimitiveBlending {
            switch self {
            case .opaque: .opaque
            case .alpha: .alpha
            case .additive: .additive
            }
        }
    }

    enum Extent: Codable {
        case pixels(Float)
        case world(Float)

        init(_ size: PointsMaterial.Size) {
            self = switch size {
            case .pixels(let value): .pixels(value)
            case .world(let value): .world(value)
            }
        }

        var pointSize: PointsMaterial.Size {
            switch self {
            case .pixels(let value): .pixels(value)
            case .world(let value): .world(value)
            }
        }

        var lineWidth: LineMaterial.Width {
            switch self {
            case .pixels(let value): .pixels(value)
            case .world(let value): .world(value)
            }
        }
    }

    struct Cloud: Codable {
        var count: Int
        var positions: Int
        var colors: Int?
        var sizes: Int?
        var extent: Extent
        var circular: Bool
        var map: Int?
        var alphaTest: Float
    }

    struct Line: Codable {
        enum Topology: String, Codable {
            case strip, loop, segments

            init(_ topology: LineGeometry.Topology) {
                self = switch topology {
                case .strip: .strip
                case .loop: .loop
                case .segments: .segments
                }
            }

            var runtime: LineGeometry.Topology {
                switch self {
                case .strip: .strip
                case .loop: .loop
                case .segments: .segments
                }
            }
        }

        enum Join: Codable {
            case miter(limit: Float)
            case round
            case bevel

            init(_ join: LineMaterial.Join) {
                self = switch join {
                case .miter(let limit): .miter(limit: limit)
                case .round: .round
                case .bevel: .bevel
                }
            }

            var runtime: LineMaterial.Join {
                switch self {
                case .miter(let limit): .miter(limit: limit)
                case .round: .round
                case .bevel: .bevel
                }
            }
        }

        enum Cap: String, Codable {
            case butt, round, square

            init(_ cap: LineMaterial.Cap) {
                self = switch cap {
                case .butt: .butt
                case .round: .round
                case .square: .square
                }
            }

            var runtime: LineMaterial.Cap {
                switch self {
                case .butt: .butt
                case .round: .round
                case .square: .square
                }
            }
        }

        struct Dash: Codable {
            var length: Float
            var gap: Float
            var offset: Float
        }

        var count: Int
        var positions: Int
        var colors: Int?
        var topology: Topology
        /// Nil encodes a Metal hairline.
        var extent: Extent?
        var join: Join
        var cap: Cap
        var dash: Dash?
    }

    struct Sprites: Codable {
        static let stride = 16

        var count: Int
        var records: Int
        var map: Int?
        var alphaTest: Float
        var sizeAttenuation: Bool

        static func pack(_ sprites: [Sprite]) -> [Float] {
            var values: [Float] = []
            values.reserveCapacity(sprites.count * stride)
            for sprite in sprites {
                values.append(contentsOf: [
                    sprite.position.x, sprite.position.y, sprite.position.z, sprite.size.x, sprite.size.y,
                    sprite.rotation, sprite.center.x, sprite.center.y,
                    sprite.color.x, sprite.color.y, sprite.color.z, sprite.color.w,
                    sprite.textureRegion.x, sprite.textureRegion.y, sprite.textureRegion.z, sprite.textureRegion.w,
                ])
            }
            return values
        }

        func unpack(using reader: ArchiveBinaryReader) throws -> [Sprite] {
            guard (1...SpriteRenderer.maximumSpriteCount).contains(count) else { throw PrimitiveRendererError.spriteCount }
            let length = try reader.product(count, Self.stride, view: records)
            let values = try reader.array(records, of: Float.self, count: length)
            return Swift.stride(from: 0, to: values.count, by: Self.stride).map { index in
                Sprite(
                    position: SIMD3(values[index], values[index + 1], values[index + 2]),
                    size: SIMD2(values[index + 3], values[index + 4]), rotation: values[index + 5],
                    center: SIMD2(values[index + 6], values[index + 7]),
                    color: SIMD4(values[index + 8], values[index + 9], values[index + 10], values[index + 11]),
                    textureRegion: SIMD4(values[index + 12], values[index + 13], values[index + 14], values[index + 15])
                )
            }
        }
    }

    enum Content: Codable {
        case points(Cloud)
        case lines(Line)
        case sprites(Sprites)
    }

    var colorPixelFormat: UInt
    var sampleCount: Int
    var color: SIMD4<Float>
    var blending: Blend
    var depthTest: Bool
    var drawRange: ArchiveDrawRange?
    var content: Content
    var softParticleFade: Fade? = nil
}
