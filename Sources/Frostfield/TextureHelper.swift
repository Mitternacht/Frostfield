import Metal
import simd

extension DebugHelpers {
    /// A live texture preview in the scene, with the mesh's normal transform and depth rules.
    /// 2D textures use an XY plane. Cubes use a box with native Metal face orientation.
    /// An array or volume uses one translucent child plane per layer at the selected mip,
    /// with its first and last layers at `-size.z / 2` and `size.z / 2` in local space.
    /// A single layer is centered. The children share one renderer.
    ///
    /// The source is sampled by the scene encode. End writes to the texture before
    /// that encode and keep it separate from the scene's attachments. The helper
    /// reads RGB in top-left image order, ignoring source alpha. It casts no shadows.
    public func texture(
        _ source: any MTLTexture, size: SIMD3<Float> = SIMD3(repeating: 1),
        mipLevel: Int = 0, sampleCount: Int = 1, transform: SceneTransform = SceneTransform()
    ) throws -> SceneNode {
        guard size.x.isNormal, size.x > 0, size.y.isNormal, size.y > 0, size.z.isNormal, size.z > 0 else {
            throw PrimitiveGeometryError.invalidHelperParameter("texture helper dimensions must be positive and normal")
        }
        let shape = try TextureDebugPass.describeColor(source, device: device, mip: mipLevel)
        let isStack = shape.kind == .colorArray || shape.kind == .color3D
        let geometry: MeshGeometry
        if shape.kind == .colorCube {
            let box = MeshGeometry.cube()
            let vertices = box.vertices.enumerated().map { index, vertex in
                // The box's four vertices per face are in Metal's cube face order.
                MeshVertex(
                    position: vertex.position * size, normal: vertex.normal,
                    textureCoordinate: vertex.textureCoordinate,
                    secondaryTextureCoordinate: SIMD2(Float(index / 4), 0)
                )
            }
            geometry = try MeshGeometry(vertices: vertices, indices: box.indices)
        } else {
            geometry = try .plane(width: size.x, height: size.y)
        }
        let layered = isStack && shape.sliceCount > 1
        var renderState: MeshRenderState = layered ? .transparent : .opaque
        renderState.side = .double
        let opacity: Float = layered ? max(1 / Float(shape.sliceCount), 0.25) : 1
        let binding = TextureHelperBinding(
            texture: source, fragment: "frostfield_texture_helper_" + shape.kind.rawValue,
            options: SIMD4(UInt32(mipLevel), UInt32(shape.sliceCount), opacity.bitPattern, 0)
        )
        let renderer = try MeshRenderer(
            device: device, colorPixelFormat: colorPixelFormat, geometry: geometry, material: .unlit,
            depthMode: depthMode, supportsInstancing: false, renderState: renderState,
            sampleCount: sampleCount, indirectCommands: false, textureHelper: binding
        )
        guard isStack else { return SceneNode(transform: transform, mesh: renderer, castsShadow: false) }

        let slices = (0..<shape.sliceCount).map { index in
            let z = layered ? size.z * (Float(index) / Float(shape.sliceCount - 1) - 0.5) : 0
            return SceneNode(
                transform: SceneTransform(translation: SIMD3(0, 0, z)), mesh: renderer,
                materialParameters: SIMD4(Float(index), 0, 0, 0), castsShadow: false
            )
        }
        return SceneNode(transform: transform, children: slices, castsShadow: false)
    }
}

struct TextureHelperBinding {
    let texture: any MTLTexture
    let fragment: String
    let options: SIMD4<UInt32>
}
