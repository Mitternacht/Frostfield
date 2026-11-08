import Foundation
import Metal
internal import FrostfieldShaderTypes

/// Indexed geometry uploaded to GPU buffers by ``MeshRenderer/makeMesh(_:)``.
///
/// The position, normal, and index buffers are written once, when the mesh
/// is created, and only read by the GPU afterwards. Drawing the mesh with a
/// new transform or camera reuses them. In-flight frames keep the buffers
/// alive, so a mesh may be released while frames that draw it are still on
/// the GPU.
public final class Mesh {
    /// The number of vertices.
    public let vertexCount: Int
    /// The number of indices, three per triangle.
    public let indexCount: Int
    /// The device that owns the mesh's buffers.
    public let device: any MTLDevice

    let positionBuffer: any MTLBuffer
    let normalBuffer: any MTLBuffer
    let indexBuffer: any MTLBuffer

    init(
        geometry: IndexedGeometry,
        device: any MTLDevice,
        positionBuffer: any MTLBuffer,
        normalBuffer: any MTLBuffer,
        indexBuffer: any MTLBuffer
    ) {
        self.vertexCount = geometry.vertexCount
        self.indexCount = geometry.indices.count
        self.device = device
        self.positionBuffer = positionBuffer
        self.normalBuffer = normalBuffer
        self.indexBuffer = indexBuffer
    }
}

/// Renders one indexed ``Mesh`` per frame, placed by a ``Transform``,
/// viewed through a ``Camera``, and shaded by a ``Material``, with depth
/// testing and back-face culling.
///
/// Triangles whose indices run counter-clockwise on screen are front faces;
/// back faces are culled, and the nearest surface wins the depth test
/// (`less`, depth cleared to 1). Every material, built-in or app-supplied,
/// shares this vertex stage, depth test, culling, and frame loop; only the
/// fragment function and its parameter block differ (`docs/materials.md`).
/// ``drawFrame(mesh:transform:camera:target:)`` uses the built-in normal
/// diagnostic: each fragment shows its model-space normal as a color,
/// `normal * 0.5 + 0.5`.
///
/// The initializer creates the normal diagnostic's pipeline, the
/// depth-stencil state, and one uniform buffer per frame slot.
/// ``makeMesh(_:)`` creates a mesh's buffers, and ``makeUnlitMaterial(color:)``
/// and ``makeMaterial(fragmentFunction:library:parameters:)`` create a
/// material's pipeline the first time its fragment function is used. Drawing
/// creates nothing: it writes `projection * view * model` into the frame
/// slot's uniform buffer and copies the material's parameters into the frame's
/// commands, so ``resourceStatistics`` stays constant while transforms,
/// cameras, and material parameters change. Frame slots, skips, and completion
/// reports behave as in ``TriangleRenderer``.
///
/// Every target's pass needs a depth attachment of ``depthPixelFormat``. For
/// an `MTKView`, set its `depthStencilPixelFormat` to the same format.
///
/// Use one renderer from one thread or actor at a time. Completion reports
/// arrive on a Metal-owned thread.
public final class MeshRenderer {
    static let vertexFunctionName = "ff_mesh_vertex"
    static let normalFragmentFunctionName = "ff_mesh_normal_fragment"
    static let unlitFragmentFunctionName = "ff_unlit_fragment"

    /// The device every resource of this renderer belongs to.
    public let device: any MTLDevice
    /// The color format the pipelines render; targets must match it.
    public let colorPixelFormat: MTLPixelFormat
    /// The depth format the pipelines test against; targets must match it.
    public let depthPixelFormat: MTLPixelFormat
    /// The maximum number of frames that may be in flight at once.
    public var maxFramesInFlight: Int { submitter.slots.capacity }
    /// Counters for submitted, completed, and skipped frames.
    public var statistics: FrameStatistics { submitter.slots.statistics }
    /// Counters for the pipelines, depth-stencil states, and buffers this
    /// renderer created, including every mesh's buffers and every material's
    /// pipeline.
    public private(set) var resourceStatistics: ResourceStatistics

    private let submitter: FrameSubmitter
    private let depthState: any MTLDepthStencilState
    /// Frostfield's packaged library, loaded once so the built-in materials'
    /// cache keys stay stable.
    private let builtInLibrary: any MTLLibrary
    private let pipelineCache: MaterialPipelineCache
    private let normalMaterial: Material<NoMaterialParameters>
    /// One uniform buffer per frame slot, indexed by slot.
    var uniformBuffers: [any MTLBuffer] { submitter.uniformBuffers }

    /// Creates the queue, the normal diagnostic's pipeline, the depth-stencil
    /// state, and the per-frame buffers.
    ///
    /// - Parameters:
    ///   - device: The device to render with; `nil` throws
    ///     ``RendererError/noMetalDevice``.
    ///   - colorPixelFormat: The pixel format of every target's first color
    ///     attachment.
    ///   - depthPixelFormat: The pixel format of every target's depth
    ///     attachment.
    ///   - maxFramesInFlight: Frame slots, from 1 to 3.
    ///   - onFrameCompleted: Called on a Metal-owned thread after each
    ///     submitted frame completes and its slot is free again.
    public init(
        device: (any MTLDevice)?,
        colorPixelFormat: MTLPixelFormat,
        depthPixelFormat: MTLPixelFormat = .depth32Float,
        maxFramesInFlight: Int = 2,
        onFrameCompleted: @escaping @Sendable (FrameCompletion) -> Void = { _ in }
    ) throws(RendererError) {
        guard let device else { throw .noMetalDevice }
        var resources = ResourceStatistics()
        let submitter = try FrameSubmitter(
            device: device,
            label: "Frostfield.MeshRenderer",
            colorPixelFormat: colorPixelFormat,
            maxFramesInFlight: maxFramesInFlight,
            uniformBufferLength: MemoryLayout<FFTransformUniforms>.stride,
            resources: &resources,
            onFrameCompleted: onFrameCompleted
        )

        let library = try TriangleRenderer.makeShaderLibrary(device: device)
        guard let vertexFunction = library.makeFunction(name: Self.vertexFunctionName) else {
            throw .shaderFunctionMissing(name: Self.vertexFunctionName)
        }
        let pipelineCache = MaterialPipelineCache(
            device: device,
            vertexFunction: vertexFunction,
            vertexDescriptor: Self.vertexDescriptor,
            colorPixelFormat: colorPixelFormat,
            depthPixelFormat: depthPixelFormat
        )
        let normalProgram: MaterialProgram
        do {
            normalProgram = try pipelineCache.program(
                functionName: Self.normalFragmentFunctionName,
                library: library,
                parameterLength: 0,
                resources: &resources
            )
        } catch .functionMissing(let name, _) {
            throw .shaderFunctionMissing(name: name)
        } catch {
            throw .pipelineCreationFailed(reason: error.description)
        }

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.label = "Frostfield.MeshDepth"
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            throw .depthStencilStateUnavailable
        }
        resources.depthStencilStatesCreated += 1

        self.device = device
        self.colorPixelFormat = colorPixelFormat
        self.depthPixelFormat = depthPixelFormat
        self.submitter = submitter
        self.depthState = depthState
        self.builtInLibrary = library
        self.pipelineCache = pipelineCache
        self.normalMaterial = Material(program: normalProgram, parameters: NoMaterialParameters())
        self.resourceStatistics = resources
    }

    /// The vertex layout `ff_mesh_vertex` reads: separate position and
    /// normal streams of `float3` with the stride of `SIMD3<Float>`.
    static var vertexDescriptor: MTLVertexDescriptor {
        let descriptor = MTLVertexDescriptor()
        let streams = [
            (FF_MESH_ATTRIBUTE_POSITION, FF_MESH_POSITION_BUFFER_INDEX),
            (FF_MESH_ATTRIBUTE_NORMAL, FF_MESH_NORMAL_BUFFER_INDEX),
        ]
        for (attribute, buffer) in streams {
            descriptor.attributes[Int(attribute)].format = .float3
            descriptor.attributes[Int(attribute)].offset = 0
            descriptor.attributes[Int(attribute)].bufferIndex = Int(buffer)
            descriptor.layouts[Int(buffer)].stride = Int(FF_MESH_VERTEX_STREAM_STRIDE)
            descriptor.layouts[Int(buffer)].stepFunction = .perVertex
        }
        return descriptor
    }

    /// Uploads validated geometry into new position, normal, and index
    /// buffers. Call this once per mesh, before drawing; drawing never
    /// allocates.
    ///
    /// - Throws: ``RendererError/bufferAllocationFailed`` when the device
    ///   cannot allocate a buffer.
    public func makeMesh(_ geometry: IndexedGeometry) throws(RendererError) -> Mesh {
        precondition(
            MemoryLayout<SIMD3<Float>>.stride == Int(FF_MESH_VERTEX_STREAM_STRIDE),
            "SIMD3<Float> must have the vertex stream stride"
        )
        func upload<Element>(_ elements: [Element], label: String) throws(RendererError) -> any MTLBuffer {
            let buffer = elements.withUnsafeBytes { bytes in
                device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
            }
            guard let buffer else { throw .bufferAllocationFailed }
            buffer.label = label
            resourceStatistics.buffersAllocated += 1
            resourceStatistics.bufferBytesAllocated += buffer.length
            return buffer
        }
        // IndexedGeometry guarantees every array is non-empty.
        let positionBuffer = try upload(geometry.positions, label: "Frostfield.MeshPositions")
        let normalBuffer = try upload(geometry.normals, label: "Frostfield.MeshNormals")
        let indexBuffer = try upload(geometry.indices, label: "Frostfield.MeshIndices")
        return Mesh(
            geometry: geometry,
            device: device,
            positionBuffer: positionBuffer,
            normalBuffer: normalBuffer,
            indexBuffer: indexBuffer
        )
    }

    /// Makes the built-in unlit material: every covered pixel receives
    /// `color`, written to the color attachment unchanged.
    ///
    /// The first call prepares the unlit pipeline; later calls reuse it.
    /// Change the material's `parameters.color` and draw again to recolor it.
    public func makeUnlitMaterial(color: SIMD4<Float>) throws(MaterialError) -> Material<UnlitParameters> {
        precondition(
            MemoryLayout<UnlitParameters>.size == MemoryLayout<FFUnlitParameters>.size
                && MemoryLayout<UnlitParameters>.offset(of: \.color) == 0,
            "UnlitParameters must have the layout of FFUnlitParameters"
        )
        return try makeMaterial(
            fragmentFunction: Self.unlitFragmentFunctionName,
            library: builtInLibrary,
            parameters: UnlitParameters(color: color)
        )
    }

    /// Makes a material from a fragment function the app supplies, with
    /// initial parameter values.
    ///
    /// The function shades what Frostfield's mesh vertex stage rasterizes. It
    /// may read the varyings `float4 [[position]]`,
    /// `float3 [[user(ff_model_normal)]]`, and
    /// `float3 [[user(ff_model_position)]]` through its `[[stage_in]]` struct,
    /// and one read-only parameter block, `constant T &parameters [[buffer(0)]]`,
    /// whose size must equal `MemoryLayout<Parameters>.size` and be smaller
    /// than 4096 bytes. An empty `Parameters` type means the function declares
    /// no parameter block. See `docs/materials.md`.
    ///
    /// The first material for a function prepares its render pipeline and
    /// checks its bindings through Metal's reflection; later materials that
    /// name the same function in the same library object reuse it.
    ///
    /// - Parameters:
    ///   - name: The fragment function's name in `library`.
    ///   - library: A library made by this renderer's device, such as the
    ///     app's `makeDefaultLibrary()` or one compiled from source.
    ///   - parameters: The material's initial parameter values.
    /// - Throws: ``MaterialError`` when the function is missing, is not a
    ///   fragment function, reads varyings the vertex stage does not write,
    ///   declares bindings Frostfield does not provide, or has a parameter
    ///   block of a different size. Nothing is encoded in any of these cases.
    public func makeMaterial<Parameters: BitwiseCopyable>(
        fragmentFunction name: String,
        library: any MTLLibrary,
        parameters: Parameters
    ) throws(MaterialError) -> Material<Parameters> {
        let program = try pipelineCache.program(
            functionName: name,
            library: library,
            parameterLength: MemoryLayout<Parameters>.size,
            resources: &resourceStatistics
        )
        return Material(program: program, parameters: parameters)
    }

    /// Encodes and commits one frame of `mesh` with the built-in normal
    /// diagnostic material, placed by `transform` and viewed through
    /// `camera`, or reports why it did not.
    ///
    /// - Precondition: `mesh` belongs to this renderer's device, and the
    ///   target's pass has a depth attachment of ``depthPixelFormat``.
    public func drawFrame(mesh: Mesh, transform: Transform, camera: Camera, target: () -> FrameTarget?) -> FrameOutcome {
        drawFrame(mesh: mesh, material: normalMaterial, transform: transform, camera: camera, target: target)
    }

    /// Encodes and commits one frame of `mesh` shaded by `material`, placed
    /// by `transform` and viewed through `camera`, or reports why it did not.
    ///
    /// The material's current ``Material/parameters`` are copied into the
    /// frame's commands while it is encoded; changing them afterwards affects
    /// only later frames. Drawing creates no pipeline or buffer.
    ///
    /// - Precondition: `mesh` belongs to this renderer's device, `material`
    ///   was made by this renderer, and the target's pass has a depth
    ///   attachment of ``depthPixelFormat``.
    public func drawFrame<Parameters>(
        mesh: Mesh,
        material: Material<Parameters>,
        transform: Transform,
        camera: Camera,
        target: () -> FrameTarget?
    ) -> FrameOutcome {
        precondition(
            mesh.device === device,
            "The mesh was created for the device \(mesh.device.name), not this renderer's device \(device.name)"
        )
        precondition(
            material.program.cache === pipelineCache,
            "The material \(material.fragmentFunctionName) was made by another MeshRenderer. "
                + "Make materials with the renderer that draws them; its pipeline matches only that renderer's formats."
        )
        return submitter.submit(label: "Frostfield.MeshFrame", target: target) { encoder, uniformBuffer, pass, colorTexture in
            let depthFormat = pass.depthAttachment.texture?.pixelFormat
            precondition(
                depthFormat == depthPixelFormat,
                "MeshRenderer needs a depth attachment of pixel format \(depthPixelFormat.rawValue), but the target has "
                    + (depthFormat.map { "format \($0.rawValue)" } ?? "none")
                    + ". For an MTKView, set depthStencilPixelFormat to the renderer's depthPixelFormat."
            )
            // The target is non-empty, so the ratio is finite and positive.
            let aspectRatio = Float(colorTexture.width) / Float(colorTexture.height)
            let uniforms = FFTransformUniforms(
                modelViewProjection: camera.uncheckedViewProjectionMatrix(aspectRatio: aspectRatio) * transform.matrix
            )
            uniformBuffer.contents().storeBytes(of: uniforms, as: FFTransformUniforms.self)

            encoder.setRenderPipelineState(material.program.pipeline)
            encoder.setDepthStencilState(depthState)
            encoder.setFrontFacing(.counterClockwise)
            encoder.setCullMode(.back)
            encoder.setVertexBuffer(uniformBuffer, offset: 0, index: Int(FF_TRANSFORM_UNIFORMS_BUFFER_INDEX))
            encoder.setVertexBuffer(mesh.positionBuffer, offset: 0, index: Int(FF_MESH_POSITION_BUFFER_INDEX))
            encoder.setVertexBuffer(mesh.normalBuffer, offset: 0, index: Int(FF_MESH_NORMAL_BUFFER_INDEX))
            // The parameter size was checked against the function's parameter
            // block when the material was made.
            if MemoryLayout<Parameters>.size > 0 {
                withUnsafeBytes(of: material.parameters) { bytes in
                    encoder.setFragmentBytes(
                        bytes.baseAddress!,
                        length: bytes.count,
                        index: Int(FF_MATERIAL_PARAMETERS_BUFFER_INDEX)
                    )
                }
            }
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: mesh.indexCount,
                indexType: .uint32,
                indexBuffer: mesh.indexBuffer,
                indexBufferOffset: 0
            )
        }
    }
}

/// The parameters of a material whose fragment function reads none.
struct NoMaterialParameters: BitwiseCopyable {}
