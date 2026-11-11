import Foundation
import Metal
import simd

/// Ordered camera views of one scene, drawn into pixel rectangles of one target.
public struct ArrayCamera: Sendable {
    public struct View: Sendable {
        /// The separate camera matrices and clip distances used by surface graphs.
        public let camera: ShaderGraphCamera
        public let position: SIMD3<Float>
        /// Orthographic shading uses a fixed world-space direction toward the camera.
        public let directionToCamera: SIMD3<Float>?
        /// Top-left attachment pixels and Metal window depth.
        public let viewport: MTLViewport
        /// An optional top-left attachment rectangle independent of the viewport.
        public let scissor: MTLScissorRect?
        /// Layers this view draws. A nil mask draws everything the graph's `cameraLayers` allow.
        /// A mask adds a second test, as a Three.js sub-camera's `layers` do: a node draws only
        /// when its layers intersect both the graph's `cameraLayers` and this mask. The graph's
        /// mask still alone decides lights, shadow casters and probe volumes.
        public let layers: SceneLayers?
        public let depthMode: MeshDepthMode

        /// Uses an authored perspective, orthographic or off-axis projection.
        public init(
            camera: ShaderGraphCamera, position: SIMD3<Float>, viewport: MTLViewport,
            directionToCamera: SIMD3<Float>? = nil, scissor: MTLScissorRect? = nil,
            layers: SceneLayers? = nil, depthMode: MeshDepthMode = .forward
        ) {
            self.camera = camera
            self.position = position
            self.directionToCamera = directionToCamera
            self.viewport = viewport
            self.scissor = scissor
            self.layers = layers
            self.depthMode = depthMode
        }

        /// Builds a perspective projection with the viewport's own aspect ratio.
        public init(
            camera: PerspectiveCamera, transform: SceneTransform, viewport: MTLViewport,
            scissor: MTLScissorRect? = nil, layers: SceneLayers? = nil, depthMode: MeshDepthMode = .forward
        ) throws {
            let projection = try camera.projection(
                aspectRatio: Float(viewport.width / viewport.height), depthMode: depthMode
            )
            let authored = try ShaderGraphCamera(
                viewMatrix: simd_inverse(transform.matrix), projectionMatrix: projection,
                near: camera.near, far: camera.far
            )
            self.init(
                camera: authored, position: transform.translation, viewport: viewport,
                scissor: scissor, layers: layers, depthMode: depthMode
            )
        }

        public var clipFromWorld: simd_float4x4 { camera.projectionMatrix * camera.viewMatrix }
    }

    public var views: [View]

    public init(views: [View] = []) {
        self.views = views
    }

    public enum Failure: LocalizedError, Equatable {
        case mixedDepthModes
        case commandBuffer
        case transientAttachment
        case onDemandSun
        case passDescriptor
        case viewIndex

        public var errorDescription: String? {
            switch self {
            case .mixedDepthModes: "Views sharing stored depth must use the same depth mode."
            case .commandBuffer: "Encode array camera views on a command buffer that was not yet committed."
            case .transientAttachment: "Multiple camera views need tracked, storable render attachments."
            case .onDemandSun: "An on-demand camera-fitted sun cannot refit distinct views in one command buffer."
            case .passDescriptor: "Could not copy the render pass for the camera views."
            case .viewIndex: "The array camera has no view at this index."
            }
        }
    }

    /// The render pass each view encodes with. One view uses the caller's descriptor as it is.
    /// Several views share every attachment, so the first clears and the later ones load, and
    /// every view except the last stores its color, depth and stencil regardless of the
    /// caller's store actions. The last view alone applies those actions and resolve targets.
    fileprivate struct Passes {
        let first: MTLRenderPassDescriptor
        let middle: MTLRenderPassDescriptor
        let last: MTLRenderPassDescriptor

        init(_ original: MTLRenderPassDescriptor, count: Int) throws {
            if count == 1 {
                first = original
                middle = original
                last = original
                return
            }
            guard Self.attachments(original).allSatisfy({ attachment in
                guard let texture = attachment.texture else { return false }
                return texture.storageMode != .memoryless && texture.hazardTrackingMode != .untracked && !texture.isAliasable()
            }) else { throw Failure.transientAttachment }
            guard let first = original.copy() as? MTLRenderPassDescriptor,
                  let middle = original.copy() as? MTLRenderPassDescriptor,
                  let last = original.copy() as? MTLRenderPassDescriptor else { throw Failure.passDescriptor }
            for pass in [first, middle] {
                for attachment in Self.attachments(pass) {
                    attachment.storeAction = .store
                    attachment.resolveTexture = nil
                }
            }
            for pass in [middle, last] {
                for attachment in Self.attachments(pass) { attachment.loadAction = .load }
            }
            self.first = first
            self.middle = middle
            self.last = last
        }

        func descriptor(for index: Int, count: Int) -> MTLRenderPassDescriptor {
            if index == 0 { return first }
            if index == count - 1 { return last }
            return middle
        }

        private static func attachments(_ pass: MTLRenderPassDescriptor) -> [MTLRenderPassAttachmentDescriptor] {
            let all: [MTLRenderPassAttachmentDescriptor] =
                (0..<8).map { pass.colorAttachments[$0]! } + [pass.depthAttachment, pass.stencilAttachment]
            return all.filter { $0.texture != nil }
        }
    }
}

public extension SceneGraph {
    /// Encodes each subview in array order and returns statistics in that order.
    /// The caller's load actions apply to the whole attachment at the first pass.
    /// Later passes load the previous color, depth and stencil. Overlapping views
    /// therefore share depth tests. Every pass but the last stores each attachment, even when the
    /// caller's store action is `dontCare`; the last pass applies the caller's store actions and
    /// resolve destinations. The pass descriptor is unchanged. An empty array performs no work.
    ///
    /// One view encodes exactly like the explicit-view overload, with the caller's pass descriptor,
    /// and accepts the same command buffers, including one with unretained references.
    /// Several views also accept an unretained command buffer; keep the attachments alive as usual.
    /// A command buffer that was already committed throws `ArrayCamera.Failure.commandBuffer`.
    ///
    /// Every view renders independently, which goes beyond the Three.js `WebGLRenderer` array-camera
    /// loop (r187 culls with one union frustum and updates levels of detail and shadow maps once):
    /// each view selects its own levels of detail, culls with its own frustum and re-encodes
    /// its shadow depth. An automatically updated directional or punctual shadow map is cleared
    /// and redrawn once per view, in order, so the map holds the last view's depth afterwards and
    /// each view's `shadowPassCount` reports its own passes. An on-demand map follows its update
    /// cache instead and draws in the first view that needs it. A camera-fitted sun refits per view;
    /// an on-demand one rejects distinct camera fits in one command buffer. Transmission copies the
    /// whole attachment into `transmissionBackdrop` before each view's overlay pass.
    ///
    /// With more than one view, every view after the first is first validated by a CPU-only
    /// pass over the same scene preparation, so a failure in a later view throws before any view
    /// encodes and nothing is encoded into the command buffer. That repeats the preparation of
    /// views after the first, and a camera-fitted sun keeps the fit of the last view checked.
    /// Failures that only happen while creating encoders or leasing deformation vertices cannot
    /// be foreseen; if one throws, discard the command buffer.
    ///
    /// The explicit-view options `screenSpaceOcclusion`, `screenSpaceDirectionalShadow`,
    /// `indirectDraws`, `updatingSkeletons`, `updatingBoxes`, `updatingPlanes`, `toonOutline`
    /// and `motionVectors` are not available. All views share the pass's attachments, slice
    /// and level, so a view cannot target a different array slice.
    @discardableResult
    func encode(
        pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer, arrayCamera: ArrayCamera,
        shadowMap: DirectionalShadowMap? = nil, transmissionBackdrop: (any MTLTexture)? = nil,
        punctualShadowMaps: PunctualShadowMaps? = nil, ordering: SceneRenderOrdering = .init()
    ) throws -> [SceneDrawStatistics] {
        try encodeArrayCamera(
            pass: pass, commandBuffer: commandBuffer, arrayCamera: arrayCamera, shadowMap: shadowMap,
            transmissionBackdrop: transmissionBackdrop, punctualShadowMaps: punctualShadowMaps,
            ordering: ordering, observingViewPasses: nil
        )
    }

    /// Picks through one view of an array camera at a top-left attachment pixel, in the units of
    /// the view's viewport. A point outside the viewport, or outside its scissor, hits nothing.
    /// A hit lies between the view's clip planes on a node the view would draw: visible, on a layer
    /// in both the graph's `cameraLayers` and the view's `layers`, with levels of detail selected at
    /// the view's position. Use `group` as the explicit-camera overload does. Other views never
    /// occlude a hit, and points, lines and sprites are not picked. Choose the view yourself where
    /// rectangles overlap, for example the last view whose viewport contains the point.
    /// Throws `ArrayCamera.Failure.viewIndex` when the array has no such view.
    func pick(
        atPixel pixel: SIMD2<Float>, in arrayCamera: ArrayCamera, viewIndex: Int, group: UUID? = nil
    ) throws -> SceneHit? {
        guard arrayCamera.views.indices.contains(viewIndex) else { throw ArrayCamera.Failure.viewIndex }
        let view = arrayCamera.views[viewIndex]
        let port = view.viewport
        guard pixel.x.isFinite, pixel.y.isFinite else { throw ScenePickingError.nonfiniteQueryPoint }
        guard port.width > 0, port.height > 0, port.width.isFinite, port.height.isFinite else {
            throw MeshRenderer.Failure.viewport
        }
        let x = Double(pixel.x), y = Double(pixel.y)
        guard x >= port.originX, y >= port.originY,
              x <= port.originX + port.width, y <= port.originY + port.height else { return nil }
        if let scissor = view.scissor {
            guard x >= Double(scissor.x), y >= Double(scissor.y),
                  x < Double(scissor.x + scissor.width), y < Double(scissor.y + scissor.height) else { return nil }
        }
        let normalized = SIMD2<Float>(
            Float(min(max((x - port.originX) / port.width, 0), 1)),
            Float(min(max((y - port.originY) / port.height, 0), 1))
        )
        return try nearestMeshHit(
            ray: view.camera.ray(at: normalized, depthMode: view.depthMode), cameraPosition: view.position,
            layers: cameraLayers, viewLayers: view.layers, group: group
        )
    }

    /// The array-camera encode. `observingViewPasses` sees each view's pass descriptor just before
    /// that view encodes, so tests can check the actions the scene encoder really receives.
    @discardableResult
    package func encodeArrayCamera(
        pass: MTLRenderPassDescriptor, commandBuffer: any MTLCommandBuffer, arrayCamera: ArrayCamera,
        shadowMap: DirectionalShadowMap?, transmissionBackdrop: (any MTLTexture)?,
        punctualShadowMaps: PunctualShadowMaps?, ordering: SceneRenderOrdering,
        observingViewPasses observer: ((Int, MTLRenderPassDescriptor) -> Void)?
    ) throws -> [SceneDrawStatistics] {
        let views = arrayCamera.views
        guard let first = views.first else { return [] }
        guard commandBuffer.status == .notEnqueued || commandBuffer.status == .enqueued else {
            throw ArrayCamera.Failure.commandBuffer
        }
        func encodeView(
            _ index: Int, pass: MTLRenderPassDescriptor, validateOnly: Bool
        ) throws -> SceneDrawStatistics {
            let view = views[index]
            return try MeshRenderer.encode(
                self, pass: pass, commandBuffer: commandBuffer, shadowMap: shadowMap,
                punctualShadowMaps: punctualShadowMaps,
                view: (view.clipFromWorld, view.position, view.directionToCamera), viewport: view.viewport,
                depthMode: view.depthMode, transmissionBackdrop: transmissionBackdrop,
                graphCamera: view.camera, scissor: view.scissor, ordering: ordering,
                viewLayers: view.layers, viewIndex: (index, views.count), validateOnly: validateOnly
            )
        }
        guard views.count > 1 else {
            observer?(0, pass)
            return [try encodeView(0, pass: pass, validateOnly: false)]
        }

        guard views.allSatisfy({ $0.depthMode == first.depthMode }) else {
            throw ArrayCamera.Failure.mixedDepthModes
        }
        if let shadowMap, shadowMap.automaticallyFitsCamera, shadowMap.updateMode == .onDemand,
           views.contains(where: { $0.camera != first.camera }) {
            throw ArrayCamera.Failure.onDemandSun
        }
        let size = try MeshRenderer.renderTargetSize(in: pass, device: commandBuffer.device)
        for view in views {
            try MeshRenderer.validateScissor(view.scissor, size: size)
            try MeshRenderer.validateViewport(view.viewport, size: size)
            _ = try MeshRenderer.explicitShadingCamera(
                clipFromWorld: view.clipFromWorld, position: view.position, direction: view.directionToCamera,
                highPrecision: highPrecision
            )
            if highPrecision, view.camera.relativeView(at: SIMD3<Double>(view.position)) == nil {
                throw MeshRenderer.Failure.highPrecisionView
            }
        }

        let passes = try ArrayCamera.Passes(pass, count: views.count)
        // The first view validates itself before it creates an encoder. Check the rest now.
        for index in views.indices.dropFirst() {
            _ = try encodeView(index, pass: passes.descriptor(for: index, count: views.count), validateOnly: true)
        }
        return try views.indices.map { index in
            let descriptor = passes.descriptor(for: index, count: views.count)
            observer?(index, descriptor)
            return try encodeView(index, pass: descriptor, validateOnly: false)
        }
    }
}
