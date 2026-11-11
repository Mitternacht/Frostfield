import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

/// A studio subject, image-based lighting, a shadow-catching sweep,
/// depth of field, and a display-referred output.
@MainActor
final class LabStudioFixture {
    static let argument = "--pbr-studio"
    static let proceduralArgument = "--procedural-camera"
    static let assetStudioArgument = "--asset-studio"
    static let assetStudioCaptureArgument = "--capture-asset-studio"
    private static let assetStudioYaw: Float = -0.36
    private let queue: any MTLCommandQueue
    private let camera: PerspectiveCamera
    private let asset: LabStudioAsset
    private let sweep: [SceneNode]
    private let shadow: DirectionalShadowMap
    private let occlusion: AmbientOcclusion?
    private let focus: DepthOfField
    private let display: HDRDisplay
    private let background: EnvironmentBackground?
    private let lightRig: [SceneNode]
    private let closeUp: Bool
    private let procedural: Bool
    private let bakesAssets: Bool
    private let environmentYaw: Float
    private struct RenderTargets {
        let scene: any MTLTexture
        let scratch: any MTLTexture
        let transmissionBackdrop: (any MTLTexture)?
    }

    private var targets: RenderTargets?
    private var angle = LabStudioAsset.suggestedAngle
    private var elevation: Float
    private var distance: Float

    init(
        device: any MTLDevice, destinationFormat: MTLPixelFormat, environmentEnabled: Bool = true,
        closeUp: Bool = false, showsEnvironment: Bool = false, usesImportedLights: Bool = false,
        occlusionEnabled: Bool = true, sourceURL: URL? = nil, procedural: Bool = false,
        generatedFinish: Bool = false, bakesAssets: Bool = false, sweepBake: LabAssetBaking.SweepBake = .staticDirect
    ) throws {
        guard let queue = device.makeCommandQueue() else { throw MeshRenderer.Failure.resource("studio command queue") }
        let procedural = procedural || bakesAssets
        self.queue = queue
        self.closeUp = closeUp
        self.procedural = procedural
        self.bakesAssets = bakesAssets
        environmentYaw = procedural ? 0 : -0.35
        if procedural {
            angle = bakesAssets ? 0 : -0.36
            elevation = closeUp ? 0.9 : 1.85
            distance = closeUp ? 4.78 : 7.25
        } else {
            elevation = closeUp ? 0.32 : 0.82
            distance = closeUp ? 1.95 : 5.65
        }
        camera = try PerspectiveCamera(verticalFieldOfView: 0.61, near: 0.1, far: 50)
        if usesImportedLights, !procedural {
            guard let rig = Bundle.main.url(forResource: "StudioRig", withExtension: "gltf") else {
                throw MeshRenderer.Failure.resource("bundled StudioRig.gltf")
            }
            let source = try GLTFMeshLoader.loadScene(contentsOf: rig, includeLights: true)
            let prepared = try PreparedGLTFScene(device: device, colorPixelFormat: .rgba16Float, scene: source)
            lightRig = try prepared.graph(camera: camera).roots
        } else {
            lightRig = []
        }
        let environment: PhysicallyBasedMaterial.Environment?
        let sweepEnvironment: PhysicallyBasedMaterial.Environment?
        if environmentEnabled {
            let lighting: EnvironmentLighting
            if procedural {
                lighting = try LabProceduralLighting.make(queue: queue)
            } else {
                guard let url = Bundle.main.url(forResource: "StudioLight", withExtension: "hdr") else {
                    throw MeshRenderer.Failure.resource("bundled StudioLight.hdr")
                }
                lighting = try EnvironmentLighting(queue: queue, radianceHDR: Data(contentsOf: url))
            }
            environment = try PhysicallyBasedMaterial.Environment(
                lighting: lighting, intensity: procedural ? 0.43 : 0.21, yaw: environmentYaw
            )
            sweepEnvironment = try PhysicallyBasedMaterial.Environment(
                lighting: lighting, intensity: procedural ? 0.12 : 0.072, yaw: environmentYaw
            )
            background = showsEnvironment
                ? try EnvironmentBackground(lighting: lighting, destinationPixelFormat: .rgba16Float) : nil
        } else {
            environment = nil
            sweepEnvironment = nil
            background = nil
        }

        let light = SIMD3<Float>(-0.52, 1.45, 0.58)
        let intensity: Float = procedural ? 2.6 : 3.3
        let subject: LabStudioAsset
        if procedural {
            let generated = LabProceduralCamera(
                device: device, light: light, intensity: intensity, environment: environment,
                generatedFinish: generatedFinish, bakedGrip: bakesAssets
            )
            subject = try LabStudioAsset(reference: generated.graph(camera: camera))
        } else {
            subject = try LabStudioAsset(
                device: device, pixelFormat: .rgba16Float, directionToLight: light, lightIntensity: intensity,
                ambientIntensity: 0.16, receivesShadows: true, environment: environment, sourceURL: sourceURL
            )
        }
        asset = subject
        shadow = try DirectionalShadowMap(
            queue: queue, resolution: 1024, directionToLight: light, center: SIMD3(0, 1.15, 0),
            halfExtent: 4.1, intensity: intensity, bias: 0.0010, lightAngularRadius: 0.12
        )
        if bakesAssets {
            let includesBackdrop = background == nil
            let floorAtlasFraction: Float = includesBackdrop ? 0.68 : 1
            let geometry = try Self.sweepGeometry(
                includesBackdrop: includesBackdrop, frontExtent: 12, halfWidth: 15,
                floorAtlasFraction: floorAtlasFraction, separatesFloor: sweepBake.usesSharedAtlas
            )
            let reference = try subject.graph(camera: camera, cameraTransform: .init(), angle: Self.assetStudioYaw)
            sweep = try LabAssetBaking.sweep(
                geometry, beneath: reference, queue: queue, light: light, intensity: intensity,
                bake: sweepBake, floorAtlasFraction: floorAtlasFraction
            )
        } else {
            let material = try PhysicallyBasedMaterial(
                metallic: 0, roughness: 0.74, directionToLight: light, lightIntensity: intensity,
                ambientIntensity: 0.22, receivesShadows: true,
                baseColorFactor: procedural ? SIMD4(0.092, 0.126, 0.16, 1) : SIMD4(0.13, 0.146, 0.17, 1),
                alphaMode: .opaque, environment: sweepEnvironment
            )
            let mesh = try MeshRenderer(
                device: device, colorPixelFormat: .rgba16Float,
                geometry: Self.sweepGeometry(includesBackdrop: background == nil, frontExtent: procedural ? 36 : 12),
                material: .physicallyBased(material)
            )
            sweep = [SceneNode(mesh: mesh, castsShadow: false)]
        }
        occlusion = occlusionEnabled ? try AmbientOcclusion(device: device) : nil
        focus = try DepthOfField(device: device)
        display = try HDRDisplay(device: device, destinationPixelFormat: destinationFormat)
    }

    static func sourceURL(arguments: [String]) throws -> URL? {
        let prefix = "--studio-asset="
        guard let argument = arguments.first(where: { $0.hasPrefix(prefix) }) else { return nil }
        let path = String(argument.dropFirst(prefix.count))
        guard !path.isEmpty else {
            throw CocoaError(.fileReadInvalidFileName, userInfo: [
                NSLocalizedDescriptionKey: "Use --studio-asset=/path/to/model.glb."
            ])
        }
        return URL(fileURLWithPath: path)
    }

    func orbit(horizontal: Float, vertical: Float) {
        guard horizontal.isFinite, vertical.isFinite else { return }
        angle = (angle + horizontal * 0.006).truncatingRemainder(dividingBy: 2 * .pi)
        elevation = min(max(elevation + vertical * 0.009, 0.05), closeUp ? 1.2 : 2.7)
    }

    func zoom(by scale: Float) {
        guard scale.isFinite, scale > 0 else { return }
        distance = min(max(distance / scale, closeUp ? 1.25 : 3.5), 10)
    }

    func draw(
        pass drawablePass: MTLRenderPassDescriptor,
        drawable: any CAMetalDrawable,
        completed: @escaping @Sendable (String?) -> Void
    ) throws {
        guard let depth = drawablePass.depthAttachment.texture else { throw MeshRenderer.Failure.attachments }
        guard let buffer = queue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("studio frame") }
        try encode(destination: drawable.texture, depth: depth, buffer: buffer)
        buffer.addCompletedHandler { command in
            completed(command.status == .completed ? nil : command.error?.localizedDescription ?? "The studio frame did not finish.")
        }
        buffer.present(drawable)
        buffer.commit()
    }

    #if os(macOS)
    func capture(to url: URL, angle: Float?) throws {
        if let angle { self.angle = angle }
        let target = try LabImageCapture.target(device: queue.device, width: 1440, height: 960)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: MeshRenderer.depthPixelFormat, width: target.width, height: target.height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.usage = [.renderTarget, .shaderRead]
        guard let depth = queue.device.makeTexture(descriptor: descriptor), let buffer = queue.makeCommandBuffer() else {
            throw MeshRenderer.Failure.resource("studio capture resources")
        }
        try encode(destination: target, depth: depth, buffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else {
            throw buffer.error ?? MeshRenderer.Failure.resource("studio capture completion")
        }
        try LabImageCapture.write(target, to: url)
    }
    #endif

    private func encode(
        destination: any MTLTexture, depth: any MTLTexture, buffer: any MTLCommandBuffer
    ) throws {
        let targets = try renderTargets(width: destination.width, height: destination.height)
        let aspect = Float(destination.width) / Float(destination.height)
        let height = procedural ? (closeUp ? 0.49 : 0.52) : (closeUp ? 0.89 : 0.53)
        let centerX: Float = procedural ? (closeUp ? -0.16 : 0.27) : 0
        let target = SIMD3<Float>(centerX, asset.dimensions.y * Float(height), 0)
        let framing = procedural ? min(1, aspect / 1.35) : max(0.52, min(1, aspect))
        let offset = SIMD3<Float>(closeUp ? 0.14 : 0.22, elevation, distance / framing)
        let orbit = simd_quatf(angle: angle, axis: SIMD3(0, 1, 0))
        // Keep the baked receiver and its casters fixed. Inspect them by moving the camera.
        let eye = target + (bakesAssets ? orbit.act(offset) : offset)
        let assetAngle = bakesAssets ? Self.assetStudioYaw : angle
        let facing = simd_normalize(target - eye)
        let pose = SceneTransform(
            translation: eye, rotation: simd_quatf(from: SIMD3<Float>(0, 0, -1), to: facing)
        )
        var graph = try asset.graph(camera: camera, cameraTransform: pose, angle: assetAngle)
        graph.roots.append(contentsOf: sweep)
        graph.roots.append(contentsOf: lightRig)

        if let background {
            try background.encode(
                destination: targets.scene, camera: camera, cameraTransform: pose,
                yaw: environmentYaw, intensity: procedural ? 0.35 : 0.18, commandBuffer: buffer
            )
        }

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.scene
        pass.colorAttachments[0].loadAction = background == nil ? .clear : .load
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.035, green: 0.04, blue: 0.05, alpha: 1)
        pass.depthAttachment.texture = depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .store
        pass.depthAttachment.clearDepth = 1
        try graph.encode(
            pass: pass, commandBuffer: buffer, shadowMap: shadow, transmissionBackdrop: targets.transmissionBackdrop
        )

        if let occlusion {
            try occlusion.encode(
                source: targets.scene, depth: depth, destination: targets.scratch, camera: camera,
                settings: .init(radius: 0.36, strength: 0.42), commandBuffer: buffer
            )
        }
        let sceneColor = occlusion == nil ? targets.scene : targets.scratch
        let focused = occlusion == nil ? targets.scratch : targets.scene

        // Pull focus toward the lens for a close-up, while leaving the rear bellows behind it.
        let focusOffset = closeUp ? asset.dimensions.z * 0.31 : (procedural ? asset.dimensions.z * 0.19 : 0)
        let lensOffset = SIMD3(sin(assetAngle), 0, cos(assetAngle)) * focusOffset
        let focusDistance = simd_dot(target + lensOffset - eye, facing)
        try focus.encode(
            source: sceneColor, depth: depth, destination: focused, camera: camera,
            settings: .init(focusDistance: focusDistance, fNumber: procedural ? 6 : 3.8, sensorHeight: 0.067, maximumRadius: 7),
            commandBuffer: buffer
        )
        try display.encode(
            source: focused, destination: destination,
            settings: .init(exposureStops: -0.18, antiAliasing: true), commandBuffer: buffer
        )
    }

    private func renderTargets(width: Int, height: Int) throws -> RenderTargets {
        if let targets, targets.scene.width == width, targets.scene.height == height { return targets }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
        )
        descriptor.storageMode = .private
        descriptor.hazardTrackingMode = .tracked
        descriptor.usage = [.renderTarget, .shaderRead]
        guard let scene = queue.device.makeTexture(descriptor: descriptor),
              let scratch = queue.device.makeTexture(descriptor: descriptor) else {
            throw MeshRenderer.Failure.resource("studio HDR render targets")
        }
        scene.label = "FrostfieldLab studio HDR color"
        scratch.label = "FrostfieldLab studio post-process color"
        let backdrop: (any MTLTexture)?
        if asset.requiresTransmissionBackdrop {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: width, height: height, mipmapped: true
            )
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.usage = .shaderRead
            guard let texture = queue.device.makeTexture(descriptor: descriptor) else {
                throw MeshRenderer.Failure.resource("studio transmission backdrop")
            }
            texture.label = "FrostfieldLab opaque transmission backdrop"
            backdrop = texture
        } else {
            backdrop = nil
        }
        let targets = RenderTargets(scene: scene, scratch: scratch, transmissionBackdrop: backdrop)
        self.targets = targets
        return targets
    }

    static func sweepGeometry(
        includesBackdrop: Bool, frontExtent: Float, halfWidth: Float = 18, floorAtlasFraction: Float? = nil,
        separatesFloor: Bool = false
    ) throws -> MeshGeometry {
        // A studio cyclorama curves from the floor to the vertical backdrop.
        var rows: [(y: Float, z: Float, normal: SIMD3<Float>)] = [
            (0, frontExtent, SIMD3(0, 1, 0)),
            (0, includesBackdrop || separatesFloor ? -3.6 : -20, SIMD3(0, 1, 0)),
        ]
        if includesBackdrop {
            let radius: Float = 2.6
            for segment in 1...18 {
                let angle = Float(segment) * (.pi / 2) / 18
                rows.append((
                    radius * (1 - cos(angle)), -3.6 - radius * sin(angle), SIMD3(0, cos(angle), sin(angle))
                ))
            }
            rows.append((10, -3.6 - radius, SIMD3(0, 0, 1)))
        } else if separatesFloor {
            rows.append((0, -20, SIMD3(0, 1, 0)))
        }

        var vertices: [MeshVertex] = []
        var indices: [UInt32] = []
        for (index, row) in rows.enumerated() {
            let v: Float
            if let floorAtlasFraction, includesBackdrop, index > 0 {
                // Give the shadow-catching floor most of the atlas resolution.
                let backdrop = Float(index - 1) / Float(rows.count - 2)
                v = floorAtlasFraction + backdrop * (1 - floorAtlasFraction)
            } else {
                v = Float(index) / Float(rows.count - 1)
            }
            vertices.append(MeshVertex(position: SIMD3(-halfWidth, row.y, row.z), normal: row.normal, textureCoordinate: SIMD2(0, v)))
            vertices.append(MeshVertex(position: SIMD3(halfWidth, row.y, row.z), normal: row.normal, textureCoordinate: SIMD2(1, v)))
            if index > 0 {
                let a = UInt32((index - 1) * 2)
                indices += [a, a + 1, a + 2, a + 1, a + 3, a + 2]
            }
        }
        return try MeshGeometry(vertices: vertices, indices: indices)
    }
}
