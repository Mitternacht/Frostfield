import Foundation
@testable import Frostfield
import Metal
import simd
import Testing

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PlanarReflectionBounceTests {
    private let gpu: GeometryRenderFixture
    private let aID = UUID()
    private let bID = UUID()
    private let aPosition = SIMD3<Float>(0, 0, -4)
    private let bPosition = SIMD3<Float>(3, 0, -4)
    private let aNormal = simd_normalize(SIMD3<Float>(1, 0, 1))
    private let bNormal = simd_normalize(SIMD3<Float>(-1, 0, 1))
    private let red = SIMD4<Float>(0.86, 0.07, 0.13, 1)
    private let green = SIMD4<Float>(0.08, 0.82, 0.24, 1)
    private let blue = SIMD4<Float>(0.03, 0.11, 0.96, 1)

    init() throws { gpu = try GeometryRenderFixture(side: 128) }

    @Test("Two mirrors sample the correct B capture for both the root view and A's reflected view",
          arguments: [MeshDepthMode.forward, .reversed], [false, true])
    func twoMirrors(mode: MeshDepthMode, transmitting: Bool) throws {
        let a = try reflection(id: aID, position: aPosition, normal: aNormal)
        let b = try reflection(id: bID, position: bPosition, normal: bNormal)
        let childBackdrop = try transmitting ? backdrop() : nil
        let rootBackdrop = try transmitting ? backdrop() : nil
        let bFromA = try capture(b, attachments: .init(transmissionBackdrop: childBackdrop))
        let aFromMain = try capture(a, children: [bFromA])
        let bFromMain = try capture(b, attachments: .init(transmissionBackdrop: rootBackdrop))
        let camera = try camera(mode)
        let glassNodes = try transmitting ? [glass(mode: mode)] : []

        // The center ray reflects from A along +X, then from B along +Z.
        // The main camera sees B from another angle and B reflects toward (-4, 0, 3).
        let markers = [
            try marker(at: SIMD3(3, 0, -1.5), normal: SIMD3(0, 0, -1), color: red, mode: mode),
            try marker(at: SIMD3(1, 0, -2.5), normal: SIMD3(0.8, 0, -0.6), color: green, mode: mode),
        ]
        let aNode = try mirror(a, position: aPosition, texture: aFromMain.color, mode: mode)
        let bNode = try mirror(b, position: bPosition, texture: bFromMain.color, mode: mode)
        var excludedA = aNode
        let rotation = aNode.transform.rotation
        var blocker = try marker(at: SIMD3(3, 0, -3), normal: SIMD3(0, 0, -1), color: blue, mode: mode)
        blocker.transform.translation = simd_inverse(rotation).act(blocker.transform.translation - aPosition)
        blocker.transform.rotation = simd_inverse(rotation) * blocker.transform.rotation
        // The ancestor contains both a read of its in-progress target and a closer blocker.
        excludedA.children = [blocker]

        let command = try #require(gpu.queue.makeCommandBuffer())
        var paths: [[UUID]] = []
        let rootColors = try PlanarReflection.encodeBounces(
            [aFromMain, bFromMain], camera: camera, depthMode: mode, commandBuffer: command
        ) { path, children in
            paths.append(path)
            if path == [aID] {
                let sampled = try #require(children[bID])
                #expect(sampled === bFromA.color)
                let localB = try mirror(b, position: bPosition, texture: sampled, mode: mode)
                return scene(markers + [aNode, localB])
            }
            #expect(children.isEmpty)
            let reflector = path == [aID, bID] ? excludedA : bNode
            return scene(markers + glassNodes + [reflector])
        }
        #expect(paths == [[aID, bID], [aID], [bID]])
        #expect(rootColors.count == 2)
        #expect(rootColors[aID] === aFromMain.color)
        #expect(rootColors[bID] === bFromMain.color)
        let main = try gpu.makeTarget()
        try scene([aNode, bNode]).encode(
            pass: gpu.pass(color: main.color, depth: main.depth, depthMode: mode), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix * camera.viewMatrix, cameraPosition: .zero,
            depthMode: mode, graphCamera: camera
        )
        try gpu.finish(command)

        let center = gpu.side / 2
        let bPixel = Int(0.875 * Float(gpu.side))
        let bounced = transmitting ? red * SIMD4(0.5, 1, 1, 1) : red
        try expectColor(bounced, in: bFromA.color, x: center, y: center)
        try expectColor(bounced, in: aFromMain.color, x: center, y: center)
        try expectColor(green, in: bFromMain.color, x: gpu.side - bPixel - 1, y: center)
        try expectColor(bounced, in: main.color, x: center, y: center)
        try expectColor(green, in: main.color, x: bPixel, y: center)
        if let childBackdrop, let rootBackdrop {
            try expectColor(red, in: childBackdrop, x: center, y: center)
            try expectColor(green, in: rootBackdrop, x: gpu.side - bPixel - 1, y: center)
        }
    }

    @Test("Automatic discovery derives nested world planes and binds the correct capture at each bounce",
          arguments: [MeshDepthMode.forward, .reversed])
    func automaticMirrors(mode: MeshDepthMode) throws {
        let mirrorGeometry = try MeshGeometry.plane(width: 1.7, height: 1.7)
        let fallback = try gpu.unlit(
            mirrorGeometry, material: .classic(ClassicMaterial(model: .basic, color: SIMD3(blue.x, blue.y, blue.z))),
            depthMode: mode
        )
        let parent = SceneTransform(
            translation: SIMD3(0.3, -0.2, 0.5), rotation: simd_quatf(angle: 0.2, axis: SIMD3(0, 1, 0)),
            scale: SIMD3(1.6, 0.8, 0.6)
        )
        let inverse = simd_inverse(parent.matrix)
        let localPosition = inverse * SIMD4(aPosition, 1)
        let localNormal = parent.matrix.transpose * SIMD4(aNormal, 0)
        let a = SceneNode(id: aID, transform: .init(
            translation: SIMD3(localPosition.x, localPosition.y, localPosition.z),
            rotation: simd_quatf(from: SIMD3(0, 0, 1), to: simd_normalize(SIMD3(localNormal.x, localNormal.y, localNormal.z)))
        ), mesh: fallback)
        let b = SceneNode(id: bID, transform: .init(
            translation: bPosition, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: bNormal)
        ), mesh: fallback)
        let hidden = SceneNode(transform: .init(translation: aPosition), mesh: fallback)
        let offLayer = SceneNode(transform: .init(translation: aPosition), mesh: fallback, layers: .init(layer: 1))
        let reflectedOnly = SceneNode(transform: .init(translation: SIMD3(50, 30, -4)), mesh: fallback)
        let offscreen = SceneNode(transform: .init(translation: SIMD3(50, 300, -4)), mesh: fallback)
        let markers = [
            try marker(at: SIMD3(3, 0, -1.5), normal: SIMD3(0, 0, -1), color: red, mode: mode),
            try marker(at: SIMD3(1, 0, -2.5), normal: SIMD3(0.8, 0, -0.6), color: green, mode: mode),
        ]
        let graph = scene([
            SceneNode(transform: parent, children: [a]), b,
            SceneNode(children: [hidden], isVisible: false), offLayer, reflectedOnly, offscreen,
        ] + markers)
        let localPlane = try Plane3D(normal: SIMD3(0, 0, 1), constant: 0)
        let descriptors = [a, b, hidden, offLayer, reflectedOnly, offscreen].map {
            PlanarReflectorSet.Mirror(nodeID: $0.id, localPlane: localPlane)
        }
        let camera = try camera(mode)
        let center = gpu.side / 2
        let bPixel = Int(0.875 * Float(gpu.side))
        let owner = try PlanarReflectorSet(queue: gpu.queue, width: gpu.side, height: gpu.side, mirrors: descriptors)
        let command = try #require(gpu.queue.makeCommandBuffer())
        let otherCommand = try #require(gpu.queue.makeCommandBuffer())
        var paths: Set<[UUID]> = []
        let frame = try owner.encode(graph, camera: camera, depthMode: mode, commandBuffer: command) { path in
            if paths.isEmpty {
                #expect(throws: PlanarReflectorSet.Failure.inFlight) {
                    try owner.encode(graph, camera: camera, depthMode: mode, commandBuffer: otherCommand)
                }
            }
            paths.insert(path)
            return .init()
        }
        #expect(paths.contains([aID]) && paths.contains([bID]) && paths.contains([aID, bID]))
        #expect(paths.contains([aID, reflectedOnly.id]) && !paths.contains([reflectedOnly.id]))
        let excluded = Set([hidden.id, offLayer.id, offscreen.id])
        #expect(paths.allSatisfy { excluded.isDisjoint(with: $0) })
        #expect(frame.captureCount == paths.count)
        #expect(graph.node(withID: aID)?.mesh === fallback)
        #expect(frame.graph.node(withID: aID)?.mesh !== fallback)
        #expect(frame.graph.node(withID: offLayer.id)?.mesh === fallback)
        #expect(throws: PlanarReflectorSet.Failure.inFlight) {
            try owner.encode(graph, camera: camera, depthMode: mode, commandBuffer: otherCommand)
        }
        let result = try renderAutomatic(frame.graph, camera: camera, mode: mode, command: command)
        try expectColor(red, in: result, x: center, y: center)
        try expectColor(green, in: result, x: bPixel, y: center)

        let reused = try owner.encode(graph, camera: camera, depthMode: mode, commandBuffer: otherCommand)
        #expect(reused.graph.node(withID: aID)?.mesh === frame.graph.node(withID: aID)?.mesh)
        try gpu.finish(otherCommand)

        let single = try PlanarReflectorSet(
            queue: gpu.queue, width: gpu.side, height: gpu.side, mirrors: descriptors, maximumDepth: 1
        )
        let singleCommand = try #require(gpu.queue.makeCommandBuffer())
        let first = try single.encode(graph, camera: camera, depthMode: mode, commandBuffer: singleCommand)
        #expect(first.captureCount == 2)
        let oneBounce = try renderAutomatic(first.graph, camera: camera, mode: mode, command: singleCommand)
        try expectColor(blue, in: oneBounce, x: center, y: center)
        try expectColor(green, in: oneBounce, x: bPixel, y: center)
    }

    @Test("Automatic filters supply each B view to a different parent and preserve authored roughness",
          arguments: [MeshDepthMode.forward, .reversed])
    func automaticFiltering(mode: MeshDepthMode) throws {
        let plane = try Plane3D(normal: SIMD3(0, 0, 1), constant: 0)
        let bAppearance = try PlanarReflectorSet.Appearance(
            blur: .init(radius: SIMD2(3.5, 2.5)), mix: .init(blurMix: 1, mirror: 1)
        )
        let owner = try PlanarReflectorSet(queue: gpu.queue, width: gpu.side, height: gpu.side, mirrors: [
            .init(nodeID: aID, localPlane: plane), .init(nodeID: bID, localPlane: plane, appearance: bAppearance),
        ])
        let geometry = try MeshGeometry.plane(width: 1.7, height: 1.7)
        let a = SceneNode(id: aID, transform: .init(
            translation: aPosition, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: aNormal)
        ), mesh: try gpu.unlit(geometry, depthMode: mode))
        let markers = [
            try marker(at: SIMD3(3, 0, -1.5), normal: SIMD3(0, 0, -1), color: red, mode: mode, halfExtent: 0.36),
            try marker(at: SIMD3(1, 0, -2.5), normal: SIMD3(0.8, 0, -0.6), color: green, mode: mode, halfExtent: 0.36),
        ]
        let camera = try camera(mode)
        func render(roughness: Float) throws -> (composed: any MTLTexture, rootBlur: any MTLTexture) {
            let pbr = try PhysicallyBasedMaterial(roughness: roughness, shadingModel: .unlit)
            let b = SceneNode(id: bID, transform: .init(
                translation: bPosition, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: bNormal)
            ), mesh: try gpu.unlit(geometry, material: .physicallyBased(pbr), depthMode: mode))
            let command = try #require(gpu.queue.makeCommandBuffer())
            var paths: Set<[UUID]> = []
            let frame = try owner.encode(scene([a, b] + markers), camera: camera, depthMode: mode, commandBuffer: command) { path in
                paths.insert(path)
                return .init()
            }
            #expect(paths.contains([aID, bID]) && paths.contains([bID]))
            let mesh = try #require(frame.graph.node(withID: bID)?.mesh)
            guard case .physicallyBased(let material) = mesh.material else { throw FixtureFailure.expectedPBR }
            #expect(PhysicallyBasedMaterial.Factors(material).roughness == roughness)
            #expect(material.surfaceGraph?.textures.map(\.name) == ["Reflection", "ReflectionBlur"])
            #expect(material.surfaceTextures.count == 2)
            #expect(material.surfaceTextures.first !== material.surfaceTextures.last)
            let rootBlur = try gpu.makeTarget().color
            let filtered = try #require(material.surfaceTextures.last)
            try TextureTransfer.encodeCopy(from: filtered, to: rootBlur, commandBuffer: command)
            return (try renderAutomatic(frame.graph, camera: camera, mode: mode, command: command), rootBlur)
        }
        let sharp = try render(roughness: 0)
        let soft = try render(roughness: 1)
        let center = gpu.side / 2
        let bPixel = Int(0.875 * Float(gpu.side))
        try expectColor(red, in: sharp.composed, x: center, y: center)
        try expectColor(green, in: sharp.composed, x: bPixel, y: center)
        let softPixels = try gpu.pixels(of: soft.composed)
        let redBounce = gpu.pixel(softPixels, x: center, y: center)
        let greenRoot = gpu.pixel(softPixels, x: bPixel, y: center)
        #expect((0.02..<(red.x * 0.8)).contains(redBounce.x), "Red bounce \(redBounce)")
        #expect(redBounce.x > 4 * redBounce.y)
        #expect((0.02..<(green.y * 0.8)).contains(greenRoot.y), "Green root \(greenRoot)")
        #expect(greenRoot.y > 4 * greenRoot.x)
        let rootBlur = try gpu.pixel(gpu.pixels(of: soft.rootBlur), x: gpu.side - bPixel - 1, y: center)
        #expect(isNearly(greenRoot, rootBlur, tolerance: 0.006))
    }

    private enum FixtureFailure: Error { case expectedPBR }

    @Test("Automatic discovery rejects an incompatible authored mirror and leaves empty views unchanged")
    func automaticValidation() throws {
        let plane = try Plane3D(normal: SIMD3(0, 0, 1), constant: 0)
        let descriptor = PlanarReflectorSet.Mirror(nodeID: aID, localPlane: plane)
        #expect(throws: PlanarReflectorSet.Failure.duplicateMirror(aID)) {
            try PlanarReflectorSet(queue: gpu.queue, width: 32, height: 32, mirrors: [descriptor, descriptor])
        }
        let owner = try PlanarReflectorSet(queue: gpu.queue, width: 32, height: 32, mirrors: [descriptor])
        let command = try #require(gpu.queue.makeCommandBuffer())
        let camera = try camera(.forward)
        let cube = SceneNode(id: aID, transform: .init(translation: aPosition), mesh: try gpu.unlit(.cube()))
        #expect(throws: PlanarReflectorSet.Failure.nonplanarMirror(aID)) {
            try owner.encode(scene([cube]), camera: camera, commandBuffer: command)
        }
        let backFacing = SceneNode(id: aID, transform: .init(
            translation: aPosition, rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        ), mesh: try gpu.unlit(MeshGeometry.plane(width: 1, height: 1)))
        let back = try owner.encode(scene([backFacing]), camera: camera, commandBuffer: command)
        #expect(back.captureCount == 0 && back.graph.roots[0].mesh === backFacing.mesh)
    }

    private func renderAutomatic(
        _ graph: SceneGraph, camera: ShaderGraphCamera, mode: MeshDepthMode, command: any MTLCommandBuffer
    ) throws -> any MTLTexture {
        let target = try gpu.makeTarget()
        try graph.encode(
            pass: gpu.pass(color: target.color, depth: target.depth, depthMode: mode), commandBuffer: command,
            clipFromWorld: camera.projectionMatrix * camera.viewMatrix, cameraPosition: .zero,
            depthMode: mode, graphCamera: camera
        )
        try gpu.finish(command)
        return target.color
    }

    @Test("Each reflected view uses its own directional cascades or punctual depth cache", arguments: [false, true])
    func reflectedShadows(punctual: Bool) throws {
        let direction = SIMD3<Float>(1, 0, -1)
        func attachments() throws -> PlanarReflection.Capture.Attachments {
            if punctual {
                return .init(punctualShadowMaps: try PunctualShadowMaps(
                    queue: gpu.queue, resolution: 128, spotCapacity: 1, pointCapacity: 0, updateMode: .onDemand
                ))
            }
            return .init(shadowMap: try DirectionalShadowMap(
                queue: gpu.queue, resolution: 256, directionToLight: direction,
                cascades: .init(count: 2, maximumDistance: 14, blendFraction: 0, casterMargin: 5),
                intensity: 4, sampling: .basic, updateMode: .onDemand
            ))
        }
        let a = try capture(reflection(id: aID, position: aPosition, normal: SIMD3(0, 0, 1)), attachments: attachments())
        let b = try capture(reflection(id: bID, position: aPosition, normal: SIMD3(0, 0, 1)), attachments: attachments())
        let receiver = SceneNode(
            transform: .init(translation: SIMD3(0, 0, -1), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))),
            mesh: try MeshRenderer(
                device: gpu.device, colorPixelFormat: .rgba16Float, geometry: GeometryRenderFixture.square(2),
                material: .physicallyBased(PhysicallyBasedMaterial(
                    roughness: 1, lightIntensity: 0, ambientIntensity: 0, receivesShadows: !punctual,
                    transmission: .init(factor: 0, indexOfRefraction: 1)
                ))
            ), castsShadow: false
        )
        let caster = SceneNode(
            transform: .init(translation: SIMD3(0.8, 0, -1.8), scale: SIMD3(repeating: 0.5)),
            mesh: try gpu.unlit(.cube())
        )
        let lightPosition = SIMD3<Float>(2, 0, -3)
        let lights: [SceneNode] = punctual ? [SceneNode(
            transform: PunctualLightingFixture.aim(lightPosition, at: receiver.transform.translation),
            light: try PunctualLight(kind: .spot(), intensity: 48, shadow: .init(normalBias: 0, sampling: .basic))
        )] : []
        let command = try #require(gpu.queue.makeCommandBuffer())
        let source = try camera(.forward)
        _ = try PlanarReflection.encodeBounces([a, b], camera: source, commandBuffer: command) { path, _ in
            var blocker = caster
            blocker.castsShadow = path == [aID]
            return scene([receiver, blocker] + lights)
        }
        try gpu.finish(command)
        let x = gpu.side / 2
        let dark = try gpu.pixel(gpu.pixels(of: a.color), x: x, y: x)
        let lit = try gpu.pixel(gpu.pixels(of: b.color), x: x, y: x)
        #expect(dark.w == 1 && lit.w == 1)
        #expect(lit.x > 0.5 && dark.x < 0.05 * lit.x)
        if let directional = a.attachments.shadowMap {
            #expect(directional.cascadeViews.count == 2)
            let sourceFit = try #require(try attachments().shadowMap)
            try sourceFit.fitCascades(view: source)
            #expect(directional.cascadeViews != sourceFit.cascadeViews)
        }
    }

    @Test("Reflection plans reject shared scratch textures and shadow map queues before asking for a scene")
    func attachmentDependencies() throws {
        let a = try reflection(id: aID, position: aPosition, normal: aNormal)
        let b = try reflection(id: bID, position: bPosition, normal: bNormal)
        let camera = try camera(.forward)
        let command = try #require(gpu.queue.makeCommandBuffer())
        func noScene(_ path: [UUID], _ colors: [UUID: any MTLTexture]) -> SceneGraph {
            Issue.record("An invalid attachment plan must fail before requesting the scene")
            return scene([])
        }
        let scratch = try backdrop()
        let sharedA = try capture(a, attachments: .init(transmissionBackdrop: scratch))
        let sharedB = try capture(b, attachments: .init(transmissionBackdrop: scratch))
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([sharedA, sharedB], camera: camera, commandBuffer: command, sceneForView: noScene)
        }
        let invalidReader = try mirror(b, position: bPosition, texture: scratch, mode: .forward)
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([sharedA], camera: camera, commandBuffer: command) { _, _ in
                scene([invalidReader])
            }
        }
        let output = try capture(b)
        let aliased = PlanarReflection.Capture(b, color: scratch, depth: output.depth)
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([sharedA, aliased], camera: camera, commandBuffer: command, sceneForView: noScene)
        }
        let noMips = try capture(a, attachments: .init(transmissionBackdrop: output.color))
        #expect(throws: MeshRenderer.Failure.invalidTransmissionBackdrop) {
            try PlanarReflection.encodeBounces([noMips], camera: camera, commandBuffer: command, sceneForView: noScene)
        }
        let shadows = try PunctualShadowMaps(queue: gpu.queue, resolution: 32, spotCapacity: 1, pointCapacity: 0)
        let first = try capture(a, attachments: .init(punctualShadowMaps: shadows))
        let second = try capture(b, attachments: .init(punctualShadowMaps: shadows))
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([first, second], camera: camera, commandBuffer: command, sceneForView: noScene)
        }
        let wrongQueue = try #require(gpu.device.makeCommandQueue())
        let wrongBuffer = try #require(wrongQueue.makeCommandBuffer())
        #expect(throws: PunctualShadowMaps.Failure.wrongQueue) {
            try PlanarReflection.encodeBounces([first], camera: camera, commandBuffer: wrongBuffer, sceneForView: noScene)
        }
    }

    @Test("The capture plan rejects sibling texture aliases, repeated reflectors and reused attachments")
    func invalidDependencies() throws {
        let a = try reflection(id: aID, position: aPosition, normal: aNormal)
        let b = try reflection(id: bID, position: bPosition, normal: bNormal)
        let captureA = try capture(a)
        let captureB = try capture(b)
        let camera = try camera(.forward)
        let alias = try #require(captureB.color.makeTextureView(pixelFormat: .rgba16Float))
        let wrongBinding = try mirror(b, position: bPosition, texture: alias, mode: .forward)
        let first = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([captureA, captureB], camera: camera, commandBuffer: first) { _, _ in
                scene([wrongBinding])
            }
        }
        let snapshotCommand = try #require(gpu.queue.makeCommandBuffer())
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([captureA, captureB], camera: camera, commandBuffer: snapshotCommand) { _, _ in
                var graph = scene([])
                graph.viewportSnapshot = try SceneViewportSnapshot(color: alias)
                return graph
            }
        }
        let command = try #require(gpu.queue.makeCommandBuffer())
        let repeated = try capture(a, children: [captureA])
        let duplicateRoot = try capture(a)
        for invalidPlan in [[repeated], [captureA, duplicateRoot]] {
            #expect(throws: PlanarReflection.Failure.duplicateReflector) {
                try PlanarReflection.encodeBounces(invalidPlan, camera: camera, commandBuffer: command) { _, _ in
                    Issue.record("An invalid reflector plan must fail before requesting the scene")
                    return scene([])
                }
            }
        }
        let shared = PlanarReflection.Capture(b, color: captureA.color, depth: captureB.depth)
        #expect(throws: PlanarReflection.Failure.feedback) {
            try PlanarReflection.encodeBounces([captureA, shared], camera: camera, commandBuffer: command) { _, _ in
                Issue.record("Shared capture attachments must fail before requesting the scene")
                return scene([])
            }
        }
        var overDepth: [PlanarReflection.Capture] = []
        for _ in 0..<5 {
            overDepth = [try capture(reflection(id: UUID(), position: aPosition, normal: -aNormal), children: overDepth)]
        }
        #expect(throws: PlanarReflection.Failure.captureLimit) {
            try PlanarReflection.encodeBounces(overDepth, camera: camera, commandBuffer: command) { _, _ in
                Issue.record("The depth bound applies even to inactive subtrees")
                return scene([])
            }
        }
    }

    @Test("An inactive child clears its own result and leaves descendants untouched")
    func inactiveSubtree() throws {
        let a = try reflection(id: aID, position: aPosition, normal: aNormal)
        let b = try reflection(id: bID, position: bPosition, normal: -bNormal)
        let skipped = try capture(reflection(id: UUID(), position: bPosition, normal: bNormal))
        let bTarget = try capture(b, children: [skipped], clearColor: MTLClearColor(red: 0.75, green: 0.25, blue: 0.5, alpha: 1))
        let aTarget = try capture(a, children: [bTarget])
        let command = try #require(gpu.queue.makeCommandBuffer())
        let sentinel = gpu.pass(color: skipped.color, depth: skipped.depth)
        sentinel.colorAttachments[0].clearColor = MTLClearColor(red: 0.125, green: 0.5, blue: 0.875, alpha: 1)
        let fill = try #require(command.makeRenderCommandEncoder(descriptor: sentinel))
        fill.endEncoding()
        var paths: [[UUID]] = []
        _ = try PlanarReflection.encodeBounces([aTarget], camera: camera(.forward), commandBuffer: command) { path, children in
            paths.append(path)
            #expect(children.count == 1)
            #expect(children[bID] === bTarget.color)
            let localB = try mirror(b, position: bPosition, texture: bTarget.color, mode: .forward, facing: bNormal)
            return scene([localB])
        }
        try gpu.finish(command)
        #expect(paths == [[aID]])
        try expectColor(SIMD4(0.75, 0.25, 0.5, 1), in: bTarget.color, x: 64, y: 64)
        try expectColor(SIMD4(0.75, 0.25, 0.5, 1), in: aTarget.color, x: 64, y: 64)
        try expectColor(SIMD4(0.125, 0.5, 0.875, 1), in: skipped.color, x: 64, y: 64)
    }

    private func camera(_ mode: MeshDepthMode) throws -> ShaderGraphCamera {
        try ShaderGraphCamera(
            viewMatrix: matrix_identity_float4x4, projectionMatrix: gpu.camera.projection(aspectRatio: 1, depthMode: mode),
            near: gpu.camera.near, far: gpu.camera.far
        )
    }

    private func scene(_ roots: [SceneNode]) -> SceneGraph {
        SceneGraph(roots: roots, camera: gpu.camera)
    }

    private func reflection(id: UUID, position: SIMD3<Float>, normal: SIMD3<Float>) throws -> PlanarReflection {
        PlanarReflection(plane: try Plane3D(normal: normal, through: position), reflector: id)
    }

    private func capture(
        _ reflection: PlanarReflection, children: [PlanarReflection.Capture] = [],
        clearColor: MTLClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1),
        attachments: PlanarReflection.Capture.Attachments = .init()
    ) throws -> PlanarReflection.Capture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        let color = try #require(gpu.device.makeTexture(descriptor: descriptor))
        descriptor.pixelFormat = .depth32Float
        descriptor.storageMode = .private
        descriptor.usage = .renderTarget
        let depth = try #require(gpu.device.makeTexture(descriptor: descriptor))
        return .init(reflection, color: color, depth: depth, children: children, clearColor: clearColor, attachments: attachments)
    }

    private func backdrop() throws -> any MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: gpu.side, height: gpu.side, mipmapped: true
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead]
        return try #require(gpu.device.makeTexture(descriptor: descriptor))
    }

    private func glass(mode: MeshDepthMode) throws -> SceneNode {
        let material = try PhysicallyBasedMaterial(
            roughness: 0, lightIntensity: 0, ambientIntensity: 0, baseColorFactor: SIMD4(0.5, 1, 1, 1),
            transmission: .init(factor: 1, indexOfRefraction: 1)
        )
        let mesh = try gpu.unlit(
            GeometryRenderFixture.square(0.45), material: .physicallyBased(material), depthMode: mode
        )
        return SceneNode(transform: .init(
            translation: SIMD3(3, 0, -2.8), rotation: simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0))
        ), mesh: mesh)
    }

    private func mirror(
        _ reflection: PlanarReflection, position: SIMD3<Float>, texture: any MTLTexture, mode: MeshDepthMode,
        facing: SIMD3<Float>? = nil
    ) throws -> SceneNode {
        let surface = try SurfaceShaderGraph { graph in
            let map = graph.texture("Reflection", addressing: .clampToEdge)
            return .init(baseColor: map.samplePlanarReflection().rgb)
        }
        let material = try PhysicallyBasedMaterial(
            surfaceGraph: surface, surfaceTextures: ["Reflection": texture], shadingModel: .unlit
        )
        let mesh = try gpu.unlit(
            MeshGeometry.plane(width: 1.7, height: 1.7), material: .physicallyBased(material), depthMode: mode
        )
        return SceneNode(
            id: reflection.reflector, transform: .init(
                translation: position, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: facing ?? SIMD3<Float>(reflection.plane.unitNormal))
            ), mesh: mesh
        )
    }

    private func marker(
        at position: SIMD3<Float>, normal: SIMD3<Float>, color: SIMD4<Float>, mode: MeshDepthMode, halfExtent: Float = 0.22
    ) throws -> SceneNode {
        let mesh = try gpu.unlit(
            GeometryRenderFixture.square(halfExtent), material: .classic(ClassicMaterial(model: .basic, color: SIMD3(color.x, color.y, color.z))),
            depthMode: mode
        )
        return SceneNode(transform: .init(translation: position, rotation: simd_quatf(from: SIMD3(0, 0, 1), to: normal)), mesh: mesh)
    }

    private func expectColor(
        _ expected: SIMD4<Float>, in texture: any MTLTexture, x: Int, y: Int, sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let pixel = try gpu.pixel(gpu.pixels(of: texture), x: x, y: y)
        #expect(isNearly(pixel, expected, tolerance: 0.006), sourceLocation: sourceLocation)
    }
}
