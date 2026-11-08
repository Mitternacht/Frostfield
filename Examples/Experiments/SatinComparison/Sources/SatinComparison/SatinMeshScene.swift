import Foundation
import Metal
import Satin
import simd

/// The Satin side of the comparison, kept inside this experiment: a Context,
/// a RenderEncoder, an orthographic camera, and one Mesh with an indexed
/// Geometry and a SourceMaterial loaded from Shaders/Stripes/Shaders.metal.
/// No Satin type leaves this file except through `encode`.
final class SatinMeshScene {
    let context: Context
    let material: SourceMaterial
    private let renderer: RenderEncoder
    private let camera: OrthographicCamera
    private let scene: Object
    private let indexStorage: UnsafeMutableRawPointer
    private var frameNumber = 0

    init(device: any MTLDevice, mesh fixtureMesh: Fixture.Mesh, shaderURL: URL, label: String = "Stripes") {
        context = Context(device: device, sampleCount: 1, colorPixelFormat: Fixture.pixelFormat)
        renderer = RenderEncoder(context: context, clearColor: Fixture.clearColor, frameBufferOnly: false)
        renderer.resize((width: Float(Fixture.width), height: Float(Fixture.height)))

        camera = OrthographicCamera(context: context, left: -1, right: 1, bottom: -1, top: 1, near: Fixture.near, far: Fixture.far)
        camera.position = SIMD3(0, 0, Fixture.cameraZ)
        camera.lookAt(target: .zero)

        let geometry = Geometry(context: context)
        geometry.addAttribute(Float3BufferAttribute(defaultValue: .zero, data: fixtureMesh.positions), for: .Position)
        geometry.addAttribute(Float3BufferAttribute(defaultValue: .zero, data: Array(repeating: SIMD3<Float>(0, 0, 1), count: fixtureMesh.positions.count)), for: .Normal)
        geometry.addAttribute(Float2BufferAttribute(defaultValue: .zero, data: fixtureMesh.texcoords), for: .Texcoord)
        // ElementBuffer takes a raw pointer whose lifetime the caller owns.
        let byteCount = fixtureMesh.indices.count * MemoryLayout<UInt32>.stride
        let storage = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: MemoryLayout<UInt32>.alignment)
        fixtureMesh.indices.withUnsafeBytes { storage.copyMemory(from: $0.baseAddress!, byteCount: byteCount) }
        indexStorage = storage
        geometry.setElements(ElementBuffer(type: .uint32, data: storage, count: fixtureMesh.indices.count, source: nil))

        material = SourceMaterial(context: context, pipelineURL: shaderURL)
        material.label = label

        let mesh = Mesh(context: context, label: "Grid", geometry: geometry, material: material)
        mesh.cullMode = .none
        mesh.position = Fixture.translation
        mesh.orientation = simd_quatf(angle: Fixture.rotationZ, axis: SIMD3(0, 0, 1))
        mesh.scale = Fixture.scale
        scene = Object(context: context, label: "Scene", [mesh])
    }

    deinit {
        indexStorage.deallocate()
    }

    func encode(into commandBuffer: any MTLCommandBuffer, target: any MTLTexture) throws {
        frameNumber += 1
        let frame = FrameTime(time: 0, deltaTime: 0, displayTime: nil, systemTime: 0, hostMediaTime: 0, frameNumber: frameNumber)
        try renderer.draw(
            frame: frame,
            renderPassDescriptor: MTLRenderPassDescriptor(),
            commandBuffer: commandBuffer,
            scene: scene,
            camera: camera,
            renderTarget: target
        )
    }
}
