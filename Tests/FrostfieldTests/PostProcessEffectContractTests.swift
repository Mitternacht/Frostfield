import Foundation
import Frostfield
import Metal
import Testing
import simd

@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "Requires a Metal device"))
struct PostProcessEffectContractTests {
    private let gpu: PostProcessFixture

    init() throws {
        gpu = try PostProcessFixture()
    }

    /// Every built-in effect, configured so that `encode` takes its widest path.
    private func effects() throws -> [(name: String, effect: any PostProcessEffect & PostProcessPipelineCounting)] {
        let device = gpu.device
        let camera = try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100)
        let lens = try CameraLens(focalLength: 50, fNumber: 2, focusDistance: 2)
        let table = try ColorLookupTable(size: 5) { SIMD3($0.z, $0.x, $0.y) }
        return [
            ("ColorLookupPass", try ColorLookupPass(device: device, table: table)),
            ("PostProcessTexturePass", try PostProcessTexturePass(
                device: device, texture: gpu.color(width: 8, height: 8) { _, _ in SIMD4(1, 0, 0, 0.5) },
                opacity: 0.5, alphaMode: .straight
            )),
            ("SceneTransitionPass", try SceneTransitionPass(
                device: device, toTexture: gpu.color(width: 16, height: 16) { _, _ in SIMD4(0, 1, 0, 1) },
                mask: gpu.color(width: 8, height: 8) { _, _ in SIMD4(0.5, 0, 0, 1) },
                settings: .init(progress: 0.5)
            )),
            ("MipmapBloom", try MipmapBloom(device: device, settings: .init(threshold: 1))),
            ("BokehDepthOfField", try BokehDepthOfField(device: device, lens: lens, camera: camera)),
            ("SelectionOutline", try SelectionOutline(device: device)),
            ("GodRays", try GodRays(device: device, settings: .init(lightPosition: SIMD2(0.5, 0.5)))),
            ("LensFlare", try LensFlare(device: device, lights: [.init(position: SIMD2(0.3, 0.4))])),
            ("SpriteLensFlare", try SpriteLensFlare(
                device: device, elements: [.init(texture: gpu.color(width: 2, height: 2) { _, _ in SIMD4(1, 0, 0, 0.5) })],
                light: .init(position: SIMD2(0.6, 0.4))
            )),
            ("GaussianBlur", try GaussianBlur(device: device, settings: .init(sigma: 1.5))),
            ("KawaseBlur", try KawaseBlur(device: device, settings: .init(resolutionScale: 0.5))),
            ("BilateralBlur", try BilateralBlur(device: device, camera: camera, settings: .init(colorSigma: 0.1))),
            ("Afterimage", try Afterimage(device: device)),
            ("PostProcessHistory", try PostProcessHistory(queue: gpu.queue, width: 16, height: 16)),
        ]
    }

    @Test("Preparing an effect builds every pipeline its encode uses")
    func prepareBuildsEverything() throws {
        let source = try gpu.color(width: 16, height: 16) { x, y in SIMD4(Float(x) / 15, Float(y) / 15, 2, 1) }
        let depth = try gpu.depth(width: 16, height: 16) { x, _ in x < 8 ? 0.5 : 1 }
        let destination = try gpu.color(width: 16, height: 16)
        let graph = SceneGraph(roots: [], camera: try PerspectiveCamera(verticalFieldOfView: 1, near: 0.1, far: 100))
        for (name, effect) in try effects() {
            try effect.prepare(pixelFormat: .rgba16Float)
            let prepared = effect.builtPipelineCount
            #expect(prepared > 0, "\(name)")
            for _ in 0..<2 {
                try gpu.run { buffer in
                    if let outline = effect as? SelectionOutline {
                        try outline.encodeSelection(graph: graph, selection: [], width: 16, height: 16, commandBuffer: buffer)
                    }
                    try effect.encode(source: source, depth: depth, destination: destination, commandBuffer: buffer)
                }
            }
            #expect(effect.builtPipelineCount == prepared, "\(name) built \(effect.builtPipelineCount - prepared) pipelines while encoding")
        }
    }
}
