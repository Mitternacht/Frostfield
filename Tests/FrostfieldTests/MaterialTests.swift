import Foundation
import FrostfieldShaderTypes
import Metal
import Testing
import simd
@testable import Frostfield

private let systemDevice = MTLCreateSystemDefaultDevice()

/// The built-in unlit material's Swift parameters are copied byte for byte
/// into the shader's `FFUnlitParameters`, so their layouts must agree.
struct MaterialLayoutTests {
    @Test func `Unlit parameters have the shader record's layout`() {
        #expect(MemoryLayout<UnlitParameters>.size == MemoryLayout<FFUnlitParameters>.size)
        #expect(MemoryLayout<UnlitParameters>.size == 16)
        #expect(MemoryLayout<UnlitParameters>.alignment == MemoryLayout<FFUnlitParameters>.alignment)
        #expect(MemoryLayout<UnlitParameters>.offset(of: \.color) == 0)
        #expect(MemoryLayout<FFUnlitParameters>.offset(of: \.color) == 0)
        var parameters = UnlitParameters(color: [0.25, 0.5, 0.75, 1])
        let floats = withUnsafeBytes(of: &parameters) { Array($0.bindMemory(to: Float.self)) }
        #expect(floats == [0.25, 0.5, 0.75, 1])
    }

    @Test func `The parameter block has its own fragment buffer index`() {
        #expect(FF_MATERIAL_PARAMETERS_BUFFER_INDEX == 0)
        #expect(MemoryLayout<NoMaterialParameters>.size == 0)
        #expect(MaterialPipelineCache.parameterLengthLimit == 4096)
    }

    @Test(.enabled(if: systemDevice != nil))
    func `The packaged library holds the built-in material functions`() throws {
        let library = try TriangleRenderer.makeShaderLibrary(device: try #require(systemDevice))
        for name in [MeshRenderer.normalFragmentFunctionName, MeshRenderer.unlitFragmentFunctionName] {
            let function = try #require(library.makeFunction(name: name), "\(name)")
            #expect(function.functionType == .fragment, "\(name)")
        }
    }
}

/// App-side shaders for the material tests, compiled at test time the way an
/// app might compile its own library. The varyings struct is the declaration
/// documented in `docs/materials.md`, written here without Frostfield's
/// header: Metal matches it to `ff_mesh_vertex`'s output by the
/// `[[user(...)]]` names.
enum MaterialTestShaders {
    static let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct Varyings {
            float4 position [[position]];
            float3 normal [[user(ff_model_normal)]];
            float3 modelPosition [[user(ff_model_position)]];
        };

        // A copy of the built-in normal diagnostic.
        fragment float4 test_normal_fragment(Varyings in [[stage_in]]) {
            return float4(normalize(in.normal) * 0.5 + 0.5, 1.0);
        }

        // Model-space position of a unit cube, mapped to [0.2, 1] per channel
        // so that no covered pixel is black.
        fragment float4 test_position_fragment(Varyings in [[stage_in]]) {
            return float4((in.modelPosition + 0.5) * 0.8 + 0.2, 1.0);
        }

        // 32 bytes: float4 color, float scale, and 12 bytes of tail padding.
        struct TintParameters {
            float4 color;
            float scale;
        };

        fragment float4 test_tint_fragment(constant TintParameters &parameters [[buffer(0)]]) {
            return parameters.color * parameters.scale;
        }

        fragment float4 test_constant_fragment() {
            return float4(0.0, 1.0, 1.0, 1.0);
        }

        struct UVVaryings {
            float4 position [[position]];
            float2 uv [[user(ff_uv)]];
        };

        fragment float4 test_unknown_varying_fragment(UVVaryings in [[stage_in]]) {
            return float4(in.uv, 0.0, 1.0);
        }

        fragment float4 test_texture_fragment(
            constant float4 &color [[buffer(0)]],
            texture2d<float> image [[texture(0)]])
        {
            return color * image.read(uint2(0, 0));
        }

        fragment float4 test_extra_buffer_fragment(
            constant float4 &color [[buffer(0)]],
            constant float4 &extra [[buffer(3)]])
        {
            return color + extra;
        }

        fragment float4 test_writable_fragment(device float4 *parameters [[buffer(0)]]) {
            parameters[1] = parameters[0];
            return parameters[0];
        }

        vertex float4 test_vertex(uint vertexID [[vertex_id]]) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }

        kernel void test_kernel(device float *values [[buffer(0)]], uint index [[thread_position_in_grid]]) {
            values[index] = 0.0;
        }
        """

    static let fragmentFunctions = [
        "test_constant_fragment", "test_extra_buffer_fragment", "test_normal_fragment",
        "test_position_fragment", "test_texture_fragment", "test_tint_fragment",
        "test_unknown_varying_fragment", "test_writable_fragment",
    ]

    static func makeLibrary(device: any MTLDevice) async throws -> any MTLLibrary {
        try await device.makeLibrary(source: source, options: nil)
    }
}

/// `test_tint_fragment`'s parameter block as a SIMD value with a defined
/// layout: color in lanes 0–3, scale in lane 4, lanes 5–7 padding.
typealias TintParameters = SIMD8<Float>

/// Draws the cube with the built-in and app-supplied materials off screen on
/// the system GPU and reads the results back.
@Suite(.enabled(if: systemDevice != nil))
struct MaterialRenderingTests {
    let device: any MTLDevice
    let completions: AsyncStream<FrameCompletion>
    let reportCompletion: AsyncStream<FrameCompletion>.Continuation

    init() throws {
        device = try #require(systemDevice)
        (completions, reportCompletion) = AsyncStream.makeStream(of: FrameCompletion.self)
    }

    func makeRenderer(device: (any MTLDevice)? = nil) throws -> MeshRenderer {
        let report = reportCompletion
        return try MeshRenderer(device: device ?? self.device, colorPixelFormat: .bgra8Unorm, onFrameCompleted: { report.yield($0) })
    }

    func makeTarget(size: Int = 128) throws -> OffscreenTarget {
        try OffscreenTarget(device: device, width: size, height: size, depthPixelFormat: .depth32Float)
    }

    static func byte(_ value: Float) -> Int { Int((value * 255).rounded()) }

    static func pixel(_ pixel: OffscreenTarget.Pixel, matches color: SIMD4<Float>, tolerance: Int = 1) -> Bool {
        abs(Int(pixel.red) - byte(color.x)) <= tolerance
            && abs(Int(pixel.green) - byte(color.y)) <= tolerance
            && abs(Int(pixel.blue) - byte(color.z)) <= tolerance
            && abs(Int(pixel.alpha) - byte(color.w)) <= tolerance
    }

    /// Built-in and app-supplied materials share the vertex stage, culling,
    /// and depth test, so they cover exactly the same pixels in every pose,
    /// and an app copy of the normal diagnostic is byte-identical to the
    /// built-in one.
    @Test(arguments: CubePose.all)
    func `Every material covers the same pixels`(pose: CubePose) async throws {
        let renderer = try makeRenderer()
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        let mesh = try renderer.makeMesh(.cube())
        let unlit = try renderer.makeUnlitMaterial(color: [1, 0.5, 0.25, 1])
        let normalCopy = try renderer.makeMaterial(fragmentFunction: "test_normal_fragment", library: library, parameters: NoMaterialParameters())
        let position = try renderer.makeMaterial(fragmentFunction: "test_position_fragment", library: library, parameters: NoMaterialParameters())

        let builtIn = try makeTarget()
        try await submitAndWait(renderer.drawFrame(mesh: mesh, transform: pose.transform, camera: pose.camera) { builtIn.frameTarget })
        let builtInPixels = builtIn.pixels()
        let builtInCoverage = builtInPixels.map { !$0.isClear }
        #expect(builtInCoverage.contains(true) && builtInCoverage.contains(false))

        let copyTarget = try makeTarget()
        try await draw(renderer, mesh, normalCopy, pose, into: copyTarget)
        let copyPixels = copyTarget.pixels()
        let differing = zip(builtInPixels, copyPixels).filter { $0 != $1 }.count
        #expect(differing == 0, "\(differing) pixels differ between the built-in and the app's normal diagnostic")

        for (name, coverage) in [
            ("unlit", try await coverage(renderer, mesh, unlit, pose)),
            ("position", try await coverage(renderer, mesh, position, pose)),
        ] {
            let differing = zip(builtInCoverage, coverage).filter { $0 != $1 }.count
            #expect(differing == 0, "\(name): \(differing) pixels differ in coverage from the built-in material")
        }
    }

    /// The app's fragment function receives `[[user(ff_model_position)]]`:
    /// every unambiguous pixel shows the model-space point where a CPU-cast
    /// ray enters the cube.
    @Test(arguments: [CubePose.all[3], CubePose.all[4], CubePose.all[5]])
    func `A custom material receives model-space positions`(pose: CubePose) async throws {
        let renderer = try makeRenderer()
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        let material = try renderer.makeMaterial(fragmentFunction: "test_position_fragment", library: library, parameters: NoMaterialParameters())
        let target = try makeTarget()
        try await draw(renderer, try renderer.makeMesh(.cube()), material, pose, into: target)

        let reference = try BoxRayCaster(camera: pose.camera, transform: pose.transform, aspectRatio: 1, size: 1)
        let pixels = target.pixels()
        var compared = 0
        var worst: Float = 0
        for y in 0..<128 {
            for x in 0..<128 {
                let ndc = target.ndc(x: x, y: y)
                guard case .face(_, let margin) = reference.hit(ndc: ndc), margin > 0.03 else { continue }
                let (origin, direction) = reference.ray(ndc: ndc)
                var entry: Float = -.infinity
                for i in 0..<3 {
                    entry = max(entry, min((-0.5 - origin[i]) / direction[i], (0.5 - origin[i]) / direction[i]))
                }
                let expected = origin + entry * direction
                let pixel = pixels[y * 128 + x]
                let decoded = (SIMD3<Float>(Float(pixel.red), Float(pixel.green), Float(pixel.blue)) / 255 - 0.2) / 0.8 - 0.5
                worst = max(worst, simd_reduce_max(simd_abs(decoded - expected)))
                compared += 1
            }
        }
        #expect(compared > 1000, "only \(compared) pixels were unambiguous")
        // One 8-bit step is 1 / (255 × 0.8) ≈ 0.005 model units.
        #expect(worst < 0.012, "largest position error \(worst) model units over \(compared) pixels")
    }

    /// Parameters reach the shader unchanged, including alpha: there is no
    /// blending, so the stored alpha is the shader's.
    @Test func `Parameters reach the fragment function`() async throws {
        let renderer = try makeRenderer()
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        let mesh = try renderer.makeMesh(.cube())
        let pose = CubePose.all[0]

        let unlit = try renderer.makeUnlitMaterial(color: [0.2, 0.4, 0.6, 0.8])
        let unlitTarget = try makeTarget(size: 64)
        try await draw(renderer, mesh, unlit, pose, into: unlitTarget)
        let unlitPixel = unlitTarget.pixel(at: [0, 0])
        #expect(Self.pixel(unlitPixel, matches: [0.2, 0.4, 0.6, 0.8]), "unlit: \(unlitPixel)")

        let tint = try renderer.makeMaterial(
            fragmentFunction: "test_tint_fragment", library: library,
            parameters: TintParameters(0.2, 0.4, 0.6, 1.0, 0.5, 0, 0, 0)
        )
        let tintTarget = try makeTarget(size: 64)
        try await draw(renderer, mesh, tint, pose, into: tintTarget)
        let tintPixel = tintTarget.pixel(at: [0, 0])
        #expect(Self.pixel(tintPixel, matches: [0.1, 0.2, 0.3, 0.5]), "tint: \(tintPixel)")
        #expect(tintTarget.pixel(at: [-0.95, 0.95]).isClear, "outside the cube")
    }

    /// The FF-08 counter test. After the materials are made and drawn once,
    /// 120 frames animate the unlit color and the custom material's scale,
    /// alternating materials every frame. Each frame shows its own values,
    /// and the renderer creates no pipeline, depth state, or buffer: the
    /// renderer runs on a ``CountingDevice``, whose log of object-creating
    /// messages is checked as well as the renderer's own counters.
    @Test func `Warmed parameter animation creates no pipelines`() async throws {
        let counting = CountingDevice(wrapping: device)
        let renderer = try makeRenderer(device: counting.device)
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        let mesh = try renderer.makeMesh(.cube())
        var unlit = try renderer.makeUnlitMaterial(color: [1, 1, 1, 1])
        var tint = try renderer.makeMaterial(
            fragmentFunction: "test_tint_fragment", library: library,
            parameters: TintParameters(1, 1, 1, 1, 1, 0, 0, 0)
        )
        let target = try makeTarget(size: 32)
        let pose = CubePose.all[0]
        try await draw(renderer, mesh, unlit, pose, into: target)
        try await draw(renderer, mesh, tint, pose, into: target)
        let warmed = renderer.resourceStatistics
        #expect(warmed.renderPipelinesCreated == 3, "normal diagnostic, unlit, and tint")
        let warmedCreations = counting.creations
        #expect(warmedCreations.filter { $0.hasPrefix("newRenderPipelineState") }.count == 3, "\(warmedCreations)")

        var mismatches: [String] = []
        for frame in 0..<120 {
            let t = Float(frame) / 119
            let expected: SIMD4<Float>
            if frame.isMultiple(of: 2) {
                unlit.parameters.color = [t, 1 - t, 0.5, 1]
                expected = unlit.parameters.color
                try await draw(renderer, mesh, unlit, pose, into: target)
            } else {
                tint.parameters = TintParameters(0.9, 0.6, 0.3, 1, 0.25 + 0.75 * t, 0, 0, 0)
                expected = tint.parameters.lowHalf * tint.parameters[4]
                try await draw(renderer, mesh, tint, pose, into: target)
            }
            let pixel = target.pixel(at: [0, 0])
            if !Self.pixel(pixel, matches: expected) {
                mismatches.append("frame \(frame): expected \(expected), got \(pixel)")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.prefix(5))")
        let createdWhileDrawing = Array(counting.creations.dropFirst(warmedCreations.count))
        #expect(createdWhileDrawing.isEmpty, "drawing created \(createdWhileDrawing.count) objects: \(Set(createdWhileDrawing).sorted())")
        #expect(renderer.resourceStatistics == warmed)
        #expect(renderer.statistics.submitted == 122)
        #expect(renderer.statistics.failed == 0)
    }

    /// Materials that name a prepared function in the same library object
    /// share its pipeline; a new library object is a new cache key.
    @Test func `Making a material again reuses its pipeline`() async throws {
        let renderer = try makeRenderer()
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        _ = try renderer.makeUnlitMaterial(color: [1, 0, 0, 1])
        _ = try renderer.makeMaterial(fragmentFunction: "test_tint_fragment", library: library, parameters: TintParameters())
        let prepared = renderer.resourceStatistics
        #expect(prepared.renderPipelinesCreated == 3)

        let blue = try renderer.makeUnlitMaterial(color: [0, 0, 1, 1])
        let tint = try renderer.makeMaterial(fragmentFunction: "test_tint_fragment", library: library, parameters: TintParameters(repeating: 1))
        #expect(renderer.resourceStatistics == prepared)
        #expect(blue.fragmentFunctionName == "ff_unlit_fragment")
        #expect(tint.fragmentFunctionName == "test_tint_fragment")

        let recompiled = try await MaterialTestShaders.makeLibrary(device: device)
        _ = try renderer.makeMaterial(fragmentFunction: "test_tint_fragment", library: recompiled, parameters: TintParameters())
        #expect(renderer.resourceStatistics.renderPipelinesCreated == prepared.renderPipelinesCreated + 1)
    }

    /// Two frames are in flight with different parameters: the first frame
    /// keeps the values it was encoded with although the material changed
    /// before the GPU ran it.
    @Test func `Parameters are copied when a frame is encoded`() async throws {
        let renderer = try makeRenderer()
        let mesh = try renderer.makeMesh(.cube())
        var material = try renderer.makeUnlitMaterial(color: [1, 0, 0, 1])
        let first = try makeTarget(size: 32), second = try makeTarget(size: 32)
        let pose = CubePose.all[0]

        let firstOutcome = renderer.drawFrame(mesh: mesh, material: material, transform: pose.transform, camera: pose.camera) { first.frameTarget }
        material.parameters.color = [0, 1, 0, 1]
        let secondOutcome = renderer.drawFrame(mesh: mesh, material: material, transform: pose.transform, camera: pose.camera) { second.frameTarget }
        #expect(firstOutcome == .submitted(frame: 1))
        #expect(secondOutcome == .submitted(frame: 2))
        var iterator = completions.makeAsyncIterator()
        for _ in 0..<2 {
            let completion = try #require(await iterator.next())
            #expect(completion.succeeded)
        }
        #expect(Self.pixel(first.pixel(at: [0, 0]), matches: [1, 0, 0, 1]), "first: \(first.pixel(at: [0, 0]))")
        #expect(Self.pixel(second.pixel(at: [0, 0]), matches: [0, 1, 0, 1]), "second: \(second.pixel(at: [0, 0]))")
    }

    // MARK: Helpers

    private func draw<Parameters>(
        _ renderer: MeshRenderer, _ mesh: Mesh, _ material: Material<Parameters>, _ pose: CubePose,
        into target: OffscreenTarget
    ) async throws {
        try await submitAndWait(
            renderer.drawFrame(mesh: mesh, material: material, transform: pose.transform, camera: pose.camera) { target.frameTarget }
        )
    }

    private func coverage<Parameters>(
        _ renderer: MeshRenderer, _ mesh: Mesh, _ material: Material<Parameters>, _ pose: CubePose
    ) async throws -> [Bool] {
        let target = try makeTarget()
        try await draw(renderer, mesh, material, pose, into: target)
        return target.pixels().map { !$0.isClear }
    }

    private func submitAndWait(_ outcome: FrameOutcome) async throws {
        guard case .submitted(let frame) = outcome else {
            Issue.record("Frame was not submitted: \(outcome)")
            return
        }
        var iterator = completions.makeAsyncIterator()
        let completion = try #require(await iterator.next())
        #expect(completion == FrameCompletion(frame: frame, succeeded: true))
    }
}

/// Every way an app's fragment function can break the material contract is
/// reported while the material is made, with a typed error that names the
/// function and the fix, before any frame is encoded.
@Suite(.enabled(if: systemDevice != nil))
struct MaterialErrorTests {
    let device: any MTLDevice

    init() throws {
        device = try #require(systemDevice)
    }

    /// 20 bytes in Swift; the matching MSL struct is padded to 32.
    struct UnpaddedTint: BitwiseCopyable {
        var color: SIMD4<Float>
        var scale: Float
    }

    /// Exactly 4096 bytes, one more than a parameter block may hold.
    struct Oversized: BitwiseCopyable {
        var lanes: (
            SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>,
            SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>,
            SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>,
            SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>
        )
    }

    struct Case: Sendable, CustomTestStringConvertible {
        var function: String
        var parameterLength: Int
        var expected: MaterialError
        /// Whether Metal creates the pipeline before the binding check
        /// rejects it.
        var createsPipeline: Bool
        var testDescription: String { "\(function), \(parameterLength) bytes" }
    }

    static let cases: [Case] = [
        Case(function: "test_missing", parameterLength: 0,
             expected: .functionMissing(name: "test_missing", fragmentFunctions: MaterialTestShaders.fragmentFunctions),
             createsPipeline: false),
        Case(function: "test_vertex", parameterLength: 0,
             expected: .notAFragmentFunction(name: "test_vertex", functionType: .vertex), createsPipeline: false),
        Case(function: "test_kernel", parameterLength: 0,
             expected: .notAFragmentFunction(name: "test_kernel", functionType: .kernel), createsPipeline: false),
        Case(function: "test_texture_fragment", parameterLength: 16,
             expected: .unsupportedBinding(function: "test_texture_fragment", argument: "image", binding: "[[texture(0)]]"),
             createsPipeline: true),
        Case(function: "test_extra_buffer_fragment", parameterLength: 16,
             expected: .unsupportedBinding(function: "test_extra_buffer_fragment", argument: "extra", binding: "[[buffer(3)]]"),
             createsPipeline: true),
        Case(function: "test_writable_fragment", parameterLength: 16,
             expected: .unsupportedBinding(
                function: "test_writable_fragment", argument: "parameters", binding: "[[buffer(0)]] with write access"
             ),
             createsPipeline: true),
        Case(function: "test_tint_fragment", parameterLength: 16,
             expected: .parameterSizeMismatch(function: "test_tint_fragment", shaderBytes: 32, hostBytes: 16),
             createsPipeline: true),
        Case(function: "test_tint_fragment", parameterLength: MemoryLayout<UnpaddedTint>.size,
             expected: .parameterSizeMismatch(function: "test_tint_fragment", shaderBytes: 32, hostBytes: 20),
             createsPipeline: true),
        Case(function: "test_tint_fragment", parameterLength: 0,
             expected: .parametersMissing(function: "test_tint_fragment", shaderBytes: 32), createsPipeline: true),
        Case(function: "test_constant_fragment", parameterLength: 16,
             expected: .parametersNotDeclared(function: "test_constant_fragment", hostBytes: 16), createsPipeline: true),
        Case(function: "test_tint_fragment", parameterLength: MemoryLayout<Oversized>.size,
             expected: .parametersTooLarge(bytes: 4096, limit: 4096), createsPipeline: false),
    ]

    @Test(arguments: cases)
    func `Contract violations throw before any frame is encoded`(testCase: Case) async throws {
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm)
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        let before = renderer.resourceStatistics

        let error = #expect(throws: MaterialError.self) {
            switch testCase.parameterLength {
            case 0:
                _ = try renderer.makeMaterial(fragmentFunction: testCase.function, library: library, parameters: NoMaterialParameters())
            case 16:
                _ = try renderer.makeMaterial(fragmentFunction: testCase.function, library: library, parameters: SIMD4<Float>())
            case MemoryLayout<UnpaddedTint>.size:
                _ = try renderer.makeMaterial(
                    fragmentFunction: testCase.function, library: library,
                    parameters: UnpaddedTint(color: .zero, scale: 0)
                )
            default:
                _ = try renderer.makeMaterial(
                    fragmentFunction: testCase.function, library: library,
                    parameters: Oversized(lanes: (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero,
                                                  .zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero))
                )
            }
        }
        #expect(error == testCase.expected)
        #expect(renderer.statistics.submitted == 0)
        let created = renderer.resourceStatistics.renderPipelinesCreated - before.renderPipelinesCreated
        #expect(created == (testCase.createsPipeline ? 1 : 0))
        #expect(renderer.resourceStatistics.buffersAllocated == before.buffersAllocated)
    }

    /// A fragment input the vertex stage does not write is Metal's own
    /// pipeline error; the description passes Metal's message on and lists
    /// the varyings the vertex stage provides.
    @Test func `An unknown varying reports Metal's reason and the available varyings`() async throws {
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm)
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        let error = #expect(throws: MaterialError.self) {
            try renderer.makeMaterial(
                fragmentFunction: "test_unknown_varying_fragment", library: library, parameters: NoMaterialParameters()
            )
        }
        guard case .pipelineCreationFailed(let function, let reason)? = error else {
            Issue.record("unexpected error: \(String(describing: error))")
            return
        }
        #expect(function == "test_unknown_varying_fragment")
        #expect(reason.contains("user(ff_uv)"), "\(reason)")
        let description = error?.description ?? ""
        #expect(description.contains("user(ff_model_normal)") && description.contains("user(ff_model_position)"))
        #expect(renderer.resourceStatistics.renderPipelinesCreated == 1)
    }

    /// After rejected functions, the same renderer still makes and draws a
    /// valid material: a failure leaves no partial state.
    @Test func `A rejected function leaves the renderer usable`() async throws {
        let (completions, report) = AsyncStream.makeStream(of: FrameCompletion.self)
        let renderer = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm, onFrameCompleted: { report.yield($0) })
        let library = try await MaterialTestShaders.makeLibrary(device: device)
        #expect(throws: MaterialError.self) {
            try renderer.makeMaterial(fragmentFunction: "test_texture_fragment", library: library, parameters: SIMD4<Float>())
        }
        #expect(throws: MaterialError.self) {
            try renderer.makeMaterial(fragmentFunction: "test_tint_fragment", library: library, parameters: SIMD4<Float>())
        }
        // The size check failed after the tint pipeline was cached, so this
        // material reuses it.
        let tint = try renderer.makeMaterial(
            fragmentFunction: "test_tint_fragment", library: library, parameters: TintParameters(0, 1, 0, 1, 1, 0, 0, 0)
        )
        #expect(renderer.resourceStatistics.renderPipelinesCreated == 3, "normal, rejected texture, tint")

        let target = try OffscreenTarget(device: device, width: 32, height: 32, depthPixelFormat: .depth32Float)
        let pose = CubePose.all[0]
        let outcome = renderer.drawFrame(mesh: try renderer.makeMesh(.cube()), material: tint, transform: pose.transform, camera: pose.camera) {
            target.frameTarget
        }
        #expect(outcome == .submitted(frame: 1))
        var iterator = completions.makeAsyncIterator()
        #expect(await iterator.next()?.succeeded == true)
        #expect(target.pixel(at: [0, 0]).dominantChannel == .green)
    }

    /// Each description names the function or value at fault and says what
    /// to change.
    @Test(arguments: [
        (MaterialError.libraryDeviceMismatch(libraryDevice: "GPU A", rendererDevice: "GPU B"), ["GPU A", "GPU B", "Make the library"]),
        (.functionMissing(name: "pulse", fragmentFunctions: ["glow", "tint"]), ["pulse", "glow, tint"]),
        (.functionMissing(name: "pulse", fragmentFunctions: []), ["pulse", "no fragment functions"]),
        (.notAFragmentFunction(name: "move", functionType: .vertex), ["move", "vertex", "`fragment` qualifier"]),
        (.pipelineCreationFailed(function: "glow", reason: "bad input"), ["glow", "bad input", "[[stage_in]]"]),
        (.unsupportedBinding(function: "glow", argument: "image", binding: "[[texture(0)]]"), ["glow", "image", "[[texture(0)]]", "[[buffer(0)]]"]),
        (.parameterSizeMismatch(function: "glow", shaderBytes: 32, hostBytes: 20), ["glow", "32-byte", "20 bytes", "padding"]),
        (.parametersNotDeclared(function: "glow", hostBytes: 16), ["glow", "16 bytes", "[[buffer(0)]]"]),
        (.parametersMissing(function: "glow", shaderBytes: 32), ["glow", "32-byte", "same layout"]),
        (.parametersTooLarge(bytes: 5000, limit: 4096), ["5000", "4096"]),
    ] as [(MaterialError, [String])])
    func `Errors name the problem and the fix`(error: MaterialError, phrases: [String]) {
        for phrase in phrases {
            #expect(error.description.contains(phrase), "\(error.description) lacks \(phrase)")
        }
    }
}

/// Drawing a material with a renderer that did not make it would bind a
/// pipeline built for other formats. The renderer stops before encoding.
@Suite(.enabled(if: systemDevice != nil))
struct MaterialPreconditionTests {
    @Test func `A material from another renderer stops before encoding`() async {
        let result = await #expect(processExitsWith: .failure, observing: [\.standardErrorContent]) {
            guard let device = MTLCreateSystemDefaultDevice() else { return }
            let maker = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm)
            let material = try maker.makeUnlitMaterial(color: [1, 1, 1, 1])
            let renderer = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm)
            let mesh = try renderer.makeMesh(.cube())
            let target = try OffscreenTarget(device: device, width: 16, height: 16, depthPixelFormat: .depth32Float)
            let camera = Camera(position: [0, 0, 3], projection: try PerspectiveProjection(verticalFieldOfView: 1, near: 0.1, far: 10))
            _ = renderer.drawFrame(mesh: mesh, material: material, transform: .identity, camera: camera) { target.frameTarget }
        }
        let message = String(decoding: result?.standardErrorContent ?? [], as: UTF8.self)
        #expect(message.contains("was made by another MeshRenderer"), "stderr: \(message)")
    }
}
