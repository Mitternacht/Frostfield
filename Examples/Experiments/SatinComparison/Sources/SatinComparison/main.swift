import Foundation
import Metal
import simd

// FF-05 comparison. Usage: SatinComparison [output-directory]
// Prints ENV, CHECK, RESULT, and FAIL lines; writes satin.png and minimal.png
// to the output directory (default: a new folder in the temporary directory).

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("FAIL no Metal device")
    exit(1)
}
let outputDirectory = CommandLine.arguments.count > 1
    ? URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    : FileManager.default.temporaryDirectory.appendingPathComponent("SatinComparison-\(UUID().uuidString)")
try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

#if DEBUG
let configuration = "Debug"
#else
let configuration = "Release"
#endif
print("ENV device=\(device.name) deviceType=\(type(of: device)) os=\(ProcessInfo.processInfo.operatingSystemVersionString) configuration=\(configuration) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)")
print("SETTINGS size=\(Fixture.width)x\(Fixture.height) format=bgra8Unorm samples=1 depth=none grid=\(Fixture.gridCells)x\(Fixture.gridCells) indices=uint32")

var failures = 0
@MainActor func check(_ condition: Bool, _ message: String) {
    print("\(condition ? "CHECK ok" : "FAIL") \(message)")
    if !condition { failures += 1 }
}

/// Commits and resumes once the GPU reports completion, without blocking a
/// thread. Returns the GPU interval in microseconds.
func commitAndWait(_ commandBuffer: any MTLCommandBuffer) async -> (succeeded: Bool, gpuMicroseconds: Double) {
    await withCheckedContinuation { continuation in
        commandBuffer.addCompletedHandler { buffer in
            continuation.resume(returning: (buffer.status == .completed, (buffer.gpuEndTime - buffer.gpuStartTime) * 1_000_000))
        }
        commandBuffer.commit()
    }
}

let mesh = Fixture.grid()
let mvp = Fixture.viewProjection * Fixture.modelMatrix

// MARK: Construction (includes run-time shader compilation for both)

var clock = ContinuousClock.now
let minimal: MinimalMeshRenderer
do {
    minimal = try MinimalMeshRenderer(device: device, mesh: mesh)
} catch {
    print("FAIL minimal renderer: \(error)")
    exit(1)
}
let minimalCreate = ContinuousClock.now - clock

guard let stripesURL = Fixture.shaderURL("Stripes/Shaders.metal"), FileManager.default.fileExists(atPath: stripesURL.path) else {
    print("FAIL Satin shader resource missing from the consumer bundle")
    exit(1)
}
clock = ContinuousClock.now
let satin = SatinMeshScene(device: device, mesh: mesh, shaderURL: stripesURL)
let satinCreate = ContinuousClock.now - clock

// MARK: One frame each, exported after completion

func renderOnce(_ encode: (any MTLCommandBuffer, any MTLTexture) throws -> Void) async throws -> (Image, Duration, Double) {
    guard let target = Fixture.makeTarget(device: device), let commandBuffer = queue.makeCommandBuffer() else {
        throw CocoaError(.featureUnsupported)
    }
    let start = ContinuousClock.now
    try encode(commandBuffer, target)
    let result = await commitAndWait(commandBuffer)
    let elapsed = ContinuousClock.now - start
    guard result.succeeded else { throw CocoaError(.executableLoad) }
    // Read only after completion: the texture is ready and no GPU work writes it.
    return (Image(texture: target), elapsed, result.gpuMicroseconds)
}

let (minimalImage, minimalFirstFrame, _) = try await renderOnce { minimal.encode(into: $0, target: $1, modelViewProjection: mvp) }
let (satinImage, satinFirstFrame, _) = try await renderOnce { try satin.encode(into: $0, target: $1) }
try minimalImage.writePNG(to: outputDirectory.appendingPathComponent("minimal.png"))
try satinImage.writePNG(to: outputDirectory.appendingPathComponent("satin.png"))
print("EXPORT \(outputDirectory.path)")

print(String(format: "CREATE minimal_ms=%.3f satin_ms=%.3f first_frame_minimal_ms=%.3f first_frame_satin_ms=%.3f",
             Double(minimalCreate.components.attoseconds) / 1e15 + Double(minimalCreate.components.seconds) * 1000,
             Double(satinCreate.components.attoseconds) / 1e15 + Double(satinCreate.components.seconds) * 1000,
             Double(minimalFirstFrame.components.attoseconds) / 1e15 + Double(minimalFirstFrame.components.seconds) * 1000,
             Double(satinFirstFrame.components.attoseconds) / 1e15 + Double(satinFirstFrame.components.seconds) * 1000))

// MARK: Correctness

let difference = ImageDifference(minimalImage, satinImage)
print("DIFF \(difference)")
let covered = minimalImage.coveredPixels
check(covered > Fixture.width * Fixture.height / 10, "minimal image covers \(covered) pixels")
check(satinImage.coveredPixels > Fixture.width * Fixture.height / 10, "satin image covers \(satinImage.coveredPixels) pixels")
// The transformed center (0.1, -0.05) has uv (0.5, 0.5): red ≈ green ≈ 128.
for (name, image) in [("minimal", minimalImage), ("satin", satinImage)] {
    let center = image.rgb(clipX: 0.1, clipY: -0.05)
    check(abs(center.x - 128) <= 8 && abs(center.y - 128) <= 8, "\(name) center uv≈(0.5,0.5): rgb=\(center)")
    check(image.rgb(clipX: -0.9, clipY: 0.9) == .zero, "\(name) corner outside the mesh is clear")
}
// Rotation: the grid's uv (1, 1) corner, model (0.5, 0.5) → rotated 30° and scaled.
let corner = Fixture.modelMatrix * SIMD4<Float>(0.45, 0.45, 0, 1)
for (name, image) in [("minimal", minimalImage), ("satin", satinImage)] {
    let rgb = image.rgb(clipX: corner.x, clipY: corner.y)
    check(rgb.x > 220 && rgb.y > 220, "\(name) near transformed uv(0.95,0.95): rgb=\(rgb)")
}
check(Double(difference.differingPixels) / Double(difference.pixelCount) < 0.01, "images agree on at least 99% of pixels")

// MARK: Shader-failure path

do {
    let brokenDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("SatinBroken-\(UUID().uuidString)/Broken")
    try FileManager.default.createDirectory(at: brokenDirectory, withIntermediateDirectories: true)
    let brokenURL = brokenDirectory.appendingPathComponent("Shaders.metal")
    try "vertex float4 brokenVertex( this is not MSL".write(to: brokenURL, atomically: true, encoding: .utf8)
    let broken = SatinMeshScene(device: device, mesh: mesh, shaderURL: brokenURL, label: "Broken")
    do {
        let (image, _, _) = try await renderOnce { try broken.encode(into: $0, target: $1) }
        print("SHADER_FAILURE satin draw did not throw; covered_pixels=\(image.coveredPixels) pipeline_error=\(broken.material.shader?.pipelineError.map { String(describing: $0).prefix(160) } ?? "nil")")
    } catch {
        print("SHADER_FAILURE satin draw threw: \(error)")
    }
    do {
        _ = try MinimalMeshRenderer(device: device, mesh: mesh, minimalSourceName: "Missing.metal")
        print("SHADER_FAILURE minimal renderer accepted a missing source")
    } catch {
        print("SHADER_FAILURE minimal renderer threw: \(error)")
    }
}

// MARK: Timing (paced: one frame in flight, as in requested drawing)

guard let timingTarget = Fixture.makeTarget(device: device) else { exit(1) }
@MainActor func measure(_ name: String, round: Int, frames: Int = 500, warmup: Int = 50, _ encode: (any MTLCommandBuffer) throws -> Void) async throws {
    var cpu: [Double] = [], gpu: [Double] = []
    for index in 0..<(warmup + frames) {
        guard let commandBuffer = queue.makeCommandBuffer() else { return }
        let start = ContinuousClock.now
        try encode(commandBuffer)
        let encoded = ContinuousClock.now - start
        let result = await commitAndWait(commandBuffer)
        if !result.succeeded { failures += 1 }
        if index >= warmup {
            cpu.append(Double(encoded.components.attoseconds) / 1e12 + Double(encoded.components.seconds) * 1e6)
            gpu.append(result.gpuMicroseconds)
        }
    }
    func p(_ values: [Double], _ q: Double) -> Double {
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, max(0, Int((q * Double(sorted.count)).rounded(.up)) - 1))]
    }
    print(String(format: "RESULT round=%d path=%@ cpu_encode_us p50=%.1f p95=%.1f p99=%.1f gpu_us p50=%.1f p95=%.1f p99=%.1f frames=%d",
                 round, name, p(cpu, 0.5), p(cpu, 0.95), p(cpu, 0.99), p(gpu, 0.5), p(gpu, 0.95), p(gpu, 0.99), frames))
}
for round in 1...3 {
    let order = round.isMultiple(of: 2) ? ["satin", "minimal"] : ["minimal", "satin"]
    for path in order {
        if path == "minimal" {
            try await measure(path, round: round) { minimal.encode(into: $0, target: timingTarget, modelViewProjection: mvp) }
        } else {
            try await measure(path, round: round) { try satin.encode(into: $0, target: timingTarget) }
        }
    }
}

print(failures == 0 ? "DONE ok" : "DONE failures=\(failures)")
exit(failures == 0 ? 0 : 1)
