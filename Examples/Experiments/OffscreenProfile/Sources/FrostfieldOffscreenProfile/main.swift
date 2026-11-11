import Darwin
import Foundation
import Frostfield
import Metal
import QuartzCore
import simd

private let width = 900
private let height = 608
private let frameInterval = 1.0 / 60

private func require<T>(_ value: T?, _ name: String) throws -> T {
    guard let value else { throw MeshRenderer.Failure.resource(name) }
    return value
}

private struct EffectsFixture {
    let queue: any MTLCommandQueue
    let renderer: MeshRenderer
    let camera: PerspectiveCamera
    let particles: ParticleEmitter
    let grade: ColorGrade
    let color: any MTLTexture
    let destination: any MTLTexture
    let meshPass: MTLRenderPassDescriptor
    let particlePass: MTLRenderPassDescriptor
    let clipFromWorld: simd_float4x4
    let cameraTransform = SceneTransform(translation: SIMD3<Float>(0, 0, 2.8))

    init(device: any MTLDevice) throws {
        queue = try require(device.makeCommandQueue(), "profile command queue")
        renderer = try MeshRenderer(device: device, colorPixelFormat: .rgba16Float, geometry: .cube(), material: .lambert)
        camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 100)
        particles = try ParticleEmitter(queue: queue, colorPixelFormat: .rgba16Float, seed: 42)
        grade = try ColorGrade(device: device, destinationPixelFormat: .bgra8Unorm_srgb)
        clipFromWorld = try camera.projection(aspectRatio: Float(width) / Float(height))
            * simd_inverse(cameraTransform.matrix)

        func makeTexture(_ format: MTLPixelFormat, _ usage: MTLTextureUsage) throws -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false
            )
            descriptor.storageMode = .private
            descriptor.usage = usage
            return try require(device.makeTexture(descriptor: descriptor), "profile texture")
        }

        color = try makeTexture(.rgba16Float, [.renderTarget, .shaderRead])
        destination = try makeTexture(.bgra8Unorm_srgb, .renderTarget)
        meshPass = MTLRenderPassDescriptor()
        meshPass.colorAttachments[0].texture = color
        meshPass.colorAttachments[0].loadAction = .clear
        meshPass.colorAttachments[0].storeAction = .store
        meshPass.colorAttachments[0].clearColor = MTLClearColor(red: 0.012, green: 0.02, blue: 0.035, alpha: 1)
        meshPass.depthAttachment.texture = try makeTexture(.depth32Float, .renderTarget)
        meshPass.depthAttachment.loadAction = .clear
        meshPass.depthAttachment.storeAction = .dontCare
        meshPass.depthAttachment.clearDepth = 1
        particlePass = MTLRenderPassDescriptor()
        particlePass.colorAttachments[0].texture = color
        particlePass.colorAttachments[0].loadAction = .load
        particlePass.colorAttachments[0].storeAction = .store
    }

    func submit(frame: Int) throws -> (cpu: Double, gpu: Double?) {
        let start = CACurrentMediaTime()
        let buffer = try require(queue.makeCommandBuffer(), "profile command buffer")
        let angle = Float(frame % 720) * .pi / 360
        let pitch = simd_quatf(angle: -0.38, axis: SIMD3<Float>(1, 0, 0))
        let yaw = simd_quatf(angle: 0.6 + angle, axis: SIMD3<Float>(0, 1, 0))
        let scene = MeshScene(
            camera: camera, transform: SceneTransform(rotation: yaw * pitch), cameraTransform: cameraTransform,
            materialParameters: SIMD4(-0.4, 0.8, 1, 0.9)
        )
        try particles.encodeUpdate(deltaTime: frame == 0 ? 0 : Float(frameInterval), commandBuffer: buffer)
        try renderer.encode(scene, pass: meshPass, commandBuffer: buffer)
        try particles.encodeDraw(pass: particlePass, clipFromWorld: clipFromWorld, commandBuffer: buffer)
        try grade.encode(source: color, destination: destination, strength: 0.72, commandBuffer: buffer)
        buffer.commit()
        let cpu = CACurrentMediaTime() - start
        buffer.waitUntilCompleted()
        guard buffer.status == .completed else {
            throw buffer.error ?? MeshRenderer.Failure.resource("profile GPU completion")
        }
        let gpu = buffer.gpuStartTime > 0 && buffer.gpuEndTime > buffer.gpuStartTime
            ? buffer.gpuEndTime - buffer.gpuStartTime : nil
        return (cpu, gpu)
    }
}

private func residentBytes() -> UInt64? {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return status == KERN_SUCCESS ? info.resident_size : nil
}

private func summary(_ name: String, _ samples: [Double]) {
    guard !samples.isEmpty else { print("\(name) unavailable"); return }
    let ordered = samples.sorted()
    let percentiles = [0.50, 0.95, 0.99].map { percentile in
        let index = max(0, Int(ceil(Double(ordered.count) * percentile)) - 1)
        return String(format: "%.3f", ordered[index] * 1_000)
    }
    print("\(name) samples=\(ordered.count) p50/p95/p99_ms=\(percentiles.joined(separator: "/"))")
}

private func run() throws {
    let arguments = CommandLine.arguments.dropFirst()
    let quick = arguments.elementsEqual(["--quick"])
    guard quick || arguments.isEmpty else {
        throw MeshRenderer.Failure.resource("expected no arguments or --quick")
    }
    let warmup = quick ? 1.0 : 30.0
    let duration = quick ? 4.0 : 600.0
    let device = try require(MTLCreateSystemDefaultDevice(), "system Metal device")
    let fixture = try EffectsFixture(device: device)
    print("device=\(device.name) size=\(width)x\(height) particles=\(fixture.particles.capacity) seed=42")
    print("warmup_s=\(warmup) observation_s=\(duration) cadence_hz=60")
    var thermalStates = Set<Int>()
    var firstResident: UInt64?
    var peakResident: UInt64?
    var cpu: [Double] = []
    var gpu: [Double] = []
    var overruns = 0
    var frame = 0
    let start = CACurrentMediaTime()
    let measurementStart = start + warmup
    var nextTick = start
    while CACurrentMediaTime() < measurementStart + duration {
        let measuring = CACurrentMediaTime() >= measurementStart
        if measuring && cpu.count.isMultiple(of: 60) {
            thermalStates.insert(ProcessInfo.processInfo.thermalState.rawValue)
            if let resident = residentBytes() {
                if firstResident == nil { firstResident = resident }
                peakResident = max(peakResident ?? resident, resident)
            }
        }
        let timing = try autoreleasepool { try fixture.submit(frame: frame) }
        frame += 1
        nextTick += frameInterval
        let remaining = nextTick - CACurrentMediaTime()
        if measuring {
            cpu.append(timing.cpu)
            if let sample = timing.gpu { gpu.append(sample) }
            if remaining <= 0 { overruns += 1 }
        }
        if remaining > 0 {
            Thread.sleep(forTimeInterval: remaining)
        } else {
            nextTick = CACurrentMediaTime()
        }
    }
    summary("CPU encode and commit", cpu)
    summary("GPU command buffer", gpu)
    print("cadence_overruns=\(overruns) in_flight_max=1 thermal_states=\(thermalStates.sorted())")
    let memory = [firstResident, peakResident, residentBytes()].map { $0.map(String.init) ?? "n/a" }
    print("resident_bytes_start/peak/end=\(memory.joined(separator: "/"))")
}

try run()
