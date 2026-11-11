// Isolated FF-04 probe. See docs/decisions/metal-backend.md for build and run commands.
import Foundation
import Metal
import Synchronization

struct ProbeError: Error, CustomStringConvertible {
    var description: String
    init(_ description: String) { self.description = description }
}

func required<T>(_ value: T?, _ description: String) throws -> T {
    guard let value else { throw ProbeError(description) }
    return value
}

let side = 256
let pixelFormat: MTLPixelFormat = .bgra8Unorm_srgb
let clear = MTLClearColor(red: 0.012, green: 0.02, blue: 0.035, alpha: 1)
guard CommandLine.arguments.count == 2 else {
    throw ProbeError("Usage: metal4-triangle PATH_TO_TRIANGLE_SHADER")
}
let shaderURL = URL(fileURLWithPath: CommandLine.arguments[1])
let device = try required(MTLCreateSystemDefaultDevice(), "No Metal device")
print("GPU: \(device.name), Metal 4: \(device.supportsFamily(.metal4))")
guard device.supportsFamily(.metal4) else { throw ProbeError("Metal 4 family unsupported") }
let source = try String(contentsOf: shaderURL, encoding: .utf8)
let library = try device.makeLibrary(source: source, options: nil)
let pipelineDesc = MTLRenderPipelineDescriptor()
pipelineDesc.vertexFunction = try required(library.makeFunction(name: "frostfield_triangle_vertex"), "vertex")
pipelineDesc.fragmentFunction = try required(library.makeFunction(name: "frostfield_triangle_fragment"), "fragment")
pipelineDesc.colorAttachments[0].pixelFormat = pixelFormat
let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDesc)

func texture() throws -> any MTLTexture {
    let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: pixelFormat, width: side, height: side, mipmapped: false)
    desc.storageMode = .shared
    desc.usage = .renderTarget
    return try required(device.makeTexture(descriptor: desc), "texture")
}

func read(_ texture: any MTLTexture) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: side * side * 4)
    bytes.withUnsafeMutableBytes {
        texture.getBytes($0.baseAddress!, bytesPerRow: side * 4, from: MTLRegionMake2D(0, 0, side, side), mipmapLevel: 0)
    }
    return bytes
}

let legacyTexture = try texture()
let legacyPass = MTLRenderPassDescriptor()
legacyPass.colorAttachments[0].texture = legacyTexture
legacyPass.colorAttachments[0].loadAction = .clear
legacyPass.colorAttachments[0].storeAction = .store
legacyPass.colorAttachments[0].clearColor = clear
let legacyQueue = try required(device.makeCommandQueue(), "Metal queue")
let legacyBuffer = try required(legacyQueue.makeCommandBuffer(), "Metal command buffer")
let legacyEncoder = try required(legacyBuffer.makeRenderCommandEncoder(descriptor: legacyPass), "Metal render encoder")
legacyEncoder.setRenderPipelineState(pipeline)
legacyEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
legacyEncoder.endEncoding()
legacyBuffer.commit()
legacyBuffer.waitUntilCompleted()
guard legacyBuffer.status == .completed else { throw legacyBuffer.error ?? ProbeError("Metal command buffer failed") }
let baseline = read(legacyTexture)

let queue = try required(device.makeMTL4CommandQueue(), "Metal 4 queue")
let allocator = try required(device.makeCommandAllocator(), "Metal 4 allocator")
let command = try required(device.makeCommandBuffer(), "Metal 4 command buffer")
let actualTexture = try texture()
let residencyDesc = MTLResidencySetDescriptor()
residencyDesc.initialCapacity = 1
let residency = try device.makeResidencySet(descriptor: residencyDesc)
residency.addAllocation(actualTexture)
residency.commit()
queue.addResidencySet(residency)
let pass = MTL4RenderPassDescriptor()
pass.colorAttachments[0].texture = actualTexture
pass.colorAttachments[0].loadAction = .clear
pass.colorAttachments[0].storeAction = .store
pass.colorAttachments[0].clearColor = clear

for frame in 1...4 {
    command.beginCommandBuffer(allocator: allocator)
    let encoder = try required(command.makeRenderCommandEncoder(descriptor: pass), "Metal 4 render encoder")
    encoder.setRenderPipelineState(pipeline)
    encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 3)
    encoder.endEncoding()
    command.endCommandBuffer()

    let semaphore = DispatchSemaphore(value: 0)
    let failure = Mutex<String?>(nil)
    let options = MTL4CommitOptions()
    options.addFeedbackHandler { feedback in
        failure.withLock { $0 = feedback.error?.localizedDescription }
        semaphore.signal()
    }
    queue.commit([command], options: options)
    guard semaphore.wait(timeout: .now() + 5) == .success else { throw ProbeError("Metal 4 completion timeout") }
    if let message = failure.withLock({ $0 }) { throw ProbeError(message) }
    allocator.reset()

    let actual = read(actualTexture)
    let diff = zip(baseline, actual).reduce(into: 0) { if $1.0 != $1.1 { $0 += 1 } }
    guard diff == 0 else { throw ProbeError("Frame \(frame): \(diff) different bytes") }
    print("Frame \(frame): \(actual.count) identical bytes")
}

func pixel(_ bytes: [UInt8], _ x: Int, _ y: Int) -> [UInt8] {
    let i = (y * side + x) * 4
    return Array(bytes[i..<(i + 4)])
}
let center = pixel(baseline, side / 2, side / 2)
let corner = pixel(baseline, 0, 0)
guard center != corner else { throw ProbeError("Triangle was not distinguished from clear color") }
print("Center BGRA \(center), clear BGRA \(corner)")
