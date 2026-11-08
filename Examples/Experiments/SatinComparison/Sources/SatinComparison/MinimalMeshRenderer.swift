import Foundation
import Metal
import simd

/// The minimal renderer for the comparison: one pipeline, one vertex buffer,
/// one index buffer, one indexed draw. The shader source is
/// Shaders/StripesColor.h followed by Shaders/Minimal.metal, compiled at
/// run time like Satin's.
final class MinimalMeshRenderer {
    enum Failure: Error, CustomStringConvertible {
        case resourceMissing(String)
        case shaderCompilation(String)
        case functionMissing(String)
        case pipeline(String)
        case buffer

        var description: String {
            switch self {
            case .resourceMissing(let path): "Shader resource missing: \(path)"
            case .shaderCompilation(let message): "Shader compilation failed: \(message)"
            case .functionMissing(let name): "Shader function missing: \(name)"
            case .pipeline(let message): "Pipeline creation failed: \(message)"
            case .buffer: "Buffer allocation failed"
            }
        }
    }

    private let pipeline: any MTLRenderPipelineState
    private let vertices: any MTLBuffer
    private let indices: any MTLBuffer
    private let indexCount: Int

    init(device: any MTLDevice, mesh: Fixture.Mesh, minimalSourceName: String = "Minimal.metal") throws(Failure) {
        let source: String
        do {
            guard let colorURL = Fixture.shaderURL("StripesColor.h") else { throw Failure.resourceMissing("StripesColor.h") }
            guard let bodyURL = Fixture.shaderURL(minimalSourceName) else { throw Failure.resourceMissing(minimalSourceName) }
            let body = try String(contentsOf: bodyURL, encoding: .utf8)
            let color = try String(contentsOf: colorURL, encoding: .utf8)
            // makeLibrary(source:) has no file context for #include, so the
            // shared header is spliced in after the standard includes.
            source = body.replacingOccurrences(of: "using namespace metal;", with: "using namespace metal;\n" + color)
        } catch let failure as Failure {
            throw failure
        } catch {
            throw .resourceMissing(String(describing: error))
        }

        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw .shaderCompilation(String(describing: error))
        }
        guard let vertex = library.makeFunction(name: "minimalVertex") else { throw .functionMissing("minimalVertex") }
        guard let fragment = library.makeFunction(name: "minimalFragment") else { throw .functionMissing("minimalFragment") }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = Fixture.pixelFormat
        do {
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw .pipeline(String(describing: error))
        }

        // MinimalVertex: float4 position, float2 texcoord, float2 padding (32 bytes).
        var packed: [SIMD4<Float>] = []
        for (position, texcoord) in zip(mesh.positions, mesh.texcoords) {
            packed.append(SIMD4(position, 1))
            packed.append(SIMD4(texcoord.x, texcoord.y, 0, 0))
        }
        guard let vertices = device.makeBuffer(bytes: packed, length: packed.count * MemoryLayout<SIMD4<Float>>.stride),
              let indices = device.makeBuffer(bytes: mesh.indices, length: mesh.indices.count * MemoryLayout<UInt32>.stride)
        else { throw .buffer }
        self.vertices = vertices
        self.indices = indices
        indexCount = mesh.indices.count
    }

    func encode(into commandBuffer: any MTLCommandBuffer, target: any MTLTexture, modelViewProjection: simd_float4x4) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        let c = Fixture.clearColor
        pass.colorAttachments[0].clearColor = MTLClearColor(red: Double(c.x), green: Double(c.y), blue: Double(c.z), alpha: Double(c.w))
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        var mvp = modelViewProjection
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vertices, offset: 0, index: 0)
        encoder.setVertexBytes(&mvp, length: MemoryLayout<simd_float4x4>.stride, index: 1)
        encoder.drawIndexedPrimitives(type: .triangle, indexCount: indexCount, indexType: .uint32, indexBuffer: indices, indexBufferOffset: 0)
        encoder.endEncoding()
    }
}
