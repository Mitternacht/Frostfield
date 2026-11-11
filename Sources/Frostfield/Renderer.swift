import Foundation
import Metal
import QuartzCore

// Used by the local FrostfieldLab fixture through Swift package access.
package final class TriangleRenderer {
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLRenderPipelineState

    package init(
        device: any MTLDevice,
        pixelFormat: MTLPixelFormat,
        vertexFunctionName: String = "frostfield_triangle_vertex"
    ) throws {
        guard let queue = device.makeCommandQueue() else {
            throw Failure.commandQueue
        }

        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .module)
        } catch {
            throw Failure.library(error)
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Frostfield triangle"
        descriptor.vertexFunction = try Self.function(named: vertexFunctionName, in: library)
        descriptor.fragmentFunction = try Self.function(named: "frostfield_triangle_fragment", in: library)
        descriptor.colorAttachments[0].pixelFormat = pixelFormat

        do {
            self.pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw Failure.pipeline(error)
        }
        self.queue = queue
    }

    package func draw(
        pass: MTLRenderPassDescriptor,
        drawable: any CAMetalDrawable,
        completed: @escaping @Sendable (String?) -> Void
    ) throws {
        guard let buffer = queue.makeCommandBuffer() else {
            throw Failure.commandBuffer
        }
        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw Failure.encoder
        }

        let width = Double(drawable.texture.width)
        let height = Double(drawable.texture.height)
        let side = min(width, height)
        encoder.setViewport(MTLViewport(
            originX: (width - side) / 2, originY: (height - side) / 2,
            width: side, height: side, znear: 0, zfar: 1
        ))
        encoder.setRenderPipelineState(pipeline)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        buffer.addCompletedHandler { buffer in
            if buffer.status == .error {
                completed(buffer.error?.localizedDescription ?? "Metal could not finish the triangle frame.")
            } else {
                completed(nil)
            }
        }
        buffer.present(drawable)
        buffer.commit()
    }

    private static func function(named name: String, in library: any MTLLibrary) throws -> any MTLFunction {
        guard let function = library.makeFunction(name: name) else {
            throw Failure.missingFunction(name)
        }
        return function
    }

    private enum Failure: LocalizedError {
        case commandQueue
        case library(any Error)
        case missingFunction(String)
        case pipeline(any Error)
        case commandBuffer
        case encoder

        var errorDescription: String? {
            switch self {
            case .commandQueue: "Metal could not create a command queue."
            case .library(let error): "Could not load Frostfield’s shader library: \(error.localizedDescription)"
            case .missingFunction(let name): "Frostfield’s shader library does not contain \(name)."
            case .pipeline(let error): "Could not prepare the triangle pipeline: \(error.localizedDescription)"
            case .commandBuffer: "Metal could not create a command buffer."
            case .encoder: "Metal could not create a render encoder."
            }
        }
    }
}
