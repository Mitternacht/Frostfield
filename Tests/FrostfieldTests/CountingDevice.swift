import Foundation
import Metal
import Synchronization
import Testing

/// A stand-in for a Metal device that records every object-creating message
/// sent through it, then forwards the message unchanged to the real device.
///
/// Metal names every method that creates an object `new…`
/// (`newBufferWithLength:options:`, `newRenderPipelineStateWithDescriptor:…`,
/// `newDepthStencilStateWithDescriptor:`, `newTextureWithDescriptor:`, and so
/// on). The wrapper implements none of `MTLDevice` itself: the Objective-C
/// runtime asks it for a forwarding target for each message, and it logs the
/// selector before handing the message to the real device. A renderer created
/// with ``device`` therefore sends all of its device calls through the log.
///
/// This measures what a renderer creates independently of its own
/// `ResourceStatistics`, which only count the creation sites the renderer
/// reports. It does not see objects created through another object's
/// `device` property (a command queue's, for example), nor command buffers
/// and encoders, which a queue and a command buffer create every frame by
/// design.
final class CountingDevice: NSObject, Sendable {
    let wrapped: any MTLDevice
    private let log = Mutex<[String]>([])

    init(wrapping device: any MTLDevice) {
        wrapped = device
    }

    /// The wrapper as an `MTLDevice`. It forwards every message, so it
    /// behaves as the wrapped device; objects it creates report the wrapped
    /// device as theirs.
    var device: any MTLDevice {
        // `any MTLDevice` is a single Objective-C object reference, so this
        // only changes the static type; every call is an Objective-C message.
        unsafeBitCast(self, to: (any MTLDevice).self)
    }

    /// The selectors of every object-creating message so far, in order.
    var creations: [String] {
        log.withLock { $0 }
    }

    override func forwardingTarget(for aSelector: Selector!) -> Any? {
        let name = NSStringFromSelector(aSelector)
        if name.hasPrefix("new") {
            log.withLock { $0.append(name) }
        }
        return wrapped
    }

    override func responds(to aSelector: Selector!) -> Bool {
        super.responds(to: aSelector) || wrapped.responds(to: aSelector)
    }
}

/// The counting device must log creations and nothing else, and must behave
/// as the device it wraps, or the resource tests that rely on it prove
/// nothing.
@Suite(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
struct CountingDeviceTests {
    @Test func `Creations are logged and forwarded; queries are not logged`() throws {
        let real = try #require(MTLCreateSystemDefaultDevice())
        let counting = CountingDevice(wrapping: real)
        let device = counting.device
        #expect(device.name == real.name)
        #expect(device.registryID == real.registryID)
        _ = device.supportsFamily(.apple7)
        #expect(counting.creations.isEmpty)

        let buffer = try #require(device.makeBuffer(length: 64, options: .storageModeShared))
        #expect(buffer.length == 64)
        #expect(buffer.device === real)
        let depthState = device.makeDepthStencilState(descriptor: MTLDepthStencilDescriptor())
        #expect(depthState != nil)
        #expect(counting.creations == ["newBufferWithLength:options:", "newDepthStencilStateWithDescriptor:"])
    }
}
