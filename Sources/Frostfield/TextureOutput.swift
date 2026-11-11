import Foundation
import Metal
import Synchronization

/// A bounded pool of offscreen frames. Both producer and consumer command
/// buffers are submitted on the queue supplied at initialization.
public final class TextureOutput: Sendable {
    private let queue: any MTLCommandQueue
    private let capacity: Int
    private let state = Mutex(PoolState())

    public init(queue: any MTLCommandQueue, capacity: Int) throws {
        guard capacity > 0 else { throw Failure.invalidCapacity }
        self.queue = queue
        self.capacity = capacity
    }

    public var device: any MTLDevice { queue.device }

    /// The callback must encode a full write to the texture and end its encoders
    /// before returning. Borrow the texture and command buffer for the callback;
    /// TextureOutput owns submission. Returns nil while every slot is leased.
    public func produce(
        width: Int,
        height: Int,
        encode: (any MTLTexture, any MTLCommandBuffer) throws -> Void
    ) throws -> Frame? {
        guard (1...16_384).contains(width), (1...16_384).contains(height) else {
            throw Failure.invalidDimensions
        }
        guard let lease = try reserve(width: width, height: height) else { return nil }
        guard let buffer = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        let hold = CompletionHold(lease)
        buffer.addCompletedHandler { buffer in
            guard let lease = hold.take() else { return }
            let readiness: Frame.Readiness = buffer.status == .completed
                ? .ready
                : .failed(buffer.error?.localizedDescription ?? "Metal failed to produce a texture frame.")
            lease.readiness.withLock { $0 = readiness }
        }
        try encode(lease.backing.texture, buffer)
        guard buffer.status == .notEnqueued else { throw Failure.bufferAlreadySubmitted }
        buffer.commit()
        return Frame(lease: lease)
    }

    /// Encodes a read-only use of a ready frame. Retain the Frame to keep it
    /// valid for another read; after its release, submitted reads still hold
    /// the slot through GPU completion. Older retained frames remain valid.
    /// Borrow the texture and buffer only for this callback and end all
    /// encoders before returning. TextureOutput owns queue and submission.
    public func consume(
        _ frame: Frame,
        completed: (@Sendable (String?) -> Void)? = nil,
        encode: (any MTLTexture, any MTLCommandBuffer) throws -> Void
    ) throws {
        let lease = frame.lease
        guard lease.owner === self else { throw Failure.foreignFrame }
        switch frame.readiness {
        case .pending: throw Failure.frameNotReady
        case .failed(let reason): throw Failure.producerFailed(reason)
        case .ready: break
        }
        guard let buffer = queue.makeCommandBuffer() else { throw Failure.commandBuffer }
        let hold = CompletionHold(lease)
        buffer.addCompletedHandler { buffer in
            _ = hold.take()
            completed?(
                buffer.status == .completed
                    ? nil
                    : buffer.error?.localizedDescription ?? "Metal failed to consume a texture frame."
            )
        }
        try encode(lease.backing.texture, buffer)
        guard buffer.status == .notEnqueued else { throw Failure.bufferAlreadySubmitted }
        buffer.commit()
    }

    /// Linear sRGB, premultiplied RGBA16Float. Normalized UV (0, 0) is the
    /// upper-left of the full texture; (1, 1) is its lower-right edge.
    public final class Frame: Sendable {
        public enum Readiness: Sendable, Equatable {
            case pending
            case ready
            case failed(String)
        }

        fileprivate let lease: Lease

        fileprivate init(lease: Lease) {
            self.lease = lease
        }

        /// Increasing within this output in reservation order. Skipped submissions may leave gaps.
        public var id: UInt64 { lease.id }
        public var producerTime: ContinuousClock.Instant { lease.producerTime }
        public var width: Int { lease.backing.width }
        public var height: Int { lease.backing.height }
        public var pixelFormat: MTLPixelFormat { .rgba16Float }
        public var device: any MTLDevice { lease.owner.device }
        public var readiness: Readiness { lease.readiness.withLock { $0 } }
    }

    public enum Failure: LocalizedError, Equatable {
        case invalidCapacity
        case invalidDimensions
        case texture
        case frameIdentityExhausted
        case commandBuffer
        case bufferAlreadySubmitted
        case foreignFrame
        case frameNotReady
        case producerFailed(String)

        public var errorDescription: String? {
            switch self {
            case .invalidCapacity: "Texture output capacity must be greater than zero."
            case .invalidDimensions: "Texture output dimensions must be between 1 and 16,384 pixels."
            case .texture: "Metal could not allocate the output texture."
            case .frameIdentityExhausted: "Texture output ran out of frame identities."
            case .commandBuffer: "Metal could not create the texture output command buffer."
            case .bufferAlreadySubmitted: "Texture output owns command buffer submission."
            case .foreignFrame: "The frame belongs to a different output and cannot be consumed here."
            case .frameNotReady: "The GPU has not finished producing this frame."
            case .producerFailed(let reason): "Texture production failed: \(reason)"
            }
        }
    }

    private func reserve(width: Int, height: Int) throws -> Lease? {
        try state.withLock { state in
            let index = state.slots.firstIndex {
                !$0.leased && $0.backing.width == width && $0.backing.height == height
            } ?? state.slots.firstIndex { !$0.leased } ?? state.slots.count
            guard index < capacity else { return nil }
            guard state.nextID < UInt64.max else { throw Failure.frameIdentityExhausted }

            if index == state.slots.count ||
                state.slots[index].backing.width != width || state.slots[index].backing.height != height {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false
                )
                descriptor.storageMode = .private
                descriptor.usage = [.renderTarget, .shaderRead]
                descriptor.hazardTrackingMode = .tracked
                guard let texture = queue.device.makeTexture(descriptor: descriptor) else { throw Failure.texture }
                texture.label = "Frostfield texture output"
                let slot = Slot(backing: Backing(texture: texture, width: width, height: height))
                if index == state.slots.count {
                    state.slots.append(slot)
                } else {
                    state.slots[index] = slot
                }
            }
            state.slots[index].leased = true
            state.nextID += 1
            return Lease(
                owner: self, index: index, id: state.nextID,
                producerTime: ContinuousClock.now, backing: state.slots[index].backing
            )
        }
    }

    private func release(_ index: Int) {
        state.withLock { $0.slots[index].leased = false }
    }

    private struct PoolState {
        var slots: [Slot] = []
        var nextID: UInt64 = 0
    }

    private struct Slot {
        let backing: Backing
        var leased = false
    }

    // This immutable handle only exposes the underlying Metal texture through
    // the pool's borrowing callbacks. Lease ownership protects GPU access.
    fileprivate final class Backing: @unchecked Sendable {
        let texture: any MTLTexture
        let width: Int
        let height: Int

        init(texture: any MTLTexture, width: Int, height: Int) {
            self.texture = texture
            self.width = width
            self.height = height
        }
    }

    fileprivate final class Lease: Sendable {
        let owner: TextureOutput
        let index: Int
        let id: UInt64
        let producerTime: ContinuousClock.Instant
        let backing: Backing
        let readiness = Mutex(Frame.Readiness.pending)

        init(owner: TextureOutput, index: Int, id: UInt64, producerTime: ContinuousClock.Instant, backing: Backing) {
            self.owner = owner
            self.index = index
            self.id = id
            self.producerTime = producerTime
            self.backing = backing
        }

        deinit { owner.release(index) }
    }

    // Release at completion even if Metal retains a buffer's handler afterward.
    private final class CompletionHold: Sendable {
        private let lease: Mutex<Lease?>

        init(_ lease: Lease) {
            self.lease = Mutex(lease)
        }

        func take() -> Lease? {
            lease.withLock {
                let value = $0
                $0 = nil
                return value
            }
        }
    }
}
