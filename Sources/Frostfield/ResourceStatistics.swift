/// Counters for the GPU objects a renderer created.
///
/// Every pipeline and buffer the renderer creates is counted here when it is
/// created. Drawing with new transforms, cameras, or other per-frame values
/// writes into existing per-frame buffers instead, so these counters stay
/// constant across such updates. A host can show them to confirm that
/// steady-state drawing creates nothing.
public struct ResourceStatistics: Sendable, Equatable {
    /// Render pipeline states compiled from the packaged shaders.
    public var renderPipelinesCreated = 0
    /// Metal buffers allocated.
    public var buffersAllocated = 0
    /// The total requested length of those buffers, in bytes.
    public var bufferBytesAllocated = 0

    public init() {}
}
