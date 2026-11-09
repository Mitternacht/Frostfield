/// Counters for the GPU objects a renderer created.
///
/// Every pipeline, depth-stencil state, and buffer the renderer creates is
/// counted here, at the place the renderer creates it: its initializer,
/// `MeshRenderer.makeMesh(_:)`, and making a material. Drawing with new
/// transforms, cameras, material parameters, or other per-frame values writes
/// into existing per-frame buffers or the frame's commands instead.
///
/// The counters are the renderer's own report. A host can show them to
/// confirm that it does not make meshes or materials during steady-state
/// drawing; they cannot show what the drawing code itself does. Frostfield's
/// package tests check that separately, by logging every object-creating
/// call the renderer sends to its device.
public struct ResourceStatistics: Sendable, Equatable {
    /// Render pipeline states created from the packaged shaders or from
    /// app-supplied material functions, including any a material check
    /// rejected after Metal created it.
    public var renderPipelinesCreated = 0
    /// Depth-stencil states created.
    public var depthStencilStatesCreated = 0
    /// Metal buffers allocated, including mesh vertex and index buffers.
    public var buffersAllocated = 0
    /// The total requested length of those buffers, in bytes.
    public var bufferBytesAllocated = 0

    public init() {}
}
