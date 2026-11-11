# Scene GPU timing

`SceneGraph.encodeTimed` instruments the regular Metal color and shadow
passes for the graph camera or an explicit raster view. It returns vertex
and fragment intervals and the span of the available stages, in seconds.
Reports follow encoder order. Directional cascades, punctual shadow slices,
and the opaque and transmission color passes each have a separate label.
Draw statistics come from the same encode.
[Apple GPU counter sampling](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers) ·
[Three.js r187 renderer](https://github.com/mrdoob/three.js/blob/r187/src/renderers/common/Renderer.js)

```swift
if SceneGPUTimingFrame.isSupported(on: command.device) {
    let frame = try graph.encodeTimed(pass: pass, commandBuffer: command)
    command.commit()
    command.waitUntilCompleted()
    let measurements = try frame.results()
}
```

The device must expose stage-boundary sampling and the timestamp counter
set. Each encode owns one bounded sample buffer, with four indices per
pass. The caller's color descriptor is copied before assigning counter
attachment zero. The encoder owns the fresh shadow descriptors. Samples
are retained through completion even when a command buffer uses
unretained references. The path tracer and raster timing use the same
GPU-to-CPU clock calibration.

Intervals measure GPU work within the sampled stages. They exclude CPU
encoding, queue wait, transmission blits and uninstrumented passes.
A missing stage has no interval. Read the frame after completion. A
pending command or invalid calibration reports a typed error.
[Metal CPU and GPU timestamp samples](https://developer.apple.com/documentation/metal/mtldevice/sampletimestamps())

`SceneGPUTimingTests` compares timed color output with the ordinary path,
checks pending and occupied-counter handling, and measures separate
directional, punctual, opaque and transmission passes. An unsupported
device test verifies that the ordinary encode still renders.

Xcode 27.2 passed the 50 focused timing and raster tests on an M4 Max with
Metal API Validation. The combined package at `969c199` passed 1,261 tests
with five optional cases skipped. On iPhone 18 Pro Simulator, the explicit
stage test skipped because stage-boundary counters were unavailable. The
capability test and ordinary raster path passed.

## Compute timing

`ShaderTextureGenerator.encodeTimed` and `ShaderBufferKernel.encodeTimed`
measure the graph's compute encoder. For a custom Metal dispatch,
`ComputeGPUTiming.encode` creates and ends a serial compute encoder around
the supplied commands. The caller submits the command buffer and reads
`duration()` in GPU seconds after completion.

```swift
if ComputeGPUTiming.isSupported(on: command.device) {
    let timing = try generator.encodeTimed(into: texture, commandBuffer: command)
    command.commit()
    command.waitUntilCompleted()
    let seconds = try timing.duration()
}
```

Each timing owns two stage-boundary samples and the handler retains the
counter buffer until completion, including with unretained commands.
Separate encoders on the same command buffer have separate results. The
interval excludes CPU encoding, queue wait, and separate encoders such
as texture mipmap generation. Unsupported counters, a pending command,
and invalid results produce typed errors.
[Metal compute counter attachments](https://developer.apple.com/documentation/metal/mtlcomputepasssamplebufferattachmentdescriptor)

`ComputeGPUTimingTests` verifies GPU intervals for graph texture and
buffer updates on the same command and for a custom Metal kernel. The
texture and buffer data are also checked, including the final generated
mip. The capability branch exercises ordinary graph output when timing
is unavailable. An iPhone Simulator probe found that direct texture
readback could return base-level pixels for lower mip requests. The mip
fixtures use an explicit Metal blit for each selected level and retain
the original texel and averaging assertions.

The iPhone 18 Pro Simulator run at `ad18bca` passed 83 focused tests with
three capability skips. It verified that graph output and lower mips remain
correct without timestamp counters, and that the path tracer reports their
absence. The custom compute and explicit raster timing fixtures require
native stage-boundary counters.

The combined macOS run passed 1,278 tests, including the native graph and
custom Metal timing cases, with five optional cases skipped. The Vision Pro
Simulator passed 14 focused tests, with the custom native counter test
skipped. Both runs used Metal API Validation.

## Render passes and post-process chains

`RenderGPUTiming.encode` samples a caller's Metal render encoder and
`PostProcessPass.encodeTimed` applies the same instrumentation to a prepared
graph. Both expose vertex and fragment intervals and their overall span.
The generic encoder copies the caller's render descriptor and uses counter
attachment zero. Read `results()` after command-buffer completion. Each
render encoder owns its samples until Metal completes, including for
unretained command buffers. The path tracer keeps its reused sample buffer
claimed until its completion handler resolves it.
[Metal render counter attachments](https://developer.apple.com/documentation/metal/mtlrenderpasssamplebufferattachmentdescriptor)

```swift
let seconds = try await chain.renderTimed(source: source, destination: displayTexture)
```

`PostProcessChain.renderTimed` submits only the chain's work in a retaining
command buffer on its queue. The async result is GPU end time minus start
time in seconds. It includes scaled passes, built-in and caller-supplied
effects, and the copy used when every effect is bypassed. Commit source
producers on that queue first. Metal can return no execution timestamps, in
which case the method reports `timingUnavailable` after completing the draw.
[Metal command-buffer GPU time](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime) ·
[Three.js r187 timestamp query pool](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgpu/utils/WebGPUTimestampQueryPool.js)

`RenderGPUTimingTests` verifies per-encoder intervals, pixel equivalence,
descriptor ownership, an occupied counter attachment, and reads before
completion. The async chain test compares a scaled bloom and grading chain
against an ordinary encode and verifies the fully bypassed copy.

The combined package at `5760918` passed 1,313 macOS tests. Native stage
intervals were verified on an M4 Max, and dedicated command-buffer timing
passed on the iPhone 18 Pro and Vision Pro simulators. All three runs used
Metal API Validation.

## Frame timing batches

Create one `GPUTimingBatch` for each frame to collect captures from the scene,
post-process graphs, shader graph dispatches and custom Metal encoders. Track
their handles, submit the command buffers, then await `resolve()`. The batch
returns separate render and compute totals in seconds, with measured and
unavailable pass counts. Totals sum pass spans, so overlapping work can add up
to more than the frame's wall-clock duration.

```swift
var batch = GPUTimingBatch()
try batch.track(try graph.encodeTimed(pass: scenePass, commandBuffer: graphics))
try batch.track(try generator.encodeTimed(into: texture, commandBuffer: compute))
graphics.commit()
compute.commit()
let timings = try await batch.resolve()
```

The caller defines the frame boundary and can query the same batch again.
Each batch tracks up to 1,024 captures. Repeated handles count once, and all
captures must be submitted before resolution. A discarded command buffer
reports a distinct error. The batch waits for the Metal callback to read
the counters even when the command buffer already reports completion.
Individual timing handles retain their synchronous result access.
[Three.js r187 timestamp query pool](https://github.com/mrdoob/three.js/blob/r187/src/renderers/webgpu/utils/WebGPUTimestampQueryPool.js)

`GPUTimingBatchTests` tracks actual scene, post-process and compute encoders
across command buffers. It compares the group totals with the individual
queries and checks duplicates, concurrent reads, and command-buffer lifecycle
errors. An empty batch has zero measured passes. The M4 Max probe observed a
completion callback reporting `.notEnqueued` when an unsubmitted buffer was
released. The batch treats that state as a discard and preserves errors from
submitted GPU work.

The combined macOS run at `b423a0a` passed 1,337 tests with five optional
cases skipped, including all three batch tests. The iPhone and Vision Pro
simulator runs passed the empty-batch case and skipped the two batch tests
that require hardware stage-boundary counters.
