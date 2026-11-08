# Decision: renderer foundation, minimal renderer versus Satin (FF-05)

Date: 2026-11-08 · Task: FF-05 · Status: **keep the minimal Frostfield renderer;
do not adopt Satin.** The UIKit-host presentation and device traces were not done.

## Question

Does a maintained graphics dependency speed up Frostfield while preserving shader
freedom, predictable resources, native hosting, and packaging? See the scorecard in
[research spikes §2](../research-spikes.md).

## Decision

Build FF-06 to FF-08 on Frostfield's own minimal Metal renderer. Satin did the narrow
job: it built in a clean Swift 6 consumer without a fork and produced a byte-identical
image. But it brings no capability the v0.1 scope needs, and it costs 2.2 times the
CPU encode time for one mesh and roughly 60,000 lines of dependency. It also reports
shader-compilation failures only through console output and a polled property, which
breaks Frostfield's explicit-failure rule. The comparison code stays in an isolated
experiment package
([`Examples/Experiments/SatinComparison`](../../Examples/Experiments/SatinComparison/README.md)).
Nothing from it enters Frostfield's manifest, library, lab app, or public API.

## Candidate inspected before building

| Item | Finding |
| --- | --- |
| Repository | [Fabric-Project/Satin](https://github.com/Fabric-Project/Satin), a maintained fork of the archived Hi-Rez/Satin |
| Revision | `ff825e4945ca19277355eda69831b9b19aad1533` (`main`, committed 2026-11-06; `git describe`: `20.0.0-Beta-1-119-gff825e49`). The latest stable tag is `13.0.0` (2023-05-10), and `20.0.0-Beta-1` dates from 2026-06-07. |
| Manifest | Swift tools 6.2, `swiftLanguageModes: [.v5]`, C++17. Platforms macOS 15, iOS 18, visionOS 2. No package dependencies. Products `Satin`, `SatinCore`, and a dynamic `Satin Dynamic`. |
| Build-tool plugins, binary targets, scripts | **None in the manifest.** There is no `Plugins/` directory and no binary artifacts (`.a`, `.dylib`, `.xcframework`, `.metallib`, archives) anywhere in the tree. `Scripts/build_xcframework` is a manual release script that SwiftPM never runs. Because the package is plain source, the build went ahead. |
| Contents | `SatinCore`: 13 Objective-C++ files and 14 headers (6,921 lines). `Satin`: 282 Swift files (44,674 lines). Resources copied verbatim: 173 MSL files (8,239 lines, compiled at run time) and 64 KB of textures. 3.4 MB of sources in total. |
| License | MIT, "Copyright (c) 2025 Hi-Rez". The bundled shader library also carries third-party notices: `Pipelines/Library/Colors.metal` is `SPDX-License-Identifier: Apache-2.0`, `Cellular.metal` and `Blend.metal` (adapted from glsl-blend) are MIT, and the motion-blur shaders credit MIT-licensed work. These files ship in `Satin_Satin.bundle`. An app that ships Satin must reproduce the MIT notice and the Apache-2.0 attribution for that file. |
| Maintenance | 285 commits from 4 authors since 2026-01-01; last commit 2026-11-06. Active, but `main` is between a 2026 beta tag and a 2023 stable tag, so an adopter would pin unreleased revisions. |

## Fixture and environment

| Item | Value |
| --- | --- |
| Mesh | A 4 × 4 grid over [-0.5, 0.5]²: 25 vertices, 96 `uint32` indices, counter-clockwise, culling off |
| Transform | translate (0.1, -0.05, 0) · rotate 30° about z · scale (1.2, 0.8, 1) |
| Camera | Orthographic, -1 to 1 on both axes, near 0.1, far 10, at z = 2 looking at the origin |
| Shading | Custom MSL (`StripesColor.h`, shared byte for byte): UV as red and green, six blue bands |
| Target | 512 × 512 `bgra8Unorm`, shared storage, one sample, no depth, cleared to opaque black |
| Minimal path | Plain `MTLRenderPipelineState`, vertex and index buffers, one `drawIndexedPrimitives`. Source compiled at run time with `makeLibrary(source:)`. |
| Satin path | `Context`, `RenderEncoder.draw(frame:renderPassDescriptor:commandBuffer:scene:camera:renderTarget:)`, `OrthographicCamera`, `Mesh`, `Geometry` with an `ElementBuffer`, and `SourceMaterial` (label `Stripes`) loading `Stripes/Shaders.metal` from the consumer's resource bundle |
| Machine | Mac with Apple M5, macOS 27.2 (26B123), Xcode 27.2 toolchain (Swift 6.4.2), Release build, Metal API validation off. Thermal state was "fair" (1) during all three runs. |
| Build | `swift build -c release` in the isolated package. Not run through the Xcode MCP bridge (see "Tooling limitation"). |

## Results

**Clean Swift 6 consumer build:** `swift package resolve` took 7.4 s and produced
`Package.resolved` pinned to the inspected revision. The checkout was byte-identical
to the inspected clone (`diff -rq`). A clean `swift build -c release` (Satin, SatinCore,
and the consumer) took 32.2 s wall, 61 s user. It produced no diagnostics in the
Swift 6 consumer. Satin's types are non-`Sendable` classes, and the consumer uses them
only on the main actor, so this narrow use did not exercise the concurrency boundary
further. The release executable is 6.3 MB (Satin linked statically), and
`Satin_Satin.bundle` is 884 KB.

**Output:** three runs. In each, `DIFF differing_pixels=0 (0.0000%) max_channel_delta=0`.
Both images cover 62,859 pixels. The transformed centre reads rgb (128, 128, 0) in both,
the transformed uv (0.95, 0.95) corner reads (243, 242, 255) in both, and an outside
corner is clear. The PNG exports were inspected visually: a rotated, scaled grid with
UV colors, y up, and twelve alternating bands. The PNGs were kept out of the
repository; they are reproducible with the command in the package README.

**Exported texture readiness and lifetime:** Satin encodes into a command buffer and
render target that the caller supplies. Readiness is therefore the caller's
command-buffer completion, and the consumer reads the texture only from its completion
handler. The texture's lifetime belongs to the caller. This is the best part of Satin's
design for Frostfield's future texture-export contract. Not tested: reuse of one target
across overlapping frames (read-before-write hazards with more than one frame in
flight).

**Shader failure path:** for a `SourceMaterial` with invalid MSL, Satin printed the
compiler errors and the absolute shader path to standard output. `draw` **did not
throw**, the frame drew nothing (0 covered pixels), and the error was only reachable
afterwards through `material.shader?.pipelineError`. The minimal renderer throws a
typed error before any work is encoded. A Satin-backed Frostfield would need to check
`pipelineError` after every material change and suppress Satin's console output. It
prints file paths, which apps should not log.

**Timing (paced, one frame in flight, 50 warm-up plus 500 frames, 3 rounds × 3 runs,
order alternating):**

| Statistic (µs) | Minimal renderer | Satin |
| --- | --- | --- |
| CPU encode p50 | 5.1–5.8 | 11.5–12.6 |
| CPU encode p95 | 5.8–7.7 | 12.5–79.9 |
| CPU encode p99 | 8.5–17.8 | 16.1–135.3 |
| GPU p50 | 13.0–13.8 | 13.0–14.4 |
| GPU p95 | 15.0–92.0 | 15.0–84.8 |
| GPU p99 | 178.6–559.3 | 175.4–634.5 |

CPU encode covers the renderer's own encode call: for Satin, scene traversal, packet
recording, and sealing; for the minimal path, one encoder and one draw. GPU time is
`gpuEndTime − gpuStartTime`. Satin's CPU cost for one mesh is about 2.2 times the
minimal renderer's; GPU time is the same. Construction times are not comparable. The
first process paid about 305 ms for the minimal renderer's first run-time compile
(Metal compiler start-up), and Satin compiles its pipeline lazily on first draw. In
later processes, construction was 1.2–2.4 ms for the minimal renderer and 4.6–10.6 ms
for Satin.

## Scorecard

| Criterion | Result |
| --- | --- |
| Swift API ownership | Achievable: no Satin type crossed `SatinMeshScene`. A real wrapper would have to hide `Context`, `Object`, `Mesh`, `Geometry`, materials, and `FrameTime`, which is most of Satin's surface. |
| Shader control | Custom MSL works, but only under Satin's conventions: an injected `Vertex` struct and `VertexUniforms`, buffer-index enums, injection-marker comments copied from Satin's own shaders, and function names derived from the material label. Compilation failures are not thrown (see above). |
| GlassWeave independence | Unaffected; the experiment links nothing into Frostfield or GlassWeave. |
| Resource control | Good for render targets and command buffers, which the caller owns. Weaker inside Satin: `Context.init` force-unwraps `makeCommandQueue()` (a crash path), and index data enters as an `UnsafeMutableRawPointer` whose lifetime the caller must manage. Satin's own test harness waits with `waitUntilCompleted()`. |
| Packaging | Plain source, with no plugins or binaries. Resources load from `Satin_Satin.bundle` through `Bundle.module`. The consumer's own shader resources loaded the same way. Only macOS was built; an iOS app build was not attempted. |
| Integration | Not tested: the UIKit host presentation was out of reach this session. |
| Cost | 2.2 times the CPU encode time for one mesh on the M5. About 60,000 lines of Swift, Objective-C++, and MSL to own or trust. Swift 5 language mode in the dependency, a 6.3 MB executable, an 884 KB resource bundle, and an extra 32 s clean build. Device traces are missing. |

**Stop conditions** (research spikes §2): the fixture needed no fork, and no Satin
type appeared in a proposed public API. The decision rests on cost and failure
semantics, not on a blocker.

## Tooling limitation

The package was built and run with SwiftPM on the command line, not through the Xcode
MCP bridge. The isolated package has to resolve its own `Package.resolved`. Opening it
on its own in the headless bridge is exactly what leaves a stale package root behind,
and the bridge does not reload a changed workspace member list, so it cannot be added
to `Frostfield.xcworkspace`. Adding it to the workspace would also make every workspace
build, including CI's, fetch Satin. Swift 6.4's `swift build` uses the Swift Build
engine and compiled Satin's Objective-C++ and Swift without Xcode project settings.

## Missing evidence (explicitly not established)

- Presentation through the same UIKit host (FF-05 criterion 1), any iOS build of the
  consumer, and any iPhone or iPad run.
- Matched GPU traces or Instruments captures.
- Memory footprint at run time, and frame-overlap hazards on a reused target.

## Revisit when

- Frostfield's roadmap needs features Satin already provides and Frostfield would
  otherwise have to build (for example loaders, text, shadows, or physically based
  materials beyond the planned Lambert slice). Rerun this package against the then
  current revision, with an iOS UIKit host and device traces, before deciding.
- Satin gains thrown, typed shader-compilation errors and Swift 6 language mode.
  Either change removes one of the reasons above.
