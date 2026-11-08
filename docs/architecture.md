# Frostfield architecture

Status: proposed design, 2026-11-08. The initial repository contains planning documents only: no package manifest, source, tests, CI implementation, or renderer. Types, target names, and contracts become public only after their implementation fixtures validate them.

## Boundaries

Frostfield turns application-supplied scene intent into Metal rendering work. Its core must work without SwiftUI, UIKit, a window, or a display loop supplied by an application UI framework. The first presentation integration is a UIKit `MTKView`; SwiftUI wraps that host after its lifecycle behavior is established. macOS is initially a core compilation environment, not a supported presentation integration.

The future structure has three responsibilities:

| Responsibility | Owns | Does not own |
|---|---|---|
| Scene description | Geometry, transforms, cameras, materials, lighting/effect parameters, stable identities | UIKit views or Metal submission timing |
| Rendering implementation | Device capabilities, prepared pipelines, resource pools, render passes, scheduling, GPU lifetime | Native control semantics or application navigation |
| Presentation adapters | Drawable size, visibility/lifecycle signals, native layout and input integration | A separate simulation/render loop or per-frame scene reconstruction |

Initially these boundaries can be internal files in one package target. Split targets when an actual independent consumer needs them. `FrostfieldUIKit` and `FrostfieldSwiftUI` are proposed optional products, not currently shipped adapters. Adapters depend on the core; the core never imports its adapters.

## Rendering approach

Start with a conventional `MTLCommandQueue` triangle and a bounded number of frames in flight. Use the same fixture for a Metal 4 spike. Metal 4 is a backend candidate, not a user-facing promise of superior performance. Apple documents incremental adoption and independent Metal 4 command types alongside shared resource types. Runtime OS availability and GPU capability checks must agree before using a path. [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api), [feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions).

Metal 4 supports A14/M1 and later hardware, but support for the API does not mean every optional feature has the same limits. In particular, do not infer device support from the package's iOS 26 deployment target. [Apple's Metal 4 overview](https://developer.apple.com/videos/play/wwdc2025/205/), [feature tables](https://developer.apple.com/metal/capabilities/).

Compare Satin with the minimal implementation in an isolated, bounded experiment. Examine shader escape hatches, resource ownership, dependency cost, licenses, and same-scene timings. The experimental common interface must not freeze the public architecture before either implementation has proven useful. The original [Hi-Rez/Satin](https://github.com/Hi-Rez/Satin) repository was archived on April 15, 2025; [Fabric-Project/Satin](https://github.com/Fabric-Project/Satin) is a candidate to revalidate, not a dependency already selected. Its inspected manifest uses tools 6.2, Swift 5 language mode, C++17, and no declared external package dependencies; its MIT license retains Hi-Rez copyright. Test integration with a Swift 6 consumer and record the exact evaluated revision. See [research spikes](research-spikes.md).

The first renderer needs ordered passes, not a general-purpose render-graph compiler. Add offscreen scene output and one post-process when required by a fixture. Consider more elaborate scheduling only when measured dependencies and reuse warrant it.

## State, threading, and lifetime

Native hosts follow the main actor's UI rules. Rendering receives a stable state snapshot or an explicit update batch; its exact representation is established by the transform/material fixtures. The UI should not publish thousands of particle positions or drive individual GPU updates through SwiftUI state.

Every native host derives size, display scale, lifecycle, and placement from its own view/window/scene context. Compute drawable dimensions from local bounds and `traitCollection.displayScale`; respect safe areas and live resizing. Do not use `UIScreen.main`, an application-wide window assumption, or a singleton render host. Two scenes must be able to resize, move between displays, pause, and resume independently.

A render owner serializes mutation of its device-specific caches and frame resources. GPU resource safety is separate from Swift actor isolation: a CPU task finishing does not mean the GPU has stopped reading its buffer. Every reusable resource needs a bounded ownership path from acquisition through submission to completion. Do not mark Metal resources broadly `Sendable` simply to silence compiler diagnostics.

Prepare shader libraries and common pipeline variants before interaction. Parameter changes update existing buffers or bindings. Resource pools are bounded, keyed by compatible descriptors, and account for textures still held by in-flight consumers. Resize creates a new usable generation while the old generation remains alive until its readers finish.

Never block the main thread waiting for a drawable or an available frame slot. A missed drawable, suspended app, or saturated consumer produces a defined skip/backpressure outcome. Paused and hidden scenes do not keep submitting frames. `MTKView` already supports requested and continuous drawing; a custom `CAMetalLayer` host needs evidence of a requirement before replacing it. [MTKView](https://developer.apple.com/documentation/metalkit/mtkview).

## Creative API progression

The first composition path is a camera plus transformed indexed geometry and a prepared material. Expose a documented shader binding contract so app authors can invent materials without waiting for a catalogue of effects. Add one directional Lambert light, one bounded particle emitter, and one full-screen post-process as separate slices.

Define matrix order, coordinate handedness, depth conventions, units, alpha handling, and color conversion once and test them with deterministic fixtures. Invalid geometry, shader functions, or incompatible bindings should produce actionable errors before unsafe work is encoded. Avoid a public type for every internal implementation detail.

A typed material graph, physically based lighting, loaders, animation systems, and editors are future design problems. A small working shader API should inform them.

The first usable v0.1 stops at camera/transforms, indexed geometry, unlit/custom material shaders, and a verified UIKit sample host. It requires its own clean-consumer and physical-device qualification at Checkpoint C. Lighting, particles, post-processing, public texture export, SwiftUI hosting, and GlassWeave interoperability are independent later milestones; their absence does not block that limited release. Any offscreen operation used by the Satin experiment is internal evidence about a backend, not a public texture API.

## GlassWeave interoperability

GlassWeave is a separate framework for native and custom glass surfaces. Neither library initially depends on the other. An example app may depend on both and connect a Frostfield offscreen output to a GlassWeave background input.

The exchange contract should be small and explicit. Its exact Swift spelling is intentionally deferred:

| Contract field | Required meaning |
|---|---|
| Texture/device | A readable texture on the same `MTLDevice` as the consumer; reject unsupported cross-device transfer |
| Frame identity | Monotonic identity and producer time; define reuse, skipped, and stale-frame behavior |
| Dimensions/region | Pixel dimensions and valid content rectangle, independent of view points and display scale |
| Coordinates | Explicit origin, normalized UV mapping, crop/transform, and edge-sampling policy |
| Format/color | Pixel format, linear versus encoded values, color primaries, alpha convention, and conversion responsibility |
| Readiness | Producer writes are ordered before consumer reads, using the agreed queue/command order or an explicit supported synchronization primitive |
| Lifetime | A lease remains alive through all consumer GPU reads; reuse/destruction occurs only after their completion |
| Backpressure | A bounded number of leases; skip or reuse the last complete frame when no slot is available, according to documented policy |

For the first interoperable fixture, prefer one agreed linear working color space and a single negotiated format, with premultiplied alpha and an explicit SDR display conversion. A possible internal working format is linear sRGB in an `rgba16Float` texture. GlassWeave v0.1 instead proposes an sRGB BGRA8 input: the bridge must export that supported format with a real conversion, or explicitly extend and validate GlassWeave's accepted formats. Metadata alone cannot convert pixel data. Verify format/usage support and both implementations before fixing the public contract. Wide-gamut/HDR and arbitrary format conversion are later capabilities. A color-chart fixture must detect accidental double encoding, incorrect premultiplication, and inverted UVs.

Begin with a coordinated same-queue producer/consumer path if both implementations can share it cleanly. Cross-queue transfer requires an explicit event/wait contract; cross-device import is out of first-release scope. In Metal 4, allocator memory must not be reset before all command buffers using it finish. [Allocator reuse](https://developer.apple.com/documentation/metal/mtl4commandallocator/reset()).

Frostfield output supplies controlled background pixels. It does not grant GlassWeave access to arbitrary app views, or make custom materials part of Apple's Liquid Glass renderer. SwiftUI's `layerEffect` applies to its own rasterized content and documents limitations for UIKit/AppKit-backed views. [layerEffect](https://developer.apple.com/documentation/swiftui/view/layereffect(_:maxsampleoffset:isenabled:)).

## When to share GPU infrastructure

Extract an optional low-level product only when both working frameworks need the same device/capability handling, pipeline preparation, resource ownership, or texture-exchange primitives. The extraction must remove measured duplication without importing scenes, meshes, loaders, lighting, or UI frameworks into GlassWeave's native-only path.

Before extraction, record two concrete consumers, an acyclic dependency graph, ownership and versioning, standalone build coverage, binary/linkage impact, and the migration cost. A product hosted in the Frostfield repository may avoid linking a scene engine while still creating a package/repository dependency; document that distinction. A third repository is not assumed necessary.

## Scope of native coexistence

Apple's documented `UIGlassEffect` controls material configuration and interactivity. `UIGlassContainerEffect` combines views using that effect. These interfaces do not document injecting Frostfield or GlassWeave shaders into the native glass renderer. Custom merging requires custom-rendered participants; native controls can remain above them. [UIGlassEffect](https://developer.apple.com/documentation/uikit/uiglasseffect), [UIGlassContainerEffect](https://developer.apple.com/documentation/uikit/uiglasscontainereffect).

The integration target is a coherent native app with programmable graphics and custom surfaces alongside system glass, with explicit cost and lifecycle behavior.

## Language decision

Use Swift for CPU implementation and native APIs, MSL for custom GPU work, and a
small verified C-compatible data-layout boundary when needed. The proposed
manifest baseline is Swift tools 6.2 with Swift 6 language mode. See the
[language strategy](language-strategy.md)
for toolchain distinctions, layout tests, ownership rules, foreign-language
admission gates, and conditional backend alternatives.
