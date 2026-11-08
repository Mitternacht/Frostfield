# Implementation language strategy

Decision recorded 2026-11-08 after an independent research session. This is an implementation recommendation; no framework has been built or benchmarked.

## Recommended stack

Use **Swift for CPU code, public APIs, and native integration**, and **Metal Shading Language (MSL) for custom GPU rendering and compute**. Permit small C-compatible declarations for shared CPU/GPU data layouts. Add C/C++ or Rust algorithms only when a measured workload or a useful existing library justifies them.

| Responsibility | Language | Reason |
| --- | --- | --- |
| Public scene/surface API, validation, errors | Swift | Fits Apple app consumers and keeps the interface independent of foreign implementation types. |
| Resource ownership, scheduling, command encoding | Swift | Direct Metal integration; explicit lifetime and bounded work matter more than the language label. |
| Transforms, coordinates, ordinary geometry/interaction math | Swift with `simd` | Start with contiguous numeric data and optimized builds. |
| Shaders, glass refraction, GPU particles and post-processing | MSL | Custom vertex, fragment, and compute work executes on the GPU. |
| UIKit and optional SwiftUI adapters | Swift | Preserves native lifecycle, controls, accessibility, and tooling. |
| GPU record definitions | Small C-compatible header, or explicitly verified serialization | Establishes byte layout rather than assuming a Swift struct matches a shader struct. |
| Proven heavy CPU kernel or adopted library | Isolated C/C++ when justified; Rust only for a concrete subsystem benefit | Keeps additional build and ownership boundaries bounded. |
| Tests and fixture tools | Swift Testing; XCTest for UI/performance; Swift command-line tools by default | Matches the consumer environment. Python is acceptable for useful offline fixture generation. |

GlassWeave native mode uses Swift and Apple's public glass APIs without creating a custom Metal renderer. Its custom mode uses Swift plus MSL. Frostfield may later have a stronger case for foreign geometry/asset kernels, but neither project currently has measurements establishing that need.

Swift CPU code does not shade every pixel. MSL does that GPU work. MSL is C++-derived, but is a separate language with GPU-specific types, address spaces, and restrictions; choosing it is not choosing a C++ host engine. [Apple MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

## Why not begin with a C++ or Rust core?

A mostly C++ core is sensible when a substantial existing engine or real multi-platform implementation must be reused. Here it would add a second API, ownership translation, and native wrappers before a benefit is established. Swift can interoperate with C++ directly; Objective-C++ is useful for particular bridges, not a mandatory layer. Verify copies, exception translation, and consumer build settings. SwiftPM C++ interop can affect dependent targets, so a Swift facade alone does not prove the consumer avoids those settings. [Swift C++ interoperability](https://www.swift.org/documentation/cxx-interop/), [package setup](https://www.swift.org/documentation/cxx-interop/project-build-setup/)

Rust supports iOS, but adds Cargo/SwiftPM/Xcode coordination, binding ownership, binary distribution, and panic/error rules. It can be valuable for a named portable library or parser. Its borrow checker cannot prove asynchronous GPU resource use or foreign Swift callers safe. Keep any future Rust boundary coarse, with explicit buffers, owners, and errors. [Rust iOS support](https://doc.rust-lang.org/rustc/platform-support/apple-ios.html), [FFI guidance](https://doc.rust-lang.org/nomicon/ffi.html)

These are integration tradeoffs, not claims that one language is universally faster.

## Toolchain policy

Recommend **Swift tools 6.2 and Swift 6 language mode** for the first manifest, replacing the earlier provisional tools-6.0 proposal. Record the exact Xcode, compiler, SDK, default isolation, and enabled language features during the package-baseline task. Keep iOS/iPadOS 26 as the proposed deployment floor and macOS 26 as a core compilation target. The local agent workflow still uses the installed Xcode 27 skills and native Xcode MCP server.

Compiler version, manifest tools version, Swift language mode, OS deployment floor, Metal submission API, and MSL dialect are separate choices. Select the least shader dialect/features needed by the fixture and verify them on the supported hardware. Do not require iOS 27 features merely because the development machine has Xcode 27. New low-level Swift types require their own compiler/runtime availability checks. [Swift 6.2 release](https://www.swift.org/blog/swift-6.2-released/)

## Efficient Swift and ownership rules

Keep approachable public objects separate from the hot rendering representation. Use numeric handles, contiguous arrays, reused capacities, bounded update batches, and persistent frame storage where useful. Avoid per-vertex objects, per-draw tasks, repeated whole-scene copies, and accidental copy-on-write in identified hot paths. Prepare pipelines and resources outside steady interaction. Measure optimized Swift before changing language. [Apple Swift performance guidance](https://developer.apple.com/videos/play/wwdc2024/10217/)

Keep native UI on the main actor and renderer mutation under one explicit owner. An `async` function is not necessarily background work. Do not suspend midway through a resource ownership transition without preserving a valid state. CPU task cancellation does not cancel GPU work; retain resources until every GPU consumer completes. C++ destructors and Rust `Drop` need the same completion-aware owner. [CPU/GPU synchronization](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work)

## CPU/GPU layout correctness

At the first real uniform/vertex-buffer slice, define size, alignment, offsets, stride, matrix convention, and binding indices. Prefer a small shared C-compatible declaration imported into Swift and included by MSL, with conditional host/shader includes. This is a local data contract, not a premature common framework. Introduce a Clang target only when real data records need it; verify packaging in a clean consumer.

Use fixed-width scalars, documented SIMD layouts, and explicit flags. Avoid raw Swift `Int`, `Bool`, collections, optionals, or references in uploaded records. `float3` and `packed_float3` have different layouts; `@frozen` does not make a Swift struct C-compatible. Validate host layout with assertions and verify distinctive values through an actual GPU round-trip. Matching total size alone is insufficient. [Swift layout guarantees](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0260-library-evolution.md#guarantees), [MSL layout rules](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

## Gate for another implementation language

A proposal must name a missing budget or a library reuse benefit. For performance, first establish that the bottleneck is CPU work the proposed change can affect. Compare optimized candidates at the same output quality, including conversion, copying, FFI, and synchronization costs. A faster isolated kernel is insufficient if the app's pacing, memory, or energy behavior worsens.

Before adoption, record a numeric success threshold, repeated named-device results, correctness and p99 behavior, clean Swift consumer builds, symbols and deployment support, license/revision, maintenance owner, and ownership/error/cancellation rules. Prefer restoring a failed product budget; otherwise require a material whole-operation improvement beyond measured run-to-run variation. Keep foreign types out of the public API unless deliberately accepted as a public dependency.

Try suitable `simd`, Accelerate, or Metal compute primitives when they fit. Reusing a well-suited existing library does not require rewriting it in Swift first. [Accelerate](https://developer.apple.com/documentation/accelerate)

## Conditional backend alternative

RealityKit's new low-level renderer is worth a bounded Frostfield comparison if an iOS 27 floor later becomes acceptable. It is a backend choice within the Swift/MSL approach, not a reason to adopt C++ or Rust. It must not block either proposed iOS 26 v0.1 or replace explicit GlassWeave backdrop/lifetime contracts. Recheck availability and shader/resource control at that future decision. [LowLevelRenderer](https://developer.apple.com/documentation/realitykit/lowlevelrenderer), [standalone context](https://developer.apple.com/documentation/realitykit/lowlevelrendercontextstandalone)

## Integration into the task list

The existing task checklist remains authoritative. Package-baseline tasks pin the toolchain. The first real buffer/material tasks add CPU/GPU layout evidence. Device qualification records allocation/copy behavior as well as timing. The Satin experiment checks consumer interop and copies. Any new language requires the gate above; no separate rewrite milestone is presumed.

## Recorded toolchain (FF-01)

Recorded 2026-11-08 while creating the package; see
[the build baseline](evidence/build-baseline.md) for the full record. Xcode 27.2
(27B52), Apple Swift 6.4.2, iOS/macOS 27.2 SDKs, tools version 6.2, Swift 6
language mode, iOS/macOS 26.0 floors, nonisolated package default isolation, and
the compiler-default MSL dialect for the 26.0 target (`__METAL_VERSION__ == 400`
on this toolchain). Example apps set `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`.
