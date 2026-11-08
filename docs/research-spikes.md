# Renderer research decisions

These experiments inform implementation. No candidate is adopted and no
benchmark has been run in the planning session. Revalidate source facts when
the experiments begin.

## 1. Conventional Metal versus Metal 4

**Question:** Does Metal 4 improve the target workload enough to justify its
resource, synchronization, and compatibility costs?

Use the same shader, geometry, drawable size, frame limit, device, and quality
for both variants. Start from the deterministic fixture in FF-02/FF-03.
Keep the variants internal to the experiment. Do not introduce a public
multi-backend protocol just to compare them.

| Evaluate | Required evidence |
| --- | --- |
| Correctness | Equivalent reference output; resize and in-flight resource stress checks |
| CPU/GPU cost | Warm p50/p95/p99 timings; cold pipeline time recorded separately |
| Memory | Peak allocated bytes, transient resources, bounded frame backlog |
| Lifecycle | Idle, background, teardown, device/resource failure behavior |
| Coverage | OS availability plus runtime GPU capabilities; unsupported path tested |
| Maintainability | Extra synchronization, residency, compiler, and allocator ownership code |

Keep the simpler implementation unless a repeatable gain or required feature
justifies the extra complexity. A performance claim needs physical-device
evidence under the validation protocol. Record whether the decision is to keep
conventional Metal, adopt Metal 4 with a compatibility path, or explicitly narrow
hardware support. Do not silently ship two full backends.

Apple documents [incremental Metal 4 adoption](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api),
[GPU feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions),
and [allocator reset responsibilities](https://developer.apple.com/documentation/metal/mtl4commandallocator/reset()).

## 2. Minimal custom renderer versus Satin

**Question:** Does a maintained graphics dependency accelerate Frostfield while
preserving shader freedom, predictable resources, native hosting, and packaging?

The original [Hi-Rez/Satin](https://github.com/Hi-Rez/Satin) repository is archived.
The [Fabric-Project/Satin](https://github.com/Fabric-Project/Satin) fork is a
candidate to inspect. As reviewed on 2026-11-08, its
[manifest](https://github.com/Fabric-Project/Satin/blob/main/Package.swift) declares
Swift tools 6.2, Swift 5 language mode, a C/C++ core, and no external package
dependencies. Validate its boundary with a Swift 6 consumer. Its
[license](https://github.com/Fabric-Project/Satin/blob/main/LICENSE) is MIT;
preserve required notices for any incorporated code. Verify the exact revision
and all bundled assets before adoption.

Limit the comparison to a transformed indexed mesh, one custom shader, offscreen
texture output, and the UIKit host. Use identical resolution and visual output.
Do not migrate or wrap the dependency's full scene system during the experiment.

| Criterion | Pass condition |
| --- | --- |
| Swift API ownership | Frostfield can expose its own small API without exporting dependency types |
| Shader control | Custom MSL, resource bindings, and compilation failures have a documented path |
| GlassWeave independence | A glass control never imports or links Satin or a Frostfield scene target |
| Resource control | Texture ownership, completion, render scheduling, and teardown can be explained and tested |
| Packaging | Clean consumer build; shader/resource loading works outside the development checkout |
| Integration | UIKit host, optional SwiftUI wrapper, and native interaction remain under our control |
| Cost | Comparable device traces plus dependency size, build time, and maintenance burden |

**Stop the experiment** if the narrow fixture requires a broad fork, uncontrolled
runtime behavior, or dependency types throughout the proposed public API. Record
the failure and proceed with the minimal implementation. Conversely, accept the
dependency if evidence shows a material advantage without violating the above
constraints. Neither a from-scratch renderer nor a dependency is a founding rule.

Time-box the first feasibility pass to one focused session. If a candidate cannot
build, document the exact failure and estimated repair scope before spending a
second session. Each experiment produces a short decision record containing
revision, fixture, environment, measurements, alternatives, and revisit trigger.

## 3. Shared Metal infrastructure extraction

Do this after both frameworks run, not during initial setup. List duplicated
behavior and at least two concrete callers for every proposed abstraction.
Candidate responsibilities are pipeline caching, bounded texture reuse, and
completion-aware resource leases. Scene graphs, mesh loading, glass policy, and
UI hosting do not belong in the common layer.

A future `FrostfieldMetal` product in the Frostfield repository is one option.
That would make GlassWeave depend on the Frostfield package/repository while
avoiding scene-engine linkage. Document that distinction and verify the package
graph in a clean GlassWeave-only consumer. If extraction saves little complexity,
keep the small implementations internal and the repositories independent.
