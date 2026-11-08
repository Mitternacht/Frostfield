# Satin comparison (FF-05 experiment)

This folder contains a timeboxed experiment, not Frostfield API. It is an isolated
Swift 6 package that depends on [Fabric-Project/Satin](https://github.com/Fabric-Project/Satin)
at one pinned revision. It renders the same transformed, indexed mesh with the same
custom MSL shading through Satin and through a minimal Metal renderer, exports both
off-screen textures as PNG files, and compares them pixel by pixel. The decision and
results are in [the renderer foundation decision](../../../docs/decisions/renderer-foundation.md).

Frostfield's manifest, library, workspace, lab app, and CI never reference this
package. It resolves its own `Package.resolved`.

## Contents

| Path | Purpose |
| --- | --- |
| `Package.swift` | Swift tools 6.2, Swift 6 language mode, macOS 26. Satin is pinned by revision. |
| `Sources/SatinComparison/Fixture.swift` | The shared fixture: a 4 × 4 indexed grid, model transform, orthographic camera, 512 × 512 `bgra8Unorm` target, PNG export, and image diff. |
| `Sources/SatinComparison/MinimalMeshRenderer.swift` | Plain Metal: one pipeline, one vertex buffer, one `uint32` index buffer, one indexed draw. |
| `Sources/SatinComparison/SatinMeshScene.swift` | Satin `Context`, `RenderEncoder`, `OrthographicCamera`, `Mesh`, `Geometry` with an `ElementBuffer`, and a `SourceMaterial`. No Satin type leaves this file. |
| `Sources/SatinComparison/Shaders/` | `StripesColor.h` (the shading both renderers use), `Stripes/Shaders.metal` (Satin conventions), and `Minimal.metal`. Both renderers compile these at run time. |

## Run

From this directory:

```sh
swift build -c release
"$(swift build -c release --show-bin-path)/SatinComparison" /path/to/output-folder
```

The tool prints `ENV`, `SETTINGS`, `CREATE`, `DIFF`, `CHECK`, `SHADER_FAILURE`, and
`RESULT` lines. It writes `minimal.png` and `satin.png` to the output folder and exits
non-zero if any check fails.

The package is built with SwiftPM rather than through the shared Xcode workspace,
because it must resolve its dependency on its own. Mac results describe that Mac
only; they are not iPhone or iPad evidence.
