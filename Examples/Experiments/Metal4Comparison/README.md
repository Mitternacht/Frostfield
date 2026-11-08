# Metal 4 comparison (FF-04 experiment)

This folder contains a timeboxed experiment, not Frostfield API. It renders
Frostfield's diagnostic triangle through conventional Metal and through the Metal 4
command path, then compares the images and the timings. The decision and the
measured results are in [the Metal backend decision](../../../docs/decisions/metal-backend.md).

Nothing here ships. The Frostfield package manifest does not reference this folder,
and these files are not compiled into the `Frostfield` library or the lab app.

## Layout

| Folder | Contents |
| --- | --- |
| `Shared/` | `ConventionalTriangle` (a mirror of `TriangleRenderer` that also reports GPU time), `Metal4Triangle`, the slot pool, backend selection, the benchmark harness, and `Shaders/Triangle.metal`. That shader file includes Frostfield's own `Sources/Frostfield/Shaders/Triangle.metal`, so both paths compile the same source. |
| `Metal4ComparisonTests/` | Swift Testing cases: byte-for-byte output equality, slot and allocator reuse, residency, resize, and backend selection. |
| `Metal4Benchmark/` | Entry point of the timing tool. |

The targets live in `FrostfieldLab.xcodeproj` (macOS targets `Metal4ComparisonTests`
and `Metal4Benchmark`) because the headless Xcode bridge used for this work would not
load a new member of the shared workspace. Their source folders are synchronized from
this directory.

## Run

Open `Examples/Frostfield.xcworkspace`.

- Scheme `Metal4ComparisonTests`, destination My Mac: runs every test with Metal
  API validation on. On an iOS Simulator destination, the Metal 4 code is compiled
  out, and the tests check the conventional fallback.
- Scheme `Metal4Benchmark`, destination My Mac: runs the Release timing tool with
  API validation and GPU capture off, and prints `ENV`, `SETTINGS`, and `RESULT`
  lines.

Mac results describe the Mac they ran on. They are not iPhone or iPad evidence.
