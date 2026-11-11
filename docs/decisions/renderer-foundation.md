# Renderer foundation and Satin feasibility

Recorded 2026-11-09 for FF-05. Continue with Frostfield's conventional Metal renderer. Satin passed an isolated offscreen feasibility probe. Adoption requires a matched native host comparison and resource lifetime evidence.

## Candidate

[Fabric-Project/Satin at `ff825e4945ca19277355eda69831b9b19aad1533`](https://github.com/Fabric-Project/Satin/commit/ff825e4945ca19277355eda69831b9b19aad1533) was the unarchived fork's `main` HEAD at evaluation. The commit was merged on 2026-11-07. The pinned manifest uses Swift tools 6.2, Swift 5 language mode and C++17. The `Satin` product links the `Satin` and `SatinCore` targets, declares no external packages and copies `Pipelines` and `Textures` into its resource bundle. Any distribution would need to retain the MIT permission and Hi-Rez copyright notices and audit bundled assets. [Manifest](https://github.com/Fabric-Project/Satin/blob/ff825e4945ca19277355eda69831b9b19aad1533/Package.swift) · [License](https://github.com/Fabric-Project/Satin/blob/ff825e4945ca19277355eda69831b9b19aad1533/LICENSE)

## Observed

- An isolated Swift 6 executable package declared macOS, iOS and visionOS 27.0 and resolved Satin at the pinned revision. Its macOS arm64 Debug build passed on an Apple M4 Max running macOS 27.2 (26B123) with Xcode 27.2 (27B52) and Apple Swift 6.4.2. Build logs showed Swift 6 for the consumer and Swift 5 for Satin. The native Xcode bridge rejected the manifest as a Swift source file and timed out resolving the package directory's initial destination, so the build used `xcodebuild`.
- `SourceMaterial` loaded a custom magenta MSL fragment and `RenderEncoder` drew a translated, scaled quad with six indices into a caller-owned 64 × 64 shared RGBA8 texture. The probe kept the texture alive, committed the caller's conventional Metal command buffer and waited for successful completion before CPU readback. With Metal API Validation enabled, the left, interior and right pixels matched `[0, 0, 0, 255]`, `[255, 0, 255, 255]` and `[0, 0, 0, 255]`. [SourceMaterial](https://github.com/Fabric-Project/Satin/blob/ff825e4945ca19277355eda69831b9b19aad1533/Sources/Satin/Materials/SourceMaterial.swift) · [RenderEncoder](https://github.com/Fabric-Project/Satin/blob/ff825e4945ca19277355eda69831b9b19aad1533/Sources/Satin/Core/RenderEncoder.swift)
- The executable, custom shader and generated `Satin_Satin.bundle` ran successfully when copied together into a separate directory with package bundle overrides unset. This Debug probe executable was 10,430,632 bytes. Its resource bundle contained 176 files and occupied 884 KiB on disk. These are local artifact sizes before distribution processing.

## Open checks

The timebox established one Satin frame on macOS. Matched presentation through the same native host, mobile and spatial builds, asynchronous producer/consumer readiness, delayed reuse, and comparative build and GPU costs remain open. The synchronous wait belongs to the command-line probe. A later comparison should use the same transformed indexed mesh, shader output and settings in both renderers, then verify reuse while a delayed GPU consumer retains the texture. Revisit adoption when that fixture and comparable traces can establish a benefit.
