# FF-01: package build baseline

2026-11-09, implementation working tree based on `e6a7144`.

| Setting | Observed value |
| --- | --- |
| Xcode | 27.2 (27B52) |
| Build host | Apple silicon, macOS 27.2 (26B123) |
| SDKs used | macOS, iOS Simulator, and visionOS Simulator 27.2 |
| Swift compiler | Apple Swift 6.4.2 (`swiftlang-6.4.2.1.9 clang-2100.4.1.1`) |
| Package tools and language mode | Swift tools 6.2; Swift 6 (`-swift-version 6`) |
| Deployment floors | iOS/iPadOS, macOS, and visionOS 27.0 |
| Package default actor isolation | `nonisolated`; the package does not opt into `MainActor` isolation or additional Swift language features |
| Shader dialect | FF-01 had no shader. With FF-02, Xcode 27.2 defaults to MSL 4.1 for the three 27.0 targets (`__METAL_VERSION__ = 410`, no `-std` override). |

`xcrun swift package dump-package` parsed all three platforms at 27.0, one automatic
library product named `Frostfield`, one target, and no package dependencies. The
deployment floors use string forms to retain the 6.2 manifest tools baseline.
Default package isolation follows [SE-0466](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0466-control-default-actor-isolation.md).

The native Xcode bridge built the `Frostfield` scheme in Debug for **My Mac**
(arm64), **Any iOS Simulator Device**, and **Any visionOS Simulator Device**.
All returned success with empty build error lists. The compiler log confirmed
Swift 6 mode. The standalone source diagnostics call failed with
`SourceEditorCallableDiagnosticError` on the bootstrap package; build diagnostics
were available. The bridge exposed no Swift package creation operation and could
not resolve a destination before the target had source, so the manifest and initial
source were bootstrapped on disk before reopening the package through the bridge.

A hosted build workflow once selected Xcode 27.2 on the
[GitHub `xcode-27` arm64 runner](https://github.com/actions/runner-images/blob/main/images/macos/xcode-27-arm64-Readme.md#xcode)
and built macOS Debug and Release and generic iOS and visionOS Simulator Release. It was
removed because the project does not use CI. Package compilation establishes no
physical device behavior or GPU performance.

All three Release destinations passed locally with `xcodebuild`. The bridge's
build operation uses the active scheme configuration and exposes no Release override.
