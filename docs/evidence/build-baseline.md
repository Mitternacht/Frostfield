# FF-01 build baseline

Date: 2026-11-08 · Task: FF-01 · Branch: `implement/milestone-1`

This record covers packaging and compilation only. It establishes no rendering,
GPU, device-compatibility, or performance result.

## Toolchain

| Item | Value |
| --- | --- |
| Xcode | 27.2 (27B52) |
| Swift compiler | Apple Swift 6.4.2 (swiftlang-6.4.2.1.9) |
| SDKs | iOS 27.2, iOS Simulator 27.2, macOS 27.2 |
| Manifest | `swift-tools-version: 6.2`, `swiftLanguageModes: [.v6]` |
| Platforms | iOS 26.0, macOS 26.0 (parsed by this toolchain without diagnostics) |
| Default actor isolation | Package: nonisolated (no `defaultIsolation` setting) |
| MSL dialect | Compiler default for the 26.0 deployment target, no `-std` flag. A probe shader built with this toolchain reported `__METAL_VERSION__ == 400` (MSL 4.0). |
| Development Mac GPU | Apple M5 (core builds and GPU unit tests only) |

## Builds

Built through the native Xcode MCP bridge (`BuildProject`) with the `Frostfield`
scheme:

| Destination | Result |
| --- | --- |
| My Mac (macOS 27.2) | Succeeded, no diagnostics |
| Any iOS Simulator Device (iOS 27.2 SDK) | Succeeded, no diagnostics |

Opening a package that has never been opened in Xcode failed the first build
with `unable to spawn process … clang-stat-cache (No such file or directory)`:
the build service's working directory `.swiftpm/xcode` did not exist yet.
Creating that (git-ignored) directory fixed it.

## Continuous integration

`.github/workflows/ci.yml` repeats these builds on a `macos-26` runner with
read-only permissions, a pinned checkout action, no signing, and an explicit
SDK check that fails when the selected Xcode cannot build for the declared
26.0 platforms. It has not run on GitHub yet; its result is recorded when it
does.

## Not established

GPU validation, rendering correctness, simulator or device behavior, and any
performance property. The iOS Simulator SDK does not declare the Metal 4 GPU
family; `RenderCapabilities` reports it unavailable in simulator builds.
