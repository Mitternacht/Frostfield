# Contributing to Frostfield

Start with [HANDOFF.md](HANDOFF.md), read the
[plan](tasks/plan.md), and use the [ordered backlog](tasks/todo.md) as the source of
implementation tasks. Review the relevant [architecture](docs/architecture.md) contracts
before adding behavior.

## Work in verifiable slices

Keep changes focused on one useful outcome. Define behavior before adding a public API;
do not publish placeholder protocols or methods that silently do nothing. Add focused
Swift Testing coverage with the first testable behavior. Use XCTest where UI automation
or performance measurement requires it. Keep the package building between slices.

For visual changes, include a small reproducible example, reference captures, and the
device/OS used. Follow the [validation protocol](docs/validation.md) for GPU correctness,
frame pacing, resource lifetime, energy, and accessibility work. Simulator checks are
useful for integration and are not evidence of physical-device performance.

## Implementation setup

Open the Swift package in Xcode 27 with the macOS, iOS, and visionOS 27 SDKs. The
manifest uses Swift tools 6.2 and Swift 6 mode and targets macOS, iOS, iPadOS, and
visionOS 27.0 or later. Run the native macOS lab for local rendering work and the
corresponding native destinations for iOS, iPadOS, and visionOS checks. See the
[example instructions](Examples/README.md) and
[build baseline](docs/evidence/build-baseline.md).

Coding agents must follow [AGENTS.md](AGENTS.md): load the matching installed Apple
Xcode 27 skill and use the native Xcode MCP server for Apple project edits, documentation,
builds, diagnostics, previews, runs, and tests. Human contributors may use Xcode directly.

## Continuous integration

The CI workflow uses Xcode 27.2 on GitHub's `xcode-27` runner to build macOS Debug
and Release, generic iOS Simulator Release, and generic visionOS Simulator Release.
It records the toolchain and checks for the Metal compiler before building. Add
focused test jobs with the first testable behavior.

Use read-only workflow permissions, pinned action revisions, and `pull_request` for
contributor builds. Avoid signing credentials for package-only checks. Keep physical
device validation separate from simulator build results.

Apple documents `xcodebuild` for external CI in its
[Swift package CI guidance](https://developer.apple.com/documentation/xcode/building-swift-packages-or-apps-that-use-them-in-continuous-integration-workflows).
Verify Xcode and SDK availability in the selected image when updating the CI
toolchain. See the [official Xcode 27 runner image](https://github.com/actions/runner-images/blob/main/images/macos/xcode-27-arm64-Readme.md#xcode).

## Pull requests and releases

Explain the problem, the resulting behavior, and checks actually performed. Update the
backlog and relevant docs in the same change. Public API changes need a compatibility
note; rendering or performance claims need evidence. Do not weaken existing checks to
make a change pass.

The first release follows its roadmap acceptance gates, includes a changelog and
usable example, and is tagged with a semantic version. Contributions are made
under the repository's [MIT license](LICENSE).
