# Contributing to Frostfield

This is a planning repository. Start with [HANDOFF.md](HANDOFF.md), read the
[plan](tasks/plan.md), and use the [ordered backlog](tasks/todo.md) as the source of
implementation tasks. Review the relevant [architecture](docs/architecture.md) contracts
before adding behavior.

## Work in verifiable slices

Keep changes focused on one useful outcome. Define behavior before adding a public API;
do not publish placeholder protocols or methods that silently do nothing. Add focused
Swift Testing coverage with the first testable behavior. Use XCTest where UI automation
or performance measurement requires it. Keep the package building between slices once
it exists.

For visual changes, include a small reproducible example, reference captures, and the
device/OS used. Follow the [validation protocol](docs/validation.md) for GPU correctness,
frame pacing, resource lifetime, energy, and accessibility work. Simulator checks are
useful for integration and are not evidence of physical-device performance.

## Implementation setup

There is no `Package.swift` or source implementation yet. The first code task will create
the package with Swift 6 mode and proposed iOS 26 and macOS 26 deployment baselines.
Native UI work starts on iOS/iPadOS; a macOS core build will not imply a macOS UI adapter.

Coding agents must follow [AGENTS.md](AGENTS.md): load the matching installed Apple
Xcode 27 skill and use the native Xcode MCP server for Apple project edits, documentation,
builds, diagnostics, previews, runs, and tests. Human contributors may use Xcode directly.

## Planned continuous integration

No CI workflow is installed at the planning stage. Add one with the first package
implementation. The proposed starting checks are macOS Debug and Release builds and a
generic iOS Simulator Release build on `macos-26`, with the selected Xcode version
recorded in logs. Use focused test jobs as soon as testable behavior exists; do not add
tests that merely prove an empty module compiles.

Use read-only workflow permissions, pinned action revisions, and `pull_request` for
contributor builds. Avoid signing credentials for package-only checks. Keep physical
device validation separate from simulator build results.

Apple documents `xcodebuild` for external CI in its
[Swift package CI guidance](https://developer.apple.com/documentation/xcode/building-swift-packages-or-apps-that-use-them-in-continuous-integration-workflows).
Verify toolchain availability against the
[official runner image manifest](https://github.com/actions/runner-images/blob/main/images/macos/macos-26-Readme.md)
when implementing CI; the installed default can change.

## Pull requests and releases

Explain the problem, the resulting behavior, and checks actually performed. Update the
backlog and relevant docs in the same change. Public API changes need a compatibility
note; rendering or performance claims need evidence. Do not weaken existing checks to
make a change pass.

`main` currently contains planning documents. There is no release tag. The first release
follows its roadmap acceptance gates, includes a changelog and usable example, and is
tagged with a semantic version. Contributions are made under the repository's
[MIT license](LICENSE).
