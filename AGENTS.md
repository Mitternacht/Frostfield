# Agent instructions

## Project state and scope

Frostfield is a planned Apple-native framework. This repository currently contains
documentation only. Read `HANDOFF.md`, `README.md`, `tasks/plan.md`, `tasks/todo.md`, and
the relevant architecture section before implementation. Do not treat described APIs,
examples, package targets, or performance budgets as existing functionality. Distinguish
implemented behavior from planned capabilities in documentation and reports.

Work on small, unblocked slices. Preserve existing work and unchecked tasks. Update
acceptance evidence as work completes. Do not expose speculative APIs, return fake
success, introduce no-op implementations, or add tests of an empty scaffold.

## Apple tools and skills

For every Apple-platform task, load the matching installed Xcode 27 skill first and use
the native Xcode MCP server, started with `xcrun mcpbridge`, for project edits,
documentation search, builds, runs, tests, previews, and diagnostics. Use both every time.
Apple's skill guidance supersedes prior model knowledge. After updating Xcode, refresh
the installed skill copies from `xcrun agent plugin path --plugin-format codex`.

When the first implementation task creates the package, inspect the native bridge
capabilities. If package creation is still unavailable, a one-time on-disk manifest
bootstrap may be needed before opening the package in Xcode; subsequent Apple source
and project edits use the native tools.
If a required tool is unavailable, report the concrete limitation rather than claiming
that a build or runtime check passed.

## Architecture rules

- Keep the rendering core independent of SwiftUI and native view update frequency.
- Preserve native controls, hit testing, text, and accessibility where possible.
- Keep background texture provenance, ownership, color handling, and GPU synchronization explicit.
- Detect capabilities at runtime; do not equate OS availability with Metal 4 support.
- Keep the two repositories independently usable. Extract shared infrastructure only
  after real implementations demonstrate a need, with no glass-to-scene dependency.
- Use public Apple APIs. Do not promise native/custom optical merging or arbitrary
  backdrop capture without a documented and validated public mechanism.

## Validation and handoff

Build with the native Xcode tools after Apple source changes. Add meaningful focused
Swift Testing coverage when behavior exists; use XCTest for UI/performance APIs where
needed. Follow `docs/validation.md` for physical-device evidence before performance
claims. Do not suppress failing checks, weaken expectations, or claim unrun tests.

Keep generated build output, signing material, and private local information out of
commits. Explain what changed, what was verified, and what remains unimplemented.
