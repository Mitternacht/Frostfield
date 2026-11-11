# Shader graph vertex revisions

A typed surface graph position offset captures its uniform values and
relevant clock revision when the draw is encoded. One command can reuse
its own unfinished vertex result. Another command uses a completed
immutable result for the same revision or computes into a separate
buffer. The command retains its result through completion, including
when Metal's retained references are disabled. An abandoned command
does not publish its result.

The renderer binds the prepared vertices with the authored index range
for color, shadow, selection depth, and surface-attribute draws. Motion
history captures those same vertices after the deformation has encoded.
The newest encoded pose that has completed stays cached while other
commands retain the poses they already reference. The cache tracks uniform and clock
revisions under the existing shader graph contract.

The original Mac reproduction encoded two commands at the same pending
revision and submitted the second first. It read a fully empty color
frame where the displaced mesh should have appeared. Regression tests
submit commands in reverse order, abandon an encoded command, reuse a
pose twice in one command, and render an older completed lease after a
new pose finishes. A scene fixture compares real pixels from color,
surface attributes, directional shadow, and motion output.

The exact candidate passed 116 tests in 16 focused suites on arm64
macOS and iPhone 18 Pro Simulator with Metal validation enabled. The
suite also exercises steady-state cache counts, forward and reversed
motion depth, curve flow and dynamic vertex history, material overrides,
and stencil. The Xcode MCP could not create DerivedData on the full
external volume, so verification used `xcodebuild` and internal caches.
