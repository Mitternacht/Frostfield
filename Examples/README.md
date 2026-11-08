# Frostfield examples

Open `Frostfield.xcworkspace`. It contains the Frostfield package and the
`FrostfieldLab` iOS/iPadOS app, which depends on the package through a local
package reference (`../..`). No signing team is configured; run the lab on an
iOS Simulator or add your own team locally without committing it.

| Scheme | Destination | What it runs |
| --- | --- | --- |
| `Frostfield` | My Mac | Package unit tests, including off-screen GPU renders with readback. The scheme's test action sets `MTL_DEBUG_LAYER=1` for Metal API validation. |
| `FrostfieldLab` | An iOS Simulator | The lab app and its XCTest UI tests. |

## FrostfieldLab

The lab is the UIKit `MTKView` fixture for the rendering host. Each window scene
creates its own `SceneViewController`, `TriangleRenderer`, and view; nothing is a
process-wide singleton. It draws the package's diagnostic triangle (red apex,
green lower left, blue lower right) from the shader library packaged in the
Frostfield resource bundle.

When the renderer cannot be created, the lab shows the typed `RendererError`
instead of a render view. Launch with `-FrostfieldLabSimulateMissingDevice` to see
that failure state on hardware that has a Metal device.

The executable work and acceptance gates are listed in [the backlog](../tasks/todo.md).
Simulator runs are integration smoke checks, not GPU or performance evidence.
