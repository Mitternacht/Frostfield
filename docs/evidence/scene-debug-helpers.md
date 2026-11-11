# Camera and skeleton debug helpers

Use `LineGeometry.cameraFrustum(camera:depthMode:colors:)` to visualize a finite explicit camera in world space. Pass the `graphCamera` and `depthMode` from a glTF camera view or `CameraUtils.frameCorners`. Using the view and projection separately locates the cone and camera-to-near line at the actual orthographic camera origin. The helper preserves the 25 reference segments and allows individual colors for the frustum, cone, up marker, target line and crosses. [Three.js r187 CameraHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/CameraHelper.js).

`DebugHelpers.addCameraFrustum(projection:toCamera:in:)` attaches the local geometry to an authored camera pose node. Its scene draw follows the node and ancestors, including nonuniform scale, pivots and inherited visibility. Refresh the attachment when the projection or its colors change. A finite far plane is required to draw finite far corners.

`DebugHelpers.addSkeleton(joints:to:)` inserts a stable world-space root helper. It defaults to links between direct joint parents and children and renders above depth-tested geometry. Optional nearest-ancestor links bridge intermediate transform nodes, and optional visibility filtering skips hidden branches. The attachment exposes separate child and parent colors. Call `update(in:)` after applying the skeleton pose or changing source visibility. A helper that temporarily has no links keeps its node identity and resumes drawing on a later refresh. [Three.js r187 SkeletonHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/SkeletonHelper.js).

Projection and skeleton refreshes keep the line pipeline when compatible and replace the point, color and segment data as an immutable renderer snapshot. Existing encoded commands retain their original buffers. An identical refresh keeps the same renderer.

Pass an attachment in `SceneGraph.encode(..., updatingSkeletons:)` to refresh it from the scene's joints at the render boundary. The default and explicit-view encoders both prepare a private draw snapshot before starting GPU encoding. Use a command buffer with retained references so it owns the temporary line resources through execution. A draw-only refresh leaves the caller's graph and archived resources intact. Repeated draws of a changed pose can prepare new buffers each time. Use the explicit attachment refresh to publish a reusable snapshot for graph queries and serialization. [Metal command-buffer retention](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences).

`LineGeometry.hemisphereLight` and `DebugHelpers.hemisphereLight` draw the hemisphere light as an octahedral wireframe. Supply the same linear sky and ground colors and direction as the corresponding `LightProbe.hemisphere`. The sky tip follows that direction, the ground tip opposes it, and the four equator edges draw once in the ground color. The prepared helper has an explicit scene position and can follow a parent when added as its child. [Three.js r187 HemisphereLightHelper](https://github.com/mrdoob/three.js/blob/r187/src/helpers/HemisphereLightHelper.js).

## Verification

The pinned r187 runtime verified orthographic world-space origins in both depth modes, the five camera color groups, off-axis screen corners, and direct joint links, custom colors and pose updates. The native fixtures additionally check reversed-depth off-axis geometry, inherited visibility, cropped frustum pixels and an earlier encoded skeleton draw after a newer pose is published.

The r187 hemisphere fixture has six vertices and twelve unique octahedral edges, with opposing sky and ground sides. Native checks verify the authored direction, radius and two-tone raster output.

The Apple silicon Mac and iPad Pro M5 simulator each passed 44 focused tests across seven suites with Metal validation requested. The Xcode bridge could not create its arena on the full external volume. The native command-line runner used internal DerivedData.

With the hemisphere helper, the focused suite passed 53 tests across nine suites on each platform, including the hemisphere light probe tests.

The render-time refresh suite passed 65 tests across ten suites on the Mac and iPad simulator. Two commands encoded against different joint poses preserve their own pixels. Both camera entry points refresh the lines, and the original archive bytes and renderer identity remain stable. The unretained-buffer test checks rejection before encoding begins.
