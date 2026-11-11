# Fixed-base FABRIK inverse kinematics

`FABRIKChain(joints:iterations:tolerance:)` takes scene node IDs from the
base through each direct child to the effector. `solve(in:target:)` returns
a posed graph, the remaining world-space error and the passes used. Keep
the input graph as the rest pose or supply the previous result to start
from the current pose. Only the rotating joints' local rotations change.
`PreparedGLTFScene.posing(_:)` rebuilds a prepared skin's joint palettes
from the returned graph.

The two reaching passes work on fixed-length segments in a frame centered
at the base. Each rotating joint uses its authored pivot when present.
The final endpoint is the effector's origin. Inherited orientation and
positive uniform scales allow these segments to keep the same world-space
lengths while the local rotations change. A finite straight chain can bend
toward a collinear interior target. Targets beyond maximum reach extend the
chain. Targets inside a nonzero minimum reach fold the shorter bones back
around the longest. The solver returns the closest radial pose in both
unreachable cases. The reported distance uses the returned rotations.

The public chain covers one unconstrained fixed-base 3D solve. It rejects
noncontiguous joints, coincident endpoints and transforms that violate
the scale or orientation contract. Connected chains, free bases, pole
targets and joint constraints are separate capabilities. The existing
`CCDIKChain` supplies constrained hinge and Euler solves.

The spatial fixture uses actual joint coordinates from
[pmndrs/math FABRIK3 at eda3d8c](https://github.com/pmndrs/math/blob/eda3d8c0380fd92ef66489d56d7a5ff96518f61c/src/ik/fabrik3.ts).
The reach construction follows [Aristidou and Lasenby](https://www.andreasaristidou.com/publications/papers/FABRIK.pdf).

The focused run passed 58 tests in 12 suites on macOS arm64 and on the iPad Pro
(M5) simulator with Metal API Validation. The suites are `FABRIKChainTests`,
`InverseKinematicsTests`, `ScenePivotTests`, `WorldOrientedNodeTests`,
`SceneAnimationTests`, `SceneGraphTests`, `GLTFSkinTests` and `MeshSkinTests`,
plus the glass shadow and sun suites that share the integrated tree. The
FABRIK fixture covers the pinned joint positions for one reachable target,
collinear folding, the inner and outer reach limits, nested pivots and scales,
and invalid inputs. Unreachable targets return the closest radial pose, and no
fixture compares them with pmndrs. Xcode's bridge reported a full workspace
volume, so verification used `xcodebuild` and the internal build caches.
