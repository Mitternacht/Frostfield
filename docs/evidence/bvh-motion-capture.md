# BVH motion capture

`BVHLoader` imports a local file, UTF-8 bytes, or a string. The result contains
a rest-pose `SceneNode` hierarchy and a playable `AnimationClip` bound to its
node IDs. End sites stay in the tree as terminal offsets. Position and rotation
tracks can be enabled separately.

```swift
let motion = try BVHLoader.load(contentsOf: file)
let rest = SceneGraph(roots: [motion.root], camera: camera)
var mixer = AnimationMixer()
let action = try mixer.addAction(for: motion.clip)
try mixer.play(action)
try mixer.update(deltaTime: deltaTime)
let posed = try mixer.apply(to: rest)
```

A translation channel adds to the joint's offset. Rotations compose in file
channel order, and the first key is at zero seconds. The last key is at
`(frameCount - 1) * frameTime`. Between keys the mixer interpolates translation
linearly and rotates along the shortest quaternion arc. Source distances and
axes are preserved. Imported clips target the imported skeleton's IDs.
Retargeting, character meshes, skin weights, and coordinate or unit conversion
are separate authoring steps.
[Three.js r187 BVHLoader](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/loaders/BVHLoader.js)

The parser accepts a single root and one motion row per frame. It reports
unknown or duplicate channels, invalid numeric values or timing, and short
or surplus rows. Limits are 64 MiB of UTF-8 input, 4,096 nodes including end
sites, 256 hierarchy levels, and 2,000,000 joint-frame pairs. Joint names are
single tokens. Repeated names remain distinct because tracks target node IDs.

`BVHLoaderTests` uses a branched skeleton with nonstandard rotation order and
translation on a child. It compares local quaternion and world-space end-site
poses at three keys and two intermediate times against Three.js r187
`BVHLoader` and `AnimationMixer`. Additional tests cover track options, source
forms, and malformed input.

A separate Mac probe imported the r187 pirouette fixture. Both engines produced
592 frames, 86 tracks, and a clip duration of 4.9249978 seconds. All 57 joint
and end-site world positions agreed within 0.00006 source units at each of
four sampled times.
[Three.js r187 pirouette](https://github.com/mrdoob/three.js/blob/r187/examples/models/bvh/pirouette.bvh)

The seven focused suites passed 47 tests on the Apple Silicon Mac and 47 on
the iPad Pro M5 simulator with Metal API Validation enabled.
