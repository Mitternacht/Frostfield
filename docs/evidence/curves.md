# Curves and rational spline surfaces

`Curve3D` evaluates positions over a normalized parameter range in Double
and returns Float vectors for the scene and mesh APIs. It exposes unit
tangents, approximate arc length, distance-fraction evaluation, and either
uniform parameter or distance sampling. Distance sampling uses a bounded
polyline table. Increase its resolution for sharply curved paths.

`Curve3D.cubicBezier` and `Curve3D.catmullRom` provide analytic tangents.
Catmull–Rom supports open or closed paths and uniform, centripetal and
chordal parameterization. The older `ProceduralPath.cubicBezier` sampling
API delegates to the same evaluator with its existing limits and errors.
The sampled points can feed `MeshGeometry.tube(along:radius:)`.
[Three.js r187 Catmull–Rom](https://github.com/mrdoob/three.js/blob/r187/src/extras/curves/CatmullRomCurve3.js)

`Curve3D.nurbs` evaluates positive weighted control points over the active
knot domain or a caller-selected knot window. The unit tangent uses the
analytic rational derivative. `NURBSSurface` evaluates a rectangular
tensor-product control net and tessellates regular patches with the existing
parametric mesh builder. Degree is bounded to 32 and control nets to
262,144 points. The regular mesh builder reports singular tangent bases.
[Three.js r187 NURBS curve](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/curves/NURBSCurve.js) ·
[Three.js r187 NURBS surface](https://github.com/mrdoob/three.js/blob/r187/examples/jsm/curves/NURBSSurface.js)

`Curve3DTests` compares all three Catmull–Rom modes in open and closed forms
against three@0.187.0 and checks repeated points and closed tangents. A
rational quadratic matches an analytic quarter circle and its tangents.
The fixtures also cover repeated knot endpoints, small knot intervals,
weighted surface tessellation, invalid inputs and sampled tube geometry.
The same checkpoint includes the complete tagged curve catalog and lofts
through sampled 3D cross sections. [Curve catalog](curve-extras.md) ·
[Loft evidence](loft-geometry.md)
