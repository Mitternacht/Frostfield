import Metal

public extension PathTracer.Lighting {
    /// Snapshots the graph's visible analytic and rectangle lights and its shared
    /// environment. An explicit `sky` wins and keeps its own intensity and yaw.
    ///
    /// Scene fallback uses the graph's intensity and full XYZ orientation. The first use of an
    /// `EnvironmentLighting` prepares an importance-sampled panorama and waits
    /// for a GPU conversion on `queue`. Prepare outside the frame loop. Later
    /// snapshots reuse that panorama, including after intensity or rotation edits.
    /// A changed lighting snapshot requires a new accumulation history.
    init(
        queue: any MTLCommandQueue, graph: SceneGraph, sky: PathTracer.Environment? = nil,
        backgroundVisible: Bool = true, fog: PathTracer.Fog? = nil
    ) throws {
        try self.init(
            queue: queue, graph: graph, sky: sky, backgroundVisible: backgroundVisible,
            fogVolumes: fog.map { [$0] } ?? []
        )
    }

    /// Snapshots scene lighting with independently authored world-space fog volumes.
    init(
        queue: any MTLCommandQueue, graph: SceneGraph, sky: PathTracer.Environment? = nil,
        backgroundVisible: Bool = true, fogVolumes: [PathTracer.Fog]
    ) throws {
        var areas: [PathTracer.AreaLight] = []
        var analytic: [PathTracer.AnalyticLight] = []
        for node in try graph.resolve() where graph.renders(node) {
            guard let light = node.light else { continue }
            if case .rectangle = light.light.kind {
                areas.append(try PathTracer.AreaLight(light))
            } else {
                analytic.append(try PathTracer.AnalyticLight(light))
            }
        }

        let resolvedSky: PathTracer.Environment
        var additionalSkyEulerRotation = SIMD3<Float>.zero
        if let sky {
            if case .map(let map, _, _) = sky, map.device !== queue.device {
                throw PathTracer.Failure.environmentDevice
            }
            resolvedSky = sky
        } else if let source = graph.environment {
            let intensity = graph.environmentIntensity
            let yaw = graph.environmentRotation
            guard (0...65_504).contains(intensity), graph.effectiveEnvironmentEulerRotation.isFiniteVector else {
                throw PathTracer.Failure.invalidLighting
            }
            resolvedSky = .map(try source.pathEnvironmentMap(queue: queue), intensity: intensity, yaw: yaw)
            additionalSkyEulerRotation = graph.environmentEulerRotation
        } else {
            resolvedSky = .constant(.zero)
        }
        try self.init(
            lights: areas, analyticLights: analytic, sky: resolvedSky, additionalSkyEulerRotation: additionalSkyEulerRotation,
            backgroundVisible: backgroundVisible, fogVolumes: fogVolumes
        )
    }
}
