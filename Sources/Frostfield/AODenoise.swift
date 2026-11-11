import simd

public extension ShaderGraphBuilder {
    /// Filters the post-process source with a 16-tap luminance, depth and normal
    /// guided kernel. Coordinates select the nearest source texel. Neighbor
    /// taps with cleared depth, unusable normals or zero alpha are excluded.
    /// A center with any of those conditions passes through. The filter works in
    /// linear unassociated RGB and restores the center's original alpha.
    ///
    /// Use the inverse of the Metal projection that produced the source depth.
    /// Supply signed view-space RGB normals in a tracked, source-size RGBA16Float
    /// or RGBA32Float `.data` texture, or omit it to reconstruct normals from depth.
    /// Radius is in pixels (1...64), depthPhi is in view units, and larger positive
    /// lumaPhi/depthPhi admit larger differences. NormalPhi controls the falloff
    /// of the cosine similarity and must be positive. All default to five.
    /// Sample index selects one of four deterministic tap rotations (default zero).
    /// Invalid settings pass the center texel through.
    func denoise(
        at coordinates: ShaderValue<SIMD2<Float>>, inverseProjection: ShaderValue<simd_float4x4>,
        depthMode: MeshDepthMode = .forward, normalTexture: ShaderTexture? = nil,
        radius: ShaderValue<Float>? = nil, lumaPhi: ShaderValue<Float>? = nil,
        depthPhi: ShaderValue<Float>? = nil, normalPhi: ShaderValue<Float>? = nil,
        sampleIndex: ShaderValue<Float>? = nil
    ) throws -> ShaderValue<SIMD4<Float>> {
        try depthMode.requireProjectionDepth()
        if let normalTexture, normalTexture.builder !== self {
            throw MaterialGraph.Failure.invalidDeclaration("Denoise normals belong to a different graph builder.")
        }
        let defaultPhi = constant(Float(5))
        let settings = vector(radius ?? defaultPhi, lumaPhi ?? defaultPhi, depthPhi ?? defaultPhi, normalPhi ?? defaultPhi)
        let sampleIndex = sampleIndex ?? constant(Float(0))
        return add(
            .denoise(
                coordinates: coordinates.id, inverseProjection: inverseProjection.id, settings: settings.id,
                sampleIndex: sampleIndex.id, normalSlot: normalTexture?.slot, reversedDepth: depthMode == .reversed
            ),
            operands: [coordinates, inverseProjection, settings, sampleIndex]
        )
    }
}

public extension PostProcessGraph {
    /// A denoise stage for a noisy AO image in RGB, or another linear RGBA frame.
    /// Bind signed view-space normals by name when one is supplied. Without a
    /// normal texture the graph reconstructs them from its scene depth. Set the
    /// depth mode and projection to those used to render the source depth.
    /// Radius, LumaPhi, DepthPhi, NormalPhi and SampleIndex are live scalar uniforms.
    static func denoise(
        camera: ShaderGraphCamera, depthMode: MeshDepthMode = .forward, normalTextureName: String? = nil
    ) throws -> PostProcessGraph {
        try PostProcessGraph { p in
            let normals = normalTextureName.map {
                p.texture($0, content: .data, addressing: .clampToEdge, filtering: .nearest)
            }
            return try p.denoise(
                at: p.screenUV, inverseProjection: p.inverseProjection(of: camera),
                depthMode: depthMode, normalTexture: normals, radius: p.uniform("Radius", Float(5)),
                lumaPhi: p.uniform("LumaPhi", Float(5)), depthPhi: p.uniform("DepthPhi", Float(5)),
                normalPhi: p.uniform("NormalPhi", Float(5)), sampleIndex: p.uniform("SampleIndex", Float(0))
            )
        }
    }
}

enum AODenoiseSource {
    static func hook(normalSlots: [Int]) -> String {
        let normalCases = normalSlots.map { "        case \($0): return frostfield_graph_unit(r.texture\($0).read(uint2(pixel), 0).xyz);" }
        return """
        static inline float3 frostfield_post_denoise_normal(
            thread const FrostfieldGraphResources& r, int2 pixel, float2 uv,
            float4x4 inverseProjection, bool reversed, int normalSlot
        ) {
            switch (normalSlot) {
        \(normalCases.joined(separator: "\n"))
                default: return frostfield_graph_depth_normal(r, uv, inverseProjection, reversed);
            }
        }

        static inline float4 frostfield_graph_denoise(
            thread const FrostfieldGraphResources& r, float2 uv, float4x4 inverseProjection,
            float4 settings, float sampleIndex, int normalSlot, bool reversed
        ) {
            int2 size(r.scene.get_width(), r.scene.get_height());
            float2 inverseSize = 1.0f / float2(size);
            if (!all(frostfield_graph_finite(uv))) return float4(0);
            int2 pixel = min(int2(saturate(uv) * float2(size)), size - 1);
            float4 center = r.scene.read(uint2(pixel), 0);
            if (!(center.a > 0.0f) || !all(frostfield_graph_finite(center))
                || !all(frostfield_graph_finite(settings))
                || !(settings.x >= 1.0f && settings.x <= 64.0f)
                || !all(settings.yzw > 0.0f)
                || !(sampleIndex >= 0.0f && sampleIndex < 4.0f)) {
                return center;
            }

            float clearDepth = reversed ? 0.0f : 1.0f;
            float depth = r.depth.read(uint2(pixel));
            float2 centerUV = (float2(pixel) + 0.5f) * inverseSize;
            float3 position;
            if (!frostfield_post_has_depth(depth, clearDepth)
                || !frostfield_post_try_view_position(centerUV, depth, inverseProjection, position)) return center;
            float3 normal = frostfield_post_denoise_normal(r, pixel, centerUV, inverseProjection, reversed, normalSlot);
            if (dot(normal, normal) == 0.0f) return center;

            constexpr float3 luminance = float3(0.2126f, 0.7152f, 0.0722f);
            float3 color = center.rgb / center.a;
            float centerLuma = dot(color, luminance);
            if (!frostfield_graph_finite(centerLuma)) return center;
            // Stable per-pixel spiral rotation. Each phase uses a distinct hash seed.
            uint hash = uint(pixel.x) * 0x1f123bb5u ^ uint(pixel.y) * 0x5f356495u
                ^ uint(sampleIndex) * 0x9e3779b9u;
            hash ^= hash >> 16;
            hash *= 0x7feb352du;
            hash ^= hash >> 15;
            float angle = float(hash & 0x00ffffffu) * (6.283185307179586f / 16777216.0f);

            float3 correction = float3(0);
            float total = 1.0f;
            for (uint tap = 0; tap < 16; ++tap) {
                float direction = angle + float(tap) * 0.785398163397448f;
                float radius = mix(1.0f, settings.x, float(tap) / 15.0f);
                int2 neighbor = pixel + int2(round(radius * float2(cos(direction), sin(direction))));
                if (any(neighbor < 0) || any(neighbor >= size)) continue;
                float sampleDepth = r.depth.read(uint2(neighbor));
                if (!frostfield_post_has_depth(sampleDepth, clearDepth)) continue;
                float4 value = r.scene.read(uint2(neighbor), 0);
                if (!(value.a > 0.0f) || !all(frostfield_graph_finite(value))) continue;
                float2 sampleUV = (float2(neighbor) + 0.5f) * inverseSize;
                float3 samplePosition;
                if (!frostfield_post_try_view_position(sampleUV, sampleDepth, inverseProjection, samplePosition)) continue;
                float planeGap = abs(dot(position - samplePosition, normal));
                float depthWeight = max(1.0f - planeGap / settings.z, 0.0f);
                if (!(depthWeight > 0.0f)) continue;

                float3 sampleColor = value.rgb / value.a;
                float sampleLuma = dot(sampleColor, luminance);
                if (!frostfield_graph_finite(sampleLuma)) continue;
                float lumaWeight = max(1.0f - abs(sampleLuma - centerLuma) / settings.y, 0.0f);
                if (!(lumaWeight > 0.0f)) continue;

                float3 sampleNormal = frostfield_post_denoise_normal(
                    r, neighbor, sampleUV, inverseProjection, reversed, normalSlot
                );
                float similarity = saturate(dot(normal, sampleNormal));
                if (!(similarity > 0.0f)) continue;
                float weight = lumaWeight * depthWeight * pow(similarity, settings.w);
                correction += (sampleColor - color) * weight;
                total += weight;
            }
            // Accumulating around the center keeps a constant image bit-exact.
            return float4(center.rgb + (correction / total) * center.a, center.a);
        }
        """
    }
}
