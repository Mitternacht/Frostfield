// Points, lines and sprites drawn by PrimitiveRenderers.swift. Every record below
// mirrors a packed Swift struct in that file; keep the layouts in lockstep.

constant bool frostfieldPrimitiveHasColors [[function_constant(20)]];
constant bool frostfieldPrimitiveHasMap [[function_constant(21)]];
constant bool frostfieldPrimitiveHasSoftDepth [[function_constant(22)]];

struct FrostfieldPrimitiveView {
    float4x4 clipFromLocal;
    // Origin x and y, then width and height, in framebuffer pixels.
    float4 viewport;
    // x: pixels per world unit at clip w = 1; y: 1 for reversed depth.
    float4 projection;
};

static inline float2 frostfieldPrimitivePixel(float4 clip, constant FrostfieldPrimitiveView& view) {
    float2 ndc = clip.xy / clip.w;
    return view.viewport.xy + float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f) * view.viewport.zw;
}

static inline float4 frostfieldPrimitiveClip(float2 pixel, float ndcDepth, float w, constant FrostfieldPrimitiveView& view) {
    float2 unit = (pixel - view.viewport.xy) / view.viewport.zw;
    return float4(float2(unit.x * 2.0f - 1.0f, 1.0f - unit.y * 2.0f) * w, ndcDepth * w, w);
}

// Positive in front of the near plane: z >= 0 for forward depth, z <= w for reversed depth.
static inline float frostfieldPrimitiveNearDistance(float4 clip, constant FrostfieldPrimitiveView& view) {
    return view.projection.y > 0.5f ? clip.w - clip.z : clip.z;
}

// A vertex outside every clip plane, for primitives that draw nothing.
constant float4 frostfieldPrimitiveCulled = float4(2.0f, 2.0f, 2.0f, 1.0f);

// Two triangles over the unit square, as top-left texture coordinates.
constant float2 frostfieldPrimitiveQuad[6] = {
    float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1)
};

// MARK: - Points and sprites

struct FrostfieldPrimitiveSoftParticle {
    float4 projection;
    float4 target;   // attachment size, window depth range
    float4 settings; // distance, contrast, normalized clear depth
};

static inline float frostfieldPrimitiveSoftFade(
    float4 position, depth2d<float> snapshot, constant FrostfieldPrimitiveSoftParticle& soft
) {
    constexpr sampler nearest(coord::normalized, address::clamp_to_edge, filter::nearest);
    float rawDepth = snapshot.sample(nearest, position.xy / soft.target.xy, level(0));
    float2 range = soft.target.zw;
    float depth = (rawDepth - range.x) / (range.y - range.x);
    if (!(depth >= 0.0f && depth <= 1.0f) || depth == soft.settings.z) return 1.0f;
    float particleZ = frostfieldParticleFragmentViewZ(position.zw, range, soft.projection);
    float sceneZ = frostfieldParticleViewZ(depth, soft.projection);
    if (!isfinite(particleZ) || !isfinite(sceneZ)) return 1.0f;
    return frostfieldParticleContrast(particleZ, sceneZ, soft.settings.x, soft.settings.y);
}

struct FrostfieldBillboardVaryings {
    float4 position [[position]];
    float2 uv;
    float4 color [[flat]];
    // Drawn radius in pixels, and the alpha scale of a footprint smaller than one pixel.
    float2 footprint [[flat]];
};

struct FrostfieldPointStyle {
    // Linear RGB and opacity, times the node tint.
    float4 color;
    // x: size in pixels or world units, y: 1 for world units, z: 1 for circles, w: alpha test.
    float4 size;
    // x: 1 to antialias circle edges.
    float4 flags;
};

vertex FrostfieldBillboardVaryings frostfield_points_vertex(
    uint vertexID [[vertex_id]],
    uint instance [[instance_id]],
    const device float4* points [[buffer(0)]],
    const device half4* colors [[buffer(1), function_constant(frostfieldPrimitiveHasColors)]],
    constant FrostfieldPrimitiveView& view [[buffer(2)]],
    constant FrostfieldPointStyle& style [[buffer(3)]]
) {
    FrostfieldBillboardVaryings out = {};
    float4 point = points[instance];
    float4 clip = view.clipFromLocal * float4(point.xyz, 1.0f);
    float size = style.size.x * point.w;
    // A zero per-point size hides the point in every blending mode.
    if (!(clip.w > 0.0f) || !(size > 0.0f)) {
        out.position = frostfieldPrimitiveCulled;
        return out;
    }
    float diameter = style.size.y > 0.5f ? size * view.projection.x / clip.w : size;
    // Footprints below one pixel draw one pixel wide with proportionally less alpha.
    float drawn = max(diameter, 1.0f);
    // Antialiased discs need a pixel of margin for their partially covered rim.
    float margin = style.size.z > 0.5f && style.flags.x > 0.5f ? 1.0f : 0.0f;
    float2 corner = frostfieldPrimitiveQuad[vertexID];
    float2 offset = (corner - 0.5f) * float2(1.0f, -1.0f) * (drawn + 2.0f * margin);
    float2 uv = (corner - 0.5f) * (drawn + 2.0f * margin) / drawn + 0.5f;
    clip.xy += offset * 2.0f / view.viewport.zw * clip.w;
    float4 color = style.color;
    if (frostfieldPrimitiveHasColors) {
        color *= float4(colors[instance]);
    }
    out.position = clip;
    out.uv = uv;
    out.color = color;
    float fraction = saturate(diameter);
    out.footprint = float2(drawn * 0.5f, fraction * fraction);
    return out;
}

#if defined(FROSTFIELD_PRIMITIVE_NODE)
static inline float4 frostfieldPrimitiveNodeColor(
    float4 color, float2 uv, constant float4* graphUniforms, constant FrostfieldGraphView& graphView
) {
    FrostfieldGraphResources graphResources = { graphUniforms, &graphView };
    FrostfieldSurfaceInput graphInput = {};
    graphInput.uv0 = uv;
    graphInput.baseColor = max(color.rgb, float3(0));
    graphInput.parameters = color;
    FrostfieldSurfaceOutput graphOutput = {
        graphInput.baseColor, 0.0f, 1.0f, float3(0), 0.0f, 1.0f, float3(0, 0, 1), false, 1.0f, -1.0f, true
    };
    frostfield_generated_surface(graphInput, graphResources, graphOutput);
    return float4(graphOutput.baseColor, saturate(color.a * saturate(graphOutput.opacity)));
}
#endif

fragment float4 frostfield_points_fragment(
    FrostfieldBillboardVaryings input [[stage_in]],
    constant FrostfieldPointStyle& style [[buffer(0)]],
    texture2d<float> map [[texture(0), function_constant(frostfieldPrimitiveHasMap)]],
    depth2d<float> snapshot [[texture(1), function_constant(frostfieldPrimitiveHasSoftDepth)]],
    constant FrostfieldPrimitiveSoftParticle& soft [[buffer(1), function_constant(frostfieldPrimitiveHasSoftDepth)]]
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    , constant float4* graphUniforms [[buffer(3)]]
    , constant FrostfieldGraphView& graphView [[buffer(4)]]
#endif
) {
    float4 color = input.color;
    if (frostfieldPrimitiveHasMap) {
        constexpr sampler mapSampler(address::clamp_to_edge, filter::linear, mip_filter::linear);
        color *= map.sample(mapSampler, input.uv);
    }
    if (frostfieldPrimitiveHasSoftDepth) {
        color.a *= frostfieldPrimitiveSoftFade(input.position, snapshot, soft);
    }
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    color = frostfieldPrimitiveNodeColor(color, input.uv, graphUniforms, graphView);
#endif
    if (color.a < style.size.w) {
        discard_fragment();
    }
    bool antialias = style.flags.x > 0.5f;
    float alpha = saturate(color.a) * (antialias ? input.footprint.y : 1.0f);
    if (style.size.z > 0.5f) {
        float radius = input.footprint.x;
        float distance = length(input.uv * 2.0f - 1.0f) * radius;
        if (antialias) {
            alpha *= saturate(radius - distance + 0.5f);
        } else if (distance > radius) {
            discard_fragment();
        }
    }
    if (alpha <= 0.0f) {
        discard_fragment();
    }
    return float4(max(color.rgb, float3(0)) * alpha, alpha);
}

struct FrostfieldSpriteRecord {
    // World position in xyz, screen rotation in radians in w.
    float4 positionRotation;
    // Size in pixels or world units in xy, top-left anchor in zw.
    float4 sizeCenter;
    float4 color;
    // Texture region origin in xy and size in zw.
    float4 region;
};

struct FrostfieldSpriteStyle {
    float4 color;
    // x: 1 for world units, y: alpha test.
    float4 parameters;
};

vertex FrostfieldBillboardVaryings frostfield_sprites_vertex(
    uint vertexID [[vertex_id]],
    uint instance [[instance_id]],
    const device FrostfieldSpriteRecord* sprites [[buffer(0)]],
    constant FrostfieldPrimitiveView& view [[buffer(1)]],
    constant FrostfieldSpriteStyle& style [[buffer(2)]]
) {
    FrostfieldBillboardVaryings out = {};
    FrostfieldSpriteRecord sprite = sprites[instance];
    float4 clip = view.clipFromLocal * float4(sprite.positionRotation.xyz, 1.0f);
    if (!(clip.w > 0.0f)) {
        out.position = frostfieldPrimitiveCulled;
        return out;
    }
    float2 size = style.parameters.x > 0.5f ? sprite.sizeCenter.xy * view.projection.x / clip.w : sprite.sizeCenter.xy;
    float2 corner = frostfieldPrimitiveQuad[vertexID];
    // Pixels from the anchor with +y up; texture coordinates grow downward.
    float2 local = float2(corner.x - sprite.sizeCenter.z, sprite.sizeCenter.w - corner.y) * size;
    float angle = sprite.positionRotation.w;
    float c = cos(angle);
    float s = sin(angle);
    float2 rotated = float2(c * local.x - s * local.y, s * local.x + c * local.y);
    clip.xy += rotated * 2.0f / view.viewport.zw * clip.w;
    out.position = clip;
    out.uv = sprite.region.xy + corner * sprite.region.zw;
    out.color = style.color * sprite.color;
    out.footprint = float2(0.0f, 1.0f);
    return out;
}

fragment float4 frostfield_sprites_fragment(
    FrostfieldBillboardVaryings input [[stage_in]],
    constant FrostfieldSpriteStyle& style [[buffer(0)]],
    texture2d<float> map [[texture(0), function_constant(frostfieldPrimitiveHasMap)]],
    depth2d<float> snapshot [[texture(1), function_constant(frostfieldPrimitiveHasSoftDepth)]],
    constant FrostfieldPrimitiveSoftParticle& soft [[buffer(1), function_constant(frostfieldPrimitiveHasSoftDepth)]]
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    , constant float4* graphUniforms [[buffer(3)]]
    , constant FrostfieldGraphView& graphView [[buffer(4)]]
#endif
) {
    float4 color = input.color;
    if (frostfieldPrimitiveHasMap) {
        constexpr sampler mapSampler(address::clamp_to_edge, filter::linear, mip_filter::linear);
        color *= map.sample(mapSampler, input.uv);
    }
    if (frostfieldPrimitiveHasSoftDepth) {
        color.a *= frostfieldPrimitiveSoftFade(input.position, snapshot, soft);
    }
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    color = frostfieldPrimitiveNodeColor(color, input.uv, graphUniforms, graphView);
#endif
    if (color.a < style.parameters.y) {
        discard_fragment();
    }
    float alpha = saturate(color.a);
    if (alpha <= 0.0f) {
        discard_fragment();
    }
    return float4(max(color.rgb, float3(0)) * alpha, alpha);
}

// MARK: - Lines

struct FrostfieldLineSegment {
    // Previous, start, end and next point indices; negative when absent.
    int4 points;
    // Line distance at the start and end.
    float2 distances;
    // The segments joined at the start and end; negative when absent.
    int2 neighbors;
};

struct FrostfieldLineStyle {
    float4 color;
    // x: half width in pixels or world units, y: 1 for world units, z: 1 to antialias, w: miter limit.
    float4 width;
    // x: join (0 miter, 1 round, 2 bevel), y: cap (0 butt, 1 round, 2 square),
    // z: 1 for additive blending, w: transverse raster padding in pixels.
    float4 shape;
    // Dash length, gap length and offset in line-distance units; w is 1 when dashed.
    float4 dash;
};

// What lies past one end of a segment, besides the join kinds 0 (miter), 1 (round) and 2 (bevel).
// A neighbour's far join is not modelled: nothing of that neighbour lies past its far end.
constant float frostfieldLineEndNothing = -3.0f;
// A butt end where the segment crosses the near plane.
constant float frostfieldLineEndTrimmed = -2.0f;
constant float frostfieldLineEndCap = -1.0f;

struct FrostfieldLineVaryings {
    float4 position [[position]];
    // Segment start and end in framebuffer pixels.
    float4 ends [[flat]];
    // Half widths in pixels at the start and end.
    float2 halfWidths [[flat]];
    // Join bisector normal pointing along the line (xy) and outer miter direction (zw).
    float4 startJoin [[flat]];
    float4 endJoin [[flat]];
    // End kind at the start and end, then the cosine of half the turning angle at each.
    float4 joins [[flat]];
    // 1/w at the start and end, then the line distance at each.
    float4 interpolation [[flat]];
    float4 startColor [[flat]];
    float4 endColor [[flat]];
    // The joined neighbours' far ends: pixel position, half width in pixels and 1/w.
    float4 previousEnd [[flat]];
    float4 nextEnd [[flat]];
    // Each neighbour's line distance at its far end and at the shared point, its
    // opacity at the far end, and its far end kind.
    float4 previousInfo [[flat]];
    float4 nextInfo [[flat]];
};

struct FrostfieldLineJoin {
    float kind;
    float2 bisector;
    float2 miter;
    float cosineHalfTurn;
    float reach;
};

// The join at `vertex` between the incoming and outgoing screen directions.
static inline FrostfieldLineJoin frostfieldLineJoin(float2 incoming, float2 outgoing, float2 normal, float halfWidth,
                                                     constant FrostfieldLineStyle& style) {
    FrostfieldLineJoin join = { frostfieldLineEndCap, float2(0), float2(0), 1.0f, halfWidth + 1.0f };
    float2 tangent = incoming + outgoing;
    float tangentLength = length(tangent);
    if (!(tangentLength > 1e-3f)) {
        return join;
    }
    join.bisector = tangent / tangentLength;
    join.cosineHalfTurn = saturate(dot(join.bisector, outgoing));
    float2 outer = incoming - outgoing;
    float outerLength = length(outer);
    join.miter = outerLength > 1e-6f ? outer / outerLength : normal;
    join.kind = style.shape.x;
    if (join.kind < 0.5f) {
        // SVG semantics: beyond the limit, the miter becomes a bevel.
        if (join.cosineHalfTurn * style.width.w < 1.0f) {
            join.kind = 2.0f;
        } else {
            float cosine = max(join.cosineHalfTurn, 1e-3f);
            join.reach = (halfWidth + 1.0f) * max(1.0f, sqrt(max(1.0f / (cosine * cosine) - 1.0f, 0.0f)));
        }
    }
    return join;
}

static inline float frostfieldLineHalfWidth(float4 clip, constant FrostfieldPrimitiveView& view,
                                            constant FrostfieldLineStyle& style) {
    return style.width.y > 0.5f ? style.width.x * view.projection.x / clip.w : style.width.x;
}

vertex FrostfieldLineVaryings frostfield_line_vertex(
    uint vertexID [[vertex_id]],
    uint instance [[instance_id]],
    const device float4* points [[buffer(0)]],
    const device half4* colors [[buffer(1), function_constant(frostfieldPrimitiveHasColors)]],
    const device FrostfieldLineSegment* segments [[buffer(2)]],
    constant FrostfieldPrimitiveView& view [[buffer(3)]],
    constant FrostfieldLineStyle& style [[buffer(4)]]
) {
    FrostfieldLineVaryings out = {};
    FrostfieldLineSegment segment = segments[instance];
    float4 a = view.clipFromLocal * float4(points[segment.points.y].xyz, 1.0f);
    float4 b = view.clipFromLocal * float4(points[segment.points.z].xyz, 1.0f);
    float nearA = frostfieldPrimitiveNearDistance(a, view);
    float nearB = frostfieldPrimitiveNearDistance(b, view);
    if (nearA < 0.0f && nearB < 0.0f) {
        out.position = frostfieldPrimitiveCulled;
        return out;
    }
    float distanceA = segment.distances.x;
    float distanceB = segment.distances.y;
    float4 colorA = style.color;
    float4 colorB = style.color;
    if (frostfieldPrimitiveHasColors) {
        colorA *= float4(colors[segment.points.y]);
        colorB *= float4(colors[segment.points.z]);
    }
    bool hasPrevious = segment.points.x >= 0;
    bool hasNext = segment.points.w >= 0;
    // Trim the segment at the near plane in clip space, which is linear in world space.
    // A trimmed end has no join and gets a butt end.
    bool trimmedA = false;
    bool trimmedB = false;
    if (nearA < 0.0f) {
        float t = nearA / (nearA - nearB);
        a = mix(a, b, t);
        distanceA = mix(distanceA, distanceB, t);
        colorA = mix(colorA, colorB, t);
        hasPrevious = false;
        trimmedA = true;
    } else if (nearB < 0.0f) {
        float t = nearB / (nearB - nearA);
        b = mix(b, a, t);
        distanceB = mix(distanceB, distanceA, t);
        colorB = mix(colorB, colorA, t);
        hasNext = false;
        trimmedB = true;
    }

    float2 pixelA = frostfieldPrimitivePixel(a, view);
    float2 pixelB = frostfieldPrimitivePixel(b, view);
    float halfA = frostfieldLineHalfWidth(a, view, style);
    float halfB = frostfieldLineHalfWidth(b, view, style);
    bool antialias = style.width.z > 0.5f;
    float drawnA = antialias ? max(halfA, 0.5f) : halfA;
    float drawnB = antialias ? max(halfB, 0.5f) : halfB;

    float2 delta = pixelB - pixelA;
    float length = metal::length(delta);
    float2 direction = length > 1e-4f ? delta / length : float2(1, 0);
    float2 normal = float2(-direction.y, direction.x);

    // Each join also records the neighbouring segment's far end, so the fragment
    // stage can tell where the neighbour's own stroke reaches.
    FrostfieldLineJoin start = { frostfieldLineEndCap, float2(0), float2(0), 1.0f, drawnA + 1.0f };
    if (hasPrevious && length > 1e-4f) {
        uint index = uint(segment.points.x);
        float4 previous = view.clipFromLocal * float4(points[index].xyz, 1.0f);
        float nearPrevious = frostfieldPrimitiveNearDistance(previous, view);
        float trim = 0.0f;
        if (nearPrevious < 0.0f) {
            trim = nearPrevious / (nearPrevious - nearA);
            previous = mix(previous, a, trim);
        }
        float2 previousPixel = frostfieldPrimitivePixel(previous, view);
        float2 incoming = pixelA - previousPixel;
        float incomingLength = metal::length(incoming);
        if (incomingLength > 1e-4f) {
            start = frostfieldLineJoin(incoming / incomingLength, direction, normal, drawnA, style);
            bool known = segment.neighbors.x >= 0;
            FrostfieldLineSegment neighbor = known ? segments[segment.neighbors.x] : segment;
            float2 distances = known ? neighbor.distances : float2(distanceA);
            float opacity = style.color.a;
            if (frostfieldPrimitiveHasColors) {
                opacity *= mix(float(colors[index].a), float(colors[segment.points.y].a), trim);
            }
            float kind = trim > 0.0f ? frostfieldLineEndTrimmed
                : (known && neighbor.points.x < 0 ? frostfieldLineEndCap : frostfieldLineEndNothing);
            out.previousEnd = float4(previousPixel, frostfieldLineHalfWidth(previous, view, style), 1.0f / previous.w);
            out.previousInfo = float4(mix(distances.x, distances.y, trim), distances.y, opacity, kind);
        }
    }
    FrostfieldLineJoin end = { frostfieldLineEndCap, float2(0), float2(0), 1.0f, drawnB + 1.0f };
    if (hasNext && length > 1e-4f) {
        uint index = uint(segment.points.w);
        float4 next = view.clipFromLocal * float4(points[index].xyz, 1.0f);
        float nearNext = frostfieldPrimitiveNearDistance(next, view);
        float trim = 0.0f;
        if (nearNext < 0.0f) {
            trim = nearNext / (nearNext - nearB);
            next = mix(next, b, trim);
        }
        float2 nextPixel = frostfieldPrimitivePixel(next, view);
        float2 outgoing = nextPixel - pixelB;
        float outgoingLength = metal::length(outgoing);
        if (outgoingLength > 1e-4f) {
            end = frostfieldLineJoin(direction, outgoing / outgoingLength, normal, drawnB, style);
            bool known = segment.neighbors.y >= 0;
            FrostfieldLineSegment neighbor = known ? segments[segment.neighbors.y] : segment;
            float2 distances = known ? neighbor.distances : float2(distanceB);
            float opacity = style.color.a;
            if (frostfieldPrimitiveHasColors) {
                opacity *= mix(float(colors[index].a), float(colors[segment.points.z].a), trim);
            }
            float kind = trim > 0.0f ? frostfieldLineEndTrimmed
                : (known && neighbor.points.w < 0 ? frostfieldLineEndCap : frostfieldLineEndNothing);
            out.nextEnd = float4(nextPixel, frostfieldLineHalfWidth(next, view, style), 1.0f / next.w);
            out.nextInfo = float4(mix(distances.y, distances.x, trim), distances.x, opacity, kind);
        }
    }
    // A butt end at the near plane needs no extension past it.
    float startReach = trimmedA ? 1.0f : start.reach;
    float endReach = trimmedB ? 1.0f : end.reach;

    // One strip of three quads: the start region, the body and the end region.
    // Each end region keeps its endpoint's depth, so caps and joins stay at the
    // vertex depth however steeply the segment recedes; across the body, depth
    // is affine in screen space along the segment and exact.
    uint column = vertexID >> 1;
    float side = (vertexID & 1u) != 0 ? 1.0f : -1.0f;
    float along = column == 0 ? -startReach : (column == 1 ? 0.0f : (column == 2 ? length : length + endReach));
    float across = max(drawnA, drawnB) + style.shape.w;
    float2 pixel = pixelA + direction * along + normal * side * across;
    float4 endpoint = column >= 2 ? b : a;
    out.position = frostfieldPrimitiveClip(pixel, endpoint.z / endpoint.w, endpoint.w, view);
    out.ends = float4(pixelA, pixelB);
    out.halfWidths = float2(halfA, halfB);
    out.startJoin = float4(start.bisector, start.miter);
    out.endJoin = float4(end.bisector, end.miter);
    out.joins = float4(trimmedA ? frostfieldLineEndTrimmed : start.kind, trimmedB ? frostfieldLineEndTrimmed : end.kind,
                       start.cosineHalfTurn, end.cosineHalfTurn);
    out.interpolation = float4(1.0f / a.w, 1.0f / b.w, distanceA, distanceB);
    out.startColor = colorA;
    out.endColor = colorB;
    return out;
}

// Signed distance in pixels past the edge of the region around one end of a segment.
static inline float frostfieldLineEndDistance(
    float2 offset, float2 outward, float2 normal, float halfWidth, float kind, float2 miter, float cosineHalfTurn, float cap
) {
    float beyond = dot(offset, outward);
    float across = abs(dot(offset, normal));
    if (kind < -1.5f) {
        // Trimmed at the near plane.
        return max(across - halfWidth, beyond);
    }
    if (kind < -0.5f) {
        if (cap < 0.5f) {
            return max(across - halfWidth, beyond);
        }
        if (cap < 1.5f) {
            return length(offset) - halfWidth;
        }
        return max(across - halfWidth, beyond - halfWidth);
    }
    if (kind < 0.5f) {
        // A miter is the stroke continued up to the bisector.
        return across - halfWidth;
    }
    if (kind < 1.5f) {
        return length(offset) - halfWidth;
    }
    return max(across - halfWidth, dot(offset, miter) - halfWidth * cosineHalfTurn);
}

// One segment's stroke as the fragment stage sees it, in framebuffer pixels.
struct FrostfieldLineShape {
    float2 start;
    float2 end;
    // Half widths at the start and end.
    float2 halfWidths;
    // Join bisector (xy) and outer miter direction (zw) at the start and end.
    float4 startJoin;
    float4 endJoin;
    // End kinds at the start and end, then the cosine of half the turning angle at each.
    float4 joins;
    // 1/w at the start and end, then the line distance at each.
    float4 interpolation;
    // Color opacity at the start and end.
    float2 opacities;
};

// Where a pixel falls relative to a segment's centerline.
struct FrostfieldLinePosition {
    float2 direction;
    float2 normal;
    float length;
    // Distance along the centerline from the start, in pixels.
    float along;
    // The nearest centerline point's screen parameter and perspective-correct parameter.
    float u;
    float t;
};

static inline FrostfieldLinePosition frostfieldLinePosition(float2 pixel, FrostfieldLineShape shape) {
    FrostfieldLinePosition position;
    float2 delta = shape.end - shape.start;
    position.length = metal::length(delta);
    position.direction = position.length > 1e-4f ? delta / position.length : float2(1, 0);
    position.normal = float2(-position.direction.y, position.direction.x);
    position.along = dot(pixel - shape.start, position.direction);
    position.u = position.length > 1e-4f ? saturate(position.along / position.length) : 0.0f;
    float inverse = mix(shape.interpolation.x, shape.interpolation.y, position.u);
    position.t = inverse > 0.0f ? position.u * shape.interpolation.y / inverse : position.u;
    return position;
}

// Whether an end of the given kind is cut square at the endpoint: a trimmed end or a butt cap.
static inline bool frostfieldLineButtEnd(float kind, constant FrostfieldLineStyle& style) {
    bool trimmed = kind < -1.5f && kind > -2.5f;
    bool cap = kind < -0.5f && kind > -1.5f;
    return trimmed || (cap && style.shape.y < 0.5f);
}

// The alpha a segment's stroke gives a pixel: coverage, dashes and opacity.
// An end region past a join counts only on this segment's side of the bisector.
static inline float frostfieldLineAlpha(
    float2 pixel, FrostfieldLineShape shape, FrostfieldLinePosition position, constant FrostfieldLineStyle& style,
    float distanceStep, bool ownsStart, bool ownsEnd
) {
    bool antialias = style.width.z > 0.5f;
    float halfWidth = mix(shape.halfWidths.x, shape.halfWidths.y, position.u);
    float startHalf = shape.halfWidths.x;
    float endHalf = shape.halfWidths.y;
    float alpha = 1.0f;
    if (antialias) {
        // Lines thinner than a pixel draw one pixel wide with proportionally less alpha.
        alpha = saturate(2.0f * halfWidth);
        halfWidth = max(halfWidth, 0.5f);
        startHalf = max(startHalf, 0.5f);
        endHalf = max(endHalf, 0.5f);
    }
    float edge;
    if (position.along < 0.0f) {
        if (!ownsStart || shape.joins.x < -2.5f) {
            return 0.0f;
        }
        edge = frostfieldLineEndDistance(pixel - shape.start, -position.direction, position.normal, startHalf,
                                         shape.joins.x, shape.startJoin.zw, shape.joins.z, style.shape.y);
    } else if (position.along > position.length) {
        if (!ownsEnd || shape.joins.y < -2.5f) {
            return 0.0f;
        }
        edge = frostfieldLineEndDistance(pixel - shape.end, position.direction, position.normal, endHalf,
                                         shape.joins.y, shape.endJoin.zw, shape.joins.w, style.shape.y);
    } else {
        edge = abs(dot(pixel - shape.start, position.normal)) - halfWidth;
        // A butt end cuts the stroke square, so its edge ramps on both sides.
        if (frostfieldLineButtEnd(shape.joins.x, style)) {
            edge = max(edge, -position.along);
        }
        if (frostfieldLineButtEnd(shape.joins.y, style)) {
            edge = max(edge, position.along - position.length);
        }
    }
    if (antialias) {
        alpha *= saturate(0.5f - edge);
        if (!(alpha > 0.0f)) {
            return 0.0f;
        }
    } else if (edge > 0.0f) {
        return 0.0f;
    }

    if (style.dash.w > 0.5f) {
        float lineDistance = mix(shape.interpolation.z, shape.interpolation.w, position.t);
        float period = style.dash.x + style.dash.y;
        float phase = fmod(lineDistance + style.dash.z, period);
        if (phase < 0.0f) {
            phase += period;
        }
        // Signed distance to the nearest dash boundary, positive inside a dash.
        float inside = phase < style.dash.x ? min(phase, style.dash.x - phase)
                                            : -min(phase - style.dash.x, period - phase);
        if (antialias && distanceStep > 0.0f) {
            alpha *= saturate(inside / distanceStep + 0.5f);
        } else if (inside < 0.0f) {
            return 0.0f;
        }
    }
    return alpha * saturate(mix(shape.opacities.x, shape.opacities.y, position.t));
}

fragment float4 frostfield_line_fragment(
    FrostfieldLineVaryings input [[stage_in]],
    constant FrostfieldLineStyle& style [[buffer(0)]]
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    , constant float4* graphUniforms [[buffer(3)]]
    , constant FrostfieldGraphView& graphView [[buffer(4)]]
#endif
) {
    float2 pixel = input.position.xy;
    FrostfieldLineShape own = {
        input.ends.xy, input.ends.zw, input.halfWidths, input.startJoin, input.endJoin, input.joins,
        input.interpolation, float2(input.startColor.a, input.endColor.a)
    };
    FrostfieldLinePosition position = frostfieldLinePosition(pixel, own);
    // Dash edges antialias over the distance's screen derivative, taken before any discard.
    float distanceStep = fwidth(mix(own.interpolation.z, own.interpolation.w, position.t));
    // Each join is split along its bisector. On its own side a segment draws its
    // stroke; on the neighbour's side the neighbour draws first.
    bool ownsStart = own.joins.x < -0.5f || dot(pixel - own.start, own.startJoin.xy) >= 0.0f;
    bool ownsEnd = own.joins.y < -0.5f || dot(pixel - own.end, own.endJoin.xy) < 0.0f;
    float alpha = frostfieldLineAlpha(pixel, own, position, style, distanceStep, ownsStart, ownsEnd);
    if (!(alpha > 0.0f)) {
        discard_fragment();
    }
    if (!(ownsStart && ownsEnd)) {
        // Past a sharp turn this segment's stroke can reach beyond a short
        // neighbour's. Draw only what the neighbour leaves uncovered, so the two
        // composite to the larger of their alphas: no gap and no double blend.
        bool additive = style.shape.z > 0.5f;
        float drawn = 0.0f;
        if (!ownsStart) {
            FrostfieldLineShape previous = {
                input.previousEnd.xy, own.start, float2(input.previousEnd.z, own.halfWidths.x),
                float4(0), own.startJoin, float4(input.previousInfo.w, own.joins.x, 1.0f, own.joins.z),
                float4(input.previousEnd.w, own.interpolation.x, input.previousInfo.x, input.previousInfo.y),
                float2(input.previousInfo.z, own.opacities.x)
            };
            drawn = frostfieldLineAlpha(pixel, previous, frostfieldLinePosition(pixel, previous), style,
                                        distanceStep, true, true);
        }
        if (!ownsEnd) {
            FrostfieldLineShape next = {
                own.end, input.nextEnd.xy, float2(own.halfWidths.y, input.nextEnd.z),
                own.endJoin, float4(0), float4(own.joins.y, input.nextInfo.w, own.joins.w, 1.0f),
                float4(own.interpolation.y, input.nextEnd.w, input.nextInfo.y, input.nextInfo.x),
                float2(own.opacities.y, input.nextInfo.z)
            };
            float neighbour = frostfieldLineAlpha(pixel, next, frostfieldLinePosition(pixel, next), style,
                                                  distanceStep, true, true);
            drawn = additive ? drawn + neighbour : drawn + neighbour - drawn * neighbour;
        }
        float target = max(alpha, drawn);
        alpha = additive ? target - drawn : (drawn < 1.0f ? (target - drawn) / (1.0f - drawn) : 0.0f);
        if (!(alpha > 1e-6f)) {
            discard_fragment();
        }
    }
    float4 color = mix(input.startColor, input.endColor, position.t);
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    color = frostfieldPrimitiveNodeColor(color, float2(position.t, 0.0f), graphUniforms, graphView);
#endif
    return float4(max(color.rgb, float3(0)) * alpha, alpha);
}

struct FrostfieldHairlineVaryings {
    float4 position [[position]];
    float4 color;
    float lineDistance;
};

vertex FrostfieldHairlineVaryings frostfield_hairline_vertex(
    uint vertexID [[vertex_id]],
    const device float4* points [[buffer(0)]],
    const device half4* colors [[buffer(1), function_constant(frostfieldPrimitiveHasColors)]],
    const device FrostfieldLineSegment* segments [[buffer(2)]],
    constant FrostfieldPrimitiveView& view [[buffer(3)]],
    constant FrostfieldLineStyle& style [[buffer(4)]]
) {
    FrostfieldLineSegment segment = segments[vertexID / 2];
    bool isEnd = (vertexID & 1u) != 0;
    int index = isEnd ? segment.points.z : segment.points.y;
    FrostfieldHairlineVaryings out;
    out.position = view.clipFromLocal * float4(points[index].xyz, 1.0f);
    float4 color = style.color;
    if (frostfieldPrimitiveHasColors) {
        color *= float4(colors[index]);
    }
    out.color = color;
    out.lineDistance = isEnd ? segment.distances.y : segment.distances.x;
    return out;
}

fragment float4 frostfield_hairline_fragment(
    FrostfieldHairlineVaryings input [[stage_in]],
    constant FrostfieldLineStyle& style [[buffer(0)]]
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    , constant float4* graphUniforms [[buffer(3)]]
    , constant FrostfieldGraphView& graphView [[buffer(4)]]
#endif
) {
    if (style.dash.w > 0.5f) {
        float period = style.dash.x + style.dash.y;
        float phase = fmod(input.lineDistance + style.dash.z, period);
        if (phase < 0.0f) {
            phase += period;
        }
        if (phase >= style.dash.x) {
            discard_fragment();
        }
    }
    float alpha = saturate(input.color.a);
#if defined(FROSTFIELD_PRIMITIVE_NODE)
    float4 color = frostfieldPrimitiveNodeColor(input.color, float2(0), graphUniforms, graphView);
    alpha = saturate(color.a);
#else
    float4 color = input.color;
#endif
    if (alpha <= 0.0f) {
        discard_fragment();
    }
    return float4(max(color.rgb, float3(0)) * alpha, alpha);
}
