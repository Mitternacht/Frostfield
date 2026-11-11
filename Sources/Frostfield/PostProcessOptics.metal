#include <metal_stdlib>
using namespace metal;

// Lens and scene-aware effects: bokeh depth of field, god rays, lens flare and
// selection outlines. Color textures hold premultiplied linear RGBA. Depth
// parameters are (near, far, reversed) for the camera that stored the depth.

constexpr sampler ff_optics_linear(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler ff_optics_nearest(coord::normalized, address::clamp_to_edge, filter::nearest);

static inline float4 ff_optics_finite(float4 c) {
    return float4(clamp(select(float3(0), c.rgb, isfinite(c.rgb)), float3(0), float3(65504.0f)),
                  isfinite(c.a) ? saturate(c.a) : 0.0f);
}

static inline float ff_distance(float stored, float4 d) {
    float forward = d.z > 0.5f ? 1.0f - stored : stored;
    return d.x * d.y / (d.y - forward * (d.y - d.x));
}

// Radius from the center to a regular polygon's boundary along `angle`, relative to the circumradius.
static inline float ff_polygon(float angle, float blades, float rotation) {
    if (blades < 3.0f) { return 1.0f; }
    float sector = 2.0f * M_PI_F / blades;
    float a = angle - rotation;
    a -= sector * floor(a / sector);
    return cos(M_PI_F / blades) / cos(a - M_PI_F / blades);
}

// MARK: - Bokeh depth of field

struct FFBokehParameters {
    float4 depth;     // near, far, reversed
    float4 lens;      // pixel radius scale K, focus distance, maximum radius, sample count
    float4 aperture;  // blades, rotation, dither, tile size
    float4 tiles;     // dilation in tiles
};

static inline float ff_signed_coc(float distance, constant FFBokehParameters& p) {
    return p.lens.x * (1.0f - p.lens.y / distance);
}

kernel void frostfield_bokeh_coc(
    depth2d<float, access::read> depth [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    constant FFBokehParameters& p [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= destination.get_width() || pixel.y >= destination.get_height()) { return; }
    destination.write(float4(ff_signed_coc(ff_distance(depth.read(pixel), p.depth), p)), pixel);
}

kernel void frostfield_bokeh_tiles(
    depth2d<float, access::read> depth [[texture(0)]],
    texture2d<float, access::write> tiles [[texture(1)]],
    constant FFBokehParameters& p [[buffer(0)]],
    uint2 tile [[thread_position_in_grid]]
) {
    if (tile.x >= tiles.get_width() || tile.y >= tiles.get_height()) { return; }
    uint size = uint(p.aperture.w);
    uint2 origin = tile * size;
    uint2 end = min(origin + size, uint2(depth.get_width(), depth.get_height()));
    float largest = 0;
    for (uint y = origin.y; y < end.y; ++y) {
        for (uint x = origin.x; x < end.x; ++x) {
            float coc = abs(ff_signed_coc(ff_distance(depth.read(uint2(x, y)), p.depth), p));
            largest = max(largest, min(coc, p.lens.z));
        }
    }
    tiles.write(float4(largest), tile);
}

kernel void frostfield_bokeh_dilate(
    texture2d<float, access::read> tiles [[texture(0)]],
    texture2d<float, access::write> dilated [[texture(1)]],
    constant FFBokehParameters& p [[buffer(0)]],
    uint2 tile [[thread_position_in_grid]]
) {
    if (tile.x >= dilated.get_width() || tile.y >= dilated.get_height()) { return; }
    int reach = int(p.tiles.x);
    int2 last = int2(tiles.get_width(), tiles.get_height()) - 1;
    float largest = 0;
    for (int y = -reach; y <= reach; ++y) {
        for (int x = -reach; x <= reach; ++x) {
            largest = max(largest, tiles.read(uint2(clamp(int2(tile) + int2(x, y), int2(0), last))).r);
        }
    }
    dilated.write(float4(largest), tile);
}

fragment float4 frostfield_bokeh_gather(
    float4 position [[position]],
    texture2d<float, access::read> color [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture2d<float, access::read> tiles [[texture(2)]],
    constant FFBokehParameters& p [[buffer(0)]]
) {
    int2 pixel = int2(position.xy);
    float4 center = color.read(uint2(pixel));
    float search = tiles.read(uint2(pixel) / uint(p.aperture.w)).r;
    if (search < 0.5f) { return center; }

    int2 size = int2(color.get_width(), color.get_height());
    float centerDistance = ff_distance(depth.read(uint2(pixel)), p.depth);
    float centerRadius = max(min(abs(ff_signed_coc(centerDistance, p)), p.lens.z), 0.5f);
    int count = int(p.lens.w);
    // Each tap stands for an equal share of the searched disk.
    float tapArea = M_PI_F * search * search / float(count);
    float centerWeight = 1.0f / (M_PI_F * centerRadius * centerRadius);
    float4 background = center * centerWeight;
    float backgroundWeight = centerWeight;
    float4 foreground = 0;
    float foregroundWeight = 0;
    float rotation = 0;
    if (p.aperture.z > 0.5f) {
        rotation = 2.0f * M_PI_F * fract(52.9829189f * fract(dot(position.xy, float2(0.06711056f, 0.00583715f))));
    }
    for (int i = 0; i < count; ++i) {
        float r = search * sqrt((float(i) + 0.5f) / float(count));
        float theta = float(i) * 2.39996323f + rotation;
        int2 tap = pixel + int2(round(r * float2(cos(theta), sin(theta))));
        if (any(tap < 0) || any(tap >= size) || all(tap == pixel)) { continue; }
        float2 offset = float2(tap - pixel);
        float reach = length(offset);
        float tapDistance = ff_distance(depth.read(uint2(tap)), p.depth);
        float tapRadius = min(abs(ff_signed_coc(tapDistance, p)), p.lens.z);
        bool nearer = tapDistance < centerDistance * 0.99f;
        if (tapDistance > centerDistance) {
            // Whatever lies behind the center is seen only through the center's own blur:
            // a blurred background cannot spread over a sharper subject in front of it,
            // and a defocused foreground turns translucent toward its silhouette.
            tapRadius = centerRadius;
        }
        float shape = ff_polygon(atan2(-offset.y, -offset.x), p.aperture.x, p.aperture.y);
        float coverage = saturate(tapRadius * shape - reach + 0.5f);
        if (coverage <= 0) { continue; }
        float spread = max(tapRadius, 0.5f);
        float weight = coverage * tapArea / (M_PI_F * spread * spread);
        float4 tapColor = color.read(uint2(tap));
        if (nearer) {
            foreground += tapColor * weight;
            foregroundWeight += weight;
        } else {
            background += tapColor * weight;
            backgroundWeight += weight;
        }
    }
    float4 result = background / backgroundWeight;
    if (foregroundWeight > 0) {
        result = mix(result, foreground / foregroundWeight, saturate(foregroundWeight));
    }
    return ff_optics_finite(result);
}

// MARK: - God rays

struct FFGodRayParameters {
    float4 light;   // light uv, density, decay
    float4 source;  // 0 sun or 1 sky, sun radius or threshold, aspect, reversed
    float4 color;   // ray color, intensity
    float4 pass;    // samples, step scale
};

kernel void frostfield_godrays_mask(
    texture2d<float> scene [[texture(0)]],
    depth2d<float> depth [[texture(1)]],
    texture2d<float, access::write> mask [[texture(2)]],
    constant FFGodRayParameters& p [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= mask.get_width() || pixel.y >= mask.get_height()) { return; }
    float2 uv = (float2(pixel) + 0.5f) / float2(mask.get_width(), mask.get_height());
    float stored = depth.sample(ff_optics_nearest, uv);
    bool sky = p.source.w > 0.5f ? stored <= 0.0f : stored >= 1.0f;
    float3 value = 0;
    if (sky) {
        if (p.source.x < 0.5f) {
            float radius = p.source.y;
            float d = length((uv - p.light.xy) * float2(p.source.z, 1.0f));
            value = float3(1.0f - smoothstep(0.5f * radius, radius, d));
        } else {
            value = max(scene.sample(ff_optics_linear, uv, level(0)).rgb - p.source.y, float3(0));
        }
    }
    mask.write(float4(value, 0), pixel);
}

kernel void frostfield_godrays_blur(
    texture2d<float> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant FFGodRayParameters& p [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= output.get_width() || pixel.y >= output.get_height()) { return; }
    float2 uv = (float2(pixel) + 0.5f) / float2(output.get_width(), output.get_height());
    float2 step = (p.light.xy - uv) * p.light.z * p.pass.y;
    int count = int(p.pass.x);
    float3 sum = 0;
    float total = 0;
    float weight = 1;
    for (int i = 0; i < count; ++i) {
        sum += input.sample(ff_optics_linear, uv + step * float(i), level(0)).rgb * weight;
        total += weight;
        weight *= p.light.w;
    }
    output.write(float4(sum / total, 0), pixel);
}

fragment float4 frostfield_godrays_composite(
    float4 position [[position]],
    texture2d<float, access::read> scene [[texture(0)]],
    texture2d<float> rays [[texture(1)]],
    constant FFGodRayParameters& p [[buffer(0)]]
) {
    float4 color = scene.read(uint2(position.xy));
    float2 uv = position.xy / float2(scene.get_width(), scene.get_height());
    float3 shafts = rays.sample(ff_optics_linear, uv, level(0)).rgb;
    return ff_optics_finite(float4(color.rgb + shafts * p.color.rgb * p.color.w, color.a));
}

// MARK: - Lens flare

struct FFFlareLight {
    float4 position;  // uv, stored depth or -1 for the sky, intensity
    float4 color;
};

struct FFFlareParameters {
    float4 glare;      // radius, intensity, aspect, light count
    float4 starburst;  // rays, intensity, length, sharpness
    float4 halo;       // radius, width, intensity, dispersion
    float4 shape;      // starburst rotation, ghost blades, ghost rotation, ghost count
};

struct FFFlareGhost {
    float4 geometry;  // distance, radius, softness
    float4 color;
};

kernel void frostfield_flare_visibility(
    depth2d<float, access::read> depth [[texture(0)]],
    constant FFFlareLight* lights [[buffer(0)]],
    constant float4& occlusion [[buffer(1)]], // radius in pixels, depth bias, reversed, samples
    device float* visibility [[buffer(2)]],
    constant uint& lightCount [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= lightCount) { return; }
    FFFlareLight light = lights[index];
    int2 size = int2(depth.get_width(), depth.get_height());
    float2 center = light.position.xy * float2(size);
    int count = int(occlusion.w);
    bool reversed = occlusion.z > 0.5f;
    float visible = 0;
    for (int i = 0; i < count; ++i) {
        float r = occlusion.x * sqrt((float(i) + 0.5f) / float(count));
        float theta = float(i) * 2.39996323f;
        int2 tap = int2(floor(center + r * float2(cos(theta), sin(theta))));
        if (any(tap < 0) || any(tap >= size)) { continue; }
        float stored = depth.read(uint2(tap));
        bool clear;
        if (light.position.z < 0) {
            clear = reversed ? stored <= 0.0f : stored >= 1.0f;
        } else {
            clear = reversed ? stored <= light.position.z + occlusion.y : stored >= light.position.z - occlusion.y;
        }
        visible += clear ? 1.0f : 0.0f;
    }
    visibility[index] = visible / float(count);
}

fragment float4 frostfield_flare_composite(
    float4 position [[position]],
    texture2d<float, access::read> scene [[texture(0)]],
    constant FFFlareLight* lights [[buffer(0)]],
    constant FFFlareParameters& p [[buffer(1)]],
    const device float* visibility [[buffer(2)]],
    constant FFFlareGhost* ghosts [[buffer(3)]]
) {
    float4 color = scene.read(uint2(position.xy));
    float2 size = float2(scene.get_width(), scene.get_height());
    float aspect = p.glare.z;
    float2 q = (position.xy / size - 0.5f) * float2(aspect, 1.0f);
    float3 added = 0;
    int lightCount = int(p.glare.w);
    int ghostCount = int(p.shape.w);
    for (int i = 0; i < lightCount; ++i) {
        float strength = visibility[i] * lights[i].position.w;
        if (strength <= 0) { continue; }
        float2 l = (lights[i].position.xy - 0.5f) * float2(aspect, 1.0f);
        float2 d = q - l;
        float r = length(d);
        float3 light = 0;
        if (p.glare.x > 0) {
            light += float3(p.glare.y * exp(-0.5f * r * r / (p.glare.x * p.glare.x)));
        }
        if (p.starburst.x > 0 && p.starburst.y > 0) {
            float angle = atan2(d.y, d.x);
            float streak = pow(abs(cos(angle * p.starburst.x * 0.5f + p.shape.x)), p.starburst.w);
            light += float3(p.starburst.y * streak * exp(-r / max(p.starburst.z, 1e-4f)));
        }
        if (p.halo.z > 0) {
            float3 radii = p.halo.x * (1.0f + p.halo.w * float3(-1, 0, 1));
            float3 ring = (r - radii) / max(p.halo.y, 1e-4f);
            light += p.halo.z * exp(-ring * ring);
        }
        for (int g = 0; g < ghostCount; ++g) {
            FFFlareGhost ghost = ghosts[g];
            float2 local = q - l * (1.0f - 2.0f * ghost.geometry.x);
            float shape = ff_polygon(atan2(local.y, local.x), p.shape.y, p.shape.z);
            float extent = length(local) / shape;
            float radius = ghost.geometry.y;
            float soft = max(ghost.geometry.z, 1e-3f);
            light += ghost.color.rgb * (1.0f - smoothstep(radius * (1.0f - soft), radius, extent));
        }
        added += light * lights[i].color.rgb * strength;
    }
    return ff_optics_finite(float4(color.rgb + added, color.a));
}

struct FFFlareSprite {
    float4 placement; // frame UV, size in pixels
    float4 tint;
};

fragment float4 frostfield_flare_sprites(
    float4 position [[position]],
    texture2d<float, access::read> scene [[texture(0)]],
    array<texture2d<float>, 16> sprites [[texture(1)]],
    constant FFFlareSprite* elements [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    const device float& visibility [[buffer(2)]]
) {
    float4 color = scene.read(uint2(position.xy));
    if (visibility <= 0.0f) { return color; }
    constexpr sampler spriteFilter(coord::normalized, address::clamp_to_edge, filter::linear, mip_filter::linear);
    float2 size = float2(scene.get_width(), scene.get_height());
    float3 added = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float4 placement = elements[i].placement;
        float2 uv = (position.xy - placement.xy * size) / placement.z + 0.5f;
        if (!all(uv >= 0.0f) || !all(uv < 1.0f)) { continue; }
        float width = float(max(sprites[i].get_width(), sprites[i].get_height()));
        float mip = max(log2(width) - log2(placement.z), 0.0f);
        float4 texel = ff_optics_finite(sprites[i].sample(spriteFilter, uv, level(mip)));
        added += texel.rgb * texel.a * elements[i].tint.rgb;
    }
    if (!any(added > 0.0f)) { return color; }
    return ff_optics_finite(float4(color.rgb + visibility * added, color.a));
}

// MARK: - Selection outline

struct FFOutlineParameters {
    float4 visibleColor;
    float4 hiddenColor;
    float4 hiddenFill;
    float thickness;
    float depthBias;
    uint reversed;
    int radius;
};

// 0 outside the selection, 1 selected and visible, 2 selected behind other geometry.
static inline int ff_selection_state(float selection, float scene, constant FFOutlineParameters& p) {
    bool reversed = p.reversed != 0;
    bool covered = reversed ? selection > 0.0f : selection < 1.0f;
    if (!covered) { return 0; }
    bool visible = reversed ? selection >= scene - p.depthBias : selection <= scene + p.depthBias;
    return visible ? 1 : 2;
}

kernel void frostfield_outline_rows(
    depth2d<float, access::read> selection [[texture(0)]],
    depth2d<float, access::read> scene [[texture(1)]],
    texture2d<float, access::write> rows [[texture(2)]],
    constant FFOutlineParameters& p [[buffer(0)]],
    uint2 pixel [[thread_position_in_grid]]
) {
    if (pixel.x >= rows.get_width() || pixel.y >= rows.get_height()) { return; }
    int width = int(rows.get_width());
    float none = float(p.radius + 1);
    float nearestVisible = none;
    float nearestHidden = none;
    int own = 0;
    for (int dx = -p.radius; dx <= p.radius; ++dx) {
        int x = int(pixel.x) + dx;
        if (x < 0 || x >= width) { continue; }
        uint2 tap = uint2(x, pixel.y);
        int state = ff_selection_state(selection.read(tap), scene.read(tap), p);
        if (dx == 0) { own = state; }
        if (state == 1) {
            nearestVisible = min(nearestVisible, float(abs(dx)));
        } else if (state == 2) {
            nearestHidden = min(nearestHidden, float(abs(dx)));
        }
    }
    rows.write(float4(nearestVisible, nearestHidden, float(own), 0), pixel);
}

static inline float4 ff_outline_edge(
    texture2d<float, access::read> rows, int2 pixel, constant FFOutlineParameters& p
) {
    float none = float(p.radius + 1);
    float visible = 4.0f * none * none;
    float hidden = visible;
    int height = int(rows.get_height());
    for (int dy = -p.radius; dy <= p.radius; ++dy) {
        int y = pixel.y + dy;
        if (y < 0 || y >= height) { continue; }
        float4 row = rows.read(uint2(pixel.x, y));
        float vertical = float(dy * dy);
        if (row.x < none) { visible = min(visible, row.x * row.x + vertical); }
        if (row.y < none) { hidden = min(hidden, row.y * row.y + vertical); }
    }
    // A pixel whose center is d pixels from the nearest selected pixel's center spans
    // d − 1 to d pixels from that pixel's edge, so it holds saturate(thickness + 1 − d)
    // of a band `thickness` pixels wide around the silhouette.
    float visibleCoverage = saturate(p.thickness + 1.0f - sqrt(visible));
    float hiddenCoverage = saturate(p.thickness + 1.0f - sqrt(hidden)) * (1.0f - visibleCoverage);
    float visibleAlpha = visibleCoverage * p.visibleColor.a;
    float hiddenAlpha = hiddenCoverage * p.hiddenColor.a;
    return float4(p.visibleColor.rgb * visibleAlpha + p.hiddenColor.rgb * hiddenAlpha, visibleAlpha + hiddenAlpha);
}

fragment float4 frostfield_outline_composite(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::read> rows [[texture(1)]],
    constant FFOutlineParameters& p [[buffer(0)]]
) {
    int2 pixel = int2(position.xy);
    float4 color = source.read(uint2(pixel));
    float own = rows.read(uint2(pixel)).z;
    float4 overlay = 0;
    if (own > 1.5f) {
        overlay = float4(p.hiddenFill.rgb * p.hiddenFill.a, p.hiddenFill.a);
    } else if (own < 0.5f) {
        overlay = ff_outline_edge(rows, pixel, p);
    }
    return ff_optics_finite(overlay + color * (1.0f - overlay.a));
}

// Write only the crisp outline, so neither scene color nor a hidden fill enters the blur.
fragment float4 frostfield_outline_edges(
    float4 position [[position]],
    texture2d<float, access::read> rows [[texture(0)]],
    constant FFOutlineParameters& p [[buffer(0)]]
) {
    int2 pixel = int2(position.xy);
    return rows.read(uint2(pixel)).z < 0.5f ? ff_optics_finite(ff_outline_edge(rows, pixel, p)) : float4(0);
}

fragment float4 frostfield_outline_styled_composite(
    float4 position [[position]],
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::read> rows [[texture(1)]],
    texture2d<float, access::read> edges [[texture(2)]],
    texture2d<float, access::read> glow [[texture(3)]],
    texture2d<float> pattern [[texture(4)]],
    constant FFOutlineParameters& p [[buffer(0)]],
    constant float4& style [[buffer(1)]]
) {
    uint2 pixel = uint2(position.xy);
    float4 color = source.read(pixel);
    float own = rows.read(pixel).z;
    if (own < 0.5f) {
        float4 edge = style.w > 0.0f ? edges.read(pixel) : ff_outline_edge(rows, int2(pixel), p);
        color = edge + color * (1.0f - edge.a);
        if (style.w > 0.0f) {
            float4 halo = glow.read(pixel) * style.w;
            color.rgb += halo.rgb;
            color.a += (1.0f - color.a) * saturate(halo.a);
        }
    } else {
        bool hidden = own > 1.5f;
        if (hidden) {
            color = float4(p.hiddenFill.rgb * p.hiddenFill.a, p.hiddenFill.a) + color * (1.0f - p.hiddenFill.a);
        }
        if (style.z > 0.0f) {
            constexpr sampler repeatLinear(coord::normalized, address::repeat, filter::linear);
            float2 uv = position.xy / float2(rows.get_width(), rows.get_height());
            float mask = 1.0f - pattern.sample(repeatLinear, uv * style.xy, level(0)).r;
            float coverage = saturate(mask) * style.z * (hidden ? 0.5f : 1.0f);
            color = float4(coverage) + color * (1.0f - coverage);
        }
    }
    return ff_optics_finite(color);
}
