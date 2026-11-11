#include <metal_stdlib>
using namespace metal;

// Physically based sky: Bruneton's atmosphere with Hillaire's multiple-scattering
// approximation, evaluated at 16 wavelengths in four float4 batches. Every
// radiance below is per unit top-of-atmosphere solar irradiance until it is
// multiplied by `solar` and converted with the linear sRGB weights.
// Mirrors SkyModel.swift and SkyEnvironment.swift.

constant float skyPi = 3.14159265358979323846f;
constant uint skyBatches = 4;
constant uint skyFlagMultipleScattering = 1u;
constant uint skyFlagGround = 2u;

struct FrostfieldSkyConstants {
    float4 radii;    // ground radius, top radius, origin altitude (at least 1 m), unused
    float4 profile;  // 1 / Rayleigh scale height, 1 / aerosol scale height, aerosol g, ground albedo
    float4 sun;      // unit direction toward the Sun, flags
    float4 rayleigh[4];
    float4 mieScattering[4];
    float4 mieExtinction[4];
    float4 ozone[4];
    float4 solar[4];
    float4 red[4];
    float4 green[4];
    float4 blue[4];
};

constant uint skyLegendreCount = 12;
constant float skyLegendreNodes[skyLegendreCount] = {
    0.98156063424671924f, 0.9041172563704748f, 0.76990267419430469f, 0.58731795428661748f,
    0.36783149899818018f, 0.12523340851146894f, -0.12523340851146891f, -0.36783149899818018f,
    -0.58731795428661748f, -0.76990267419430469f, -0.9041172563704748f, -0.98156063424671924f,
};
constant float skyLegendreWeights[skyLegendreCount] = {
    0.047175336386511835f, 0.10693932599531818f, 0.16007832854334633f, 0.20316742672306584f,
    0.23349253653835478f, 0.24914704581340288f, 0.24914704581340288f, 0.23349253653835478f,
    0.20316742672306584f, 0.16007832854334633f, 0.10693932599531818f, 0.047175336386511835f,
};

struct SkySteps {
    uint transmittance;
    uint multipleScatteringDirections; // azimuths per cosine node
    uint multipleScatteringSteps;
    uint view;
};

static float skyCoord(float unit, float size) {
    return 0.5f / size + saturate(unit) * (1.0f - 1.0f / size);
}

static float4 skyBilinear(texture2d<float, access::read> source, float2 uv) {
    int2 size = int2(source.get_width(), source.get_height());
    float2 location = uv * float2(size) - 0.5f;
    int2 corner = int2(floor(location));
    float2 f = location - floor(location);
    int2 a = clamp(corner, int2(0), size - 1);
    int2 b = clamp(corner + 1, int2(0), size - 1);
    float4 top = mix(source.read(uint2(a.x, a.y)), source.read(uint2(b.x, a.y)), f.x);
    float4 bottom = mix(source.read(uint2(a.x, b.y)), source.read(uint2(b.x, b.y)), f.x);
    return mix(top, bottom, f.y);
}

static float4 skyBilinear(texture2d_array<float, access::read> source, float2 uv, uint slice) {
    int2 size = int2(source.get_width(), source.get_height());
    float2 location = uv * float2(size) - 0.5f;
    int2 corner = int2(floor(location));
    float2 f = location - floor(location);
    int2 a = clamp(corner, int2(0), size - 1);
    int2 b = clamp(corner + 1, int2(0), size - 1);
    float4 top = mix(source.read(uint2(a.x, a.y), slice), source.read(uint2(b.x, a.y), slice), f.x);
    float4 bottom = mix(source.read(uint2(a.x, b.y), slice), source.read(uint2(b.x, b.y), slice), f.x);
    return mix(top, bottom, f.y);
}

// Distances use the altitude h = r − ground, so r² − ground² = h (2 ground + h)
// keeps its precision near the surface in single precision.
static float skyDistanceToTop(float ground, float top, float h, float mu) {
    float r = ground + h;
    float discriminant = r * r * mu * mu + (top - r) * (top + r);
    return max(-r * mu + sqrt(max(discriminant, 0.0f)), 0.0f);
}

/// The distance to the ground, or a negative value when the ray misses it.
static float skyDistanceToGround(float ground, float h, float mu) {
    float r = ground + h;
    float discriminant = r * r * mu * mu - h * (2.0f * ground + h);
    if (mu >= 0.0f || discriminant < 0.0f) {
        return -1.0f;
    }
    return max(-r * mu - sqrt(discriminant), 0.0f);
}

static float skyAltitudeAlong(float ground, float h, float mu, float t) {
    float r = ground + h;
    float squares = max(t * t + 2.0f * r * mu * t + h * (2.0f * ground + h), 0.0f);
    return squares / (sqrt(squares + ground * ground) + ground);
}

static float3 skyDensity(constant FrostfieldSkyConstants& c, float h) {
    h = max(h, 0.0f);
    return float3(exp(-h * c.profile.x), exp(-h * c.profile.y), max(0.0f, 1.0f - abs(h - 25000.0f) / 15000.0f));
}

static float2 skyTransmittanceUV(constant FrostfieldSkyConstants& c, float h, float mu, float2 size) {
    float ground = c.radii.x;
    float top = c.radii.y;
    float horizon = sqrt((top - ground) * (top + ground));
    h = max(h, 0.0f);
    float rho = sqrt(h * (2.0f * ground + h));
    float d = skyDistanceToTop(ground, top, h, mu);
    float dMin = top - (ground + h);
    float dMax = rho + horizon;
    return float2(skyCoord((d - dMin) / max(dMax - dMin, 1e-3f), size.x), skyCoord(rho / horizon, size.y));
}

/// Transmittance from altitude h along a ray with cosine mu to the top of the
/// atmosphere, or zero when the ray meets the ground.
static float4 skyTransmittanceToTop(
    constant FrostfieldSkyConstants& c, texture2d<float, access::read> lut, float h, float mu, uint batch
) {
    if (skyDistanceToGround(c.radii.x, max(h, 0.0f), mu) >= 0.0f) {
        return float4(0);
    }
    float2 size = float2(lut.get_width(), lut.get_height());
    float3 lengths = skyBilinear(lut, skyTransmittanceUV(c, h, mu, size)).xyz;
    return exp(-(c.rayleigh[batch] * lengths.x + c.mieExtinction[batch] * lengths.y + c.ozone[batch] * lengths.z));
}

static float4 skyMultipleScattering(
    constant FrostfieldSkyConstants& c, texture2d_array<float, access::read> lut, float h, float muSun, uint batch
) {
    float2 size = float2(lut.get_width(), lut.get_height());
    float2 uv = float2(
        skyCoord(0.5f * (muSun + 1.0f), size.x),
        skyCoord(h / (c.radii.y - c.radii.x), size.y)
    );
    return skyBilinear(lut, uv, batch);
}

/// (1 − e^(−σ dt)) / σ, with a series where the exponential loses precision.
static float4 skySegment(float4 sigma, float dt) {
    float4 x = sigma * dt;
    float4 series = dt * (1.0f - x * (0.5f - x * (1.0f / 6.0f)));
    float4 exact = (1.0f - exp(-x)) / max(sigma, float4(1e-30f));
    return select(exact, series, x < 1e-3f);
}

static float skyRayleighPhase(float nu) {
    return 3.0f / (16.0f * skyPi) * (1.0f + nu * nu);
}

/// Henyey–Greenstein phase per steradian. Writing 1 + g² − 2gν as
/// (1 − g)² + 2g(1 − ν) keeps precision for strongly forward aerosols near the Sun.
static float skyHenyeyGreenstein(float nu, float g) {
    float oneMinusG = 1.0f - g;
    float denominator = oneMinusG * oneMinusG + 2.0f * g * (1.0f - nu);
    return (1.0f - g * g) / (4.0f * skyPi * denominator * sqrt(denominator));
}

/// One minus the cosine of the angle to the Sun whose Henyey–Greenstein
/// cumulative probability, counted from directly away from the Sun, is ξ.
static float skyHenyeyGreensteinInverse(float xi, float g) {
    if (g < 1e-3f) {
        return 2.0f * (1.0f - xi);
    }
    float oneMinusG = 1.0f - g;
    float x = (1.0f - g * g) / (oneMinusG + 2.0f * g * xi);
    return clamp((x - oneMinusG) * (x + oneMinusG) / (2.0f * g), 0.0f, 2.0f);
}

struct SkyRay {
    float h;      // altitude of the origin in meters
    float mu;     // cosine between the ray and the local vertical
    float muSun;  // cosine between the Sun and the local vertical
    float nu;     // cosine between the ray and the Sun
};

/// Integrates in-scattered radiance and transmittance over [t0, t1] with uniform
/// midpoint steps, using the exact exponential integral within each step.
static void skyIntegrate(
    constant FrostfieldSkyConstants& c, texture2d<float, access::read> transmittance,
    texture2d_array<float, access::read> multiple, SkyRay ray, float t0, float t1, uint steps,
    uint batch, thread float4& radiance, thread float4& throughput
) {
    float ground = c.radii.x;
    uint flags = as_type<uint>(c.sun.w);
    float rayleighPhase = skyRayleighPhase(ray.nu);
    float miePhase = skyHenyeyGreenstein(ray.nu, c.profile.z);
    float r0 = ground + ray.h;
    float dt = (t1 - t0) / float(steps);
    for (uint i = 0; i < steps; ++i) {
        float t = t0 + (float(i) + 0.5f) * dt;
        float h = skyAltitudeAlong(ground, ray.h, ray.mu, t);
        float r = ground + h;
        float muSun = clamp((r0 * ray.muSun + t * ray.nu) / r, -1.0f, 1.0f);
        float3 density = skyDensity(c, h);
        float4 rayleigh = c.rayleigh[batch] * density.x;
        float4 mie = c.mieScattering[batch] * density.y;
        float4 extinction = rayleigh + c.mieExtinction[batch] * density.y + c.ozone[batch] * density.z;
        float4 sun = skyTransmittanceToTop(c, transmittance, h, muSun, batch);
        float4 source = (rayleigh * rayleighPhase + mie * miePhase) * sun;
        if ((flags & skyFlagMultipleScattering) != 0u) {
            source += (rayleigh + mie) * skyMultipleScattering(c, multiple, h, muSun, batch);
        }
        radiance += throughput * source * skySegment(extinction, dt);
        throughput *= exp(-extinction * dt);
    }
}

/// Radiance of a view ray that leaves the origin, per unit solar irradiance, with
/// steps on a quadratic partition that refines near the dense start. A ray that
/// meets the ground adds the Lambertian ground lit by the Sun and by the sky's
/// horizontal irradiance `skyIrradiance`.
static float4 skyView(
    constant FrostfieldSkyConstants& c, texture2d<float, access::read> transmittance,
    texture2d_array<float, access::read> multiple, SkyRay ray, uint steps, uint batch, float4 skyIrradiance
) {
    float ground = c.radii.x;
    float tGround = skyDistanceToGround(ground, ray.h, ray.mu);
    bool hitsGround = tGround >= 0.0f;
    float tEnd = hitsGround ? tGround : skyDistanceToTop(ground, c.radii.y, ray.h, ray.mu);
    float4 radiance = float4(0);
    float4 throughput = float4(1);
    for (uint i = 0; i < steps; ++i) {
        float s0 = float(i) / float(steps);
        float s1 = float(i + 1) / float(steps);
        skyIntegrate(c, transmittance, multiple, ray, tEnd * s0 * s0, tEnd * s1 * s1, 1, batch, radiance, throughput);
    }
    uint flags = as_type<uint>(c.sun.w);
    if (hitsGround && (flags & skyFlagGround) != 0u) {
        float muSun = clamp(((ground + ray.h) * ray.muSun + tEnd * ray.nu) / ground, -1.0f, 1.0f);
        float4 sun = muSun > 0.0f ? skyTransmittanceToTop(c, transmittance, 0.0f, muSun, batch) * muSun : float4(0);
        radiance += throughput * (c.profile.w / skyPi) * (sun + skyIrradiance);
    }
    return radiance;
}

static float3 skyLinearSRGB(constant FrostfieldSkyConstants& c, thread const float4* spectrum) {
    float3 rgb = float3(0);
    for (uint b = 0; b < skyBatches; ++b) {
        float4 value = spectrum[b] * c.solar[b];
        rgb += float3(dot(value, c.red[b]), dot(value, c.green[b]), dot(value, c.blue[b]));
    }
    return rgb;
}

// MARK: - Look-up tables

/// Density-weighted path lengths (Rayleigh, aerosol, ozone) to the top of the
/// atmosphere, in Bruneton's (r, μ) parameterization that refines the horizon.
kernel void frostfield_sky_transmittance(
    texture2d<float, access::write> output [[texture(0)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant SkySteps& steps [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 size = uint2(output.get_width(), output.get_height());
    if (any(gid >= size)) {
        return;
    }
    float ground = c.radii.x;
    float top = c.radii.y;
    float horizon = sqrt((top - ground) * (top + ground));
    float xMu = float(gid.x) / float(size.x - 1);
    float xR = float(gid.y) / float(size.y - 1);
    float rho = horizon * xR;
    float r = sqrt(rho * rho + ground * ground);
    float h = rho * rho / (r + ground);
    float dMin = top - r;
    float dMax = rho + horizon;
    float d = dMin + xMu * (dMax - dMin);
    float mu = d <= 0.0f ? 1.0f : clamp((horizon * horizon - rho * rho - d * d) / (2.0f * r * d), -1.0f, 1.0f);

    float3 lengths = float3(0);
    uint count = steps.transmittance;
    for (uint i = 0; i < count; ++i) {
        float s0 = float(i) / float(count);
        float s1 = float(i + 1) / float(count);
        float t = d * 0.25f * (s0 + s1) * (s0 + s1);
        float dt = d * (s1 * s1 - s0 * s0);
        lengths += skyDensity(c, skyAltitudeAlong(ground, h, mu, t)) * dt;
    }
    output.write(float4(lengths, 0), gid);
}

/// Hillaire's Ψ_ms: the isotropic second-order radiance at an altitude and Sun
/// angle, multiplied by 1 / (1 − f_ms) to sum every higher order.
kernel void frostfield_sky_multiple_scattering(
    texture2d<float, access::read> transmittance [[texture(0)]],
    texture2d_array<float, access::write> output [[texture(1)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant SkySteps& steps [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint2 size = uint2(output.get_width(), output.get_height());
    if (any(gid.xy >= size) || gid.z >= skyBatches) {
        return;
    }
    uint batch = gid.z;
    float ground = c.radii.x;
    float top = c.radii.y;
    float muSun = -1.0f + 2.0f * float(gid.x) / float(size.x - 1);
    float h = clamp((top - ground) * float(gid.y) / float(size.y - 1), 1.0f, top - ground - 1.0f);
    float r0 = ground + h;
    float sinSun = sqrt(max(1.0f - muSun * muSun, 0.0f));
    float isotropic = 1.0f / (4.0f * skyPi);
    float albedo = c.profile.w;

    uint axis = steps.multipleScatteringDirections;
    uint count = steps.multipleScatteringSteps;
    float4 secondOrder = float4(0);
    float4 transfer = float4(0);
    // Gauss–Legendre nodes in cos θ, separately below and above the horizon where
    // the ground cuts rays short, times `axis` uniform azimuths. Each node carries
    // its share of the sphere's measure dμ / 2.
    float horizon = -sqrt(h * (2.0f * ground + h)) / r0;
    for (uint j = 0; j < 2 * skyLegendreCount; ++j) {
        bool lower = j < skyLegendreCount;
        float low = lower ? -1.0f : horizon;
        float high = lower ? horizon : 1.0f;
        uint node = j % skyLegendreCount;
        float mu = 0.5f * (high - low) * skyLegendreNodes[node] + 0.5f * (high + low);
        float rowWeight = 0.25f * (high - low) * skyLegendreWeights[node];
        float sine = sqrt(max(1.0f - mu * mu, 0.0f));
        for (uint i = 0; i < axis; ++i) {
            float phi = 2.0f * skyPi * (float(i) + 0.5f) / float(axis);
            float nu = mu * muSun + sine * cos(phi) * sinSun;
            float tGround = skyDistanceToGround(ground, h, mu);
            bool hitsGround = tGround >= 0.0f;
            float tEnd = hitsGround ? tGround : skyDistanceToTop(ground, top, h, mu);
            float4 radiance = float4(0);
            float4 scattered = float4(0);
            float4 throughput = float4(1);
            for (uint k = 0; k < count; ++k) {
                // Midpoints of a quadratic partition refine the dense start of the ray.
                float s0 = float(k) / float(count);
                float s1 = float(k + 1) / float(count);
                float t = tEnd * 0.25f * (s0 + s1) * (s0 + s1);
                float dt = tEnd * (s1 * s1 - s0 * s0);
                float hs = skyAltitudeAlong(ground, h, mu, t);
                float rs = ground + hs;
                float muSample = clamp((r0 * muSun + t * nu) / rs, -1.0f, 1.0f);
                float3 density = skyDensity(c, hs);
                float4 scattering = c.rayleigh[batch] * density.x + c.mieScattering[batch] * density.y;
                float4 extinction = c.rayleigh[batch] * density.x + c.mieExtinction[batch] * density.y
                    + c.ozone[batch] * density.z;
                float4 sun = skyTransmittanceToTop(c, transmittance, hs, muSample, batch);
                float4 segment = skySegment(extinction, dt);
                radiance += throughput * scattering * sun * isotropic * segment;
                scattered += throughput * scattering * segment;
                throughput *= exp(-extinction * dt);
            }
            if (hitsGround) {
                float muGround = clamp((r0 * muSun + tEnd * nu) / ground, -1.0f, 1.0f);
                if (muGround > 0.0f) {
                    float4 sun = skyTransmittanceToTop(c, transmittance, 0.0f, muGround, batch);
                    radiance += throughput * sun * muGround * albedo / skyPi;
                }
            }
            secondOrder += radiance * rowWeight;
            transfer += scattered * rowWeight;
        }
    }
    // Both sphere integrals carry the isotropic phase 1/4π, which turns them into averages.
    float4 secondOrderAverage = secondOrder / float(axis);
    float4 transferAverage = transfer / float(axis);
    float4 psi = secondOrderAverage / max(1.0f - transferAverage, float4(1e-4f));
    output.write(psi, gid.xy, batch);
}

// MARK: - Sky

/// Horizontal spectral irradiance from the sky at the origin, per unit solar
/// irradiance. A strongly forward aerosol phase concentrates sky light in a glow a
/// fraction of a degree around the Sun that cosine-weighted directions alone miss,
/// so the integral combines two stratified 32 × 32 grids by multiple importance
/// sampling with the balance heuristic: cosine-weighted directions about the zenith
/// and directions distributed like the aerosol phase function about the Sun.
kernel void frostfield_sky_irradiance(
    texture2d<float, access::read> transmittance [[texture(0)]],
    texture2d_array<float, access::read> multiple [[texture(1)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant SkySteps& steps [[buffer(1)]],
    device float4* result [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float4 partial[256][4];
    constexpr uint axis = 32;
    constexpr uint count = axis * axis;
    float4 sums[4] = {float4(0), float4(0), float4(0), float4(0)};
    float3 sun = c.sun.xyz;
    float g = c.profile.z;
    float3 helper = abs(sun.y) < 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, sun));
    float3 bitangent = cross(sun, tangent);
    for (uint index = thread_index; index < 2 * count; index += threads) {
        uint cell = index % count;
        float u = (float(cell % axis) + 0.5f) / float(axis);
        float v = (float(cell / axis) + 0.5f) / float(axis);
        float phi = 2.0f * skyPi * v;
        float3 direction;
        if (index < count) {
            float radius = sqrt(u);
            direction = float3(radius * cos(phi), sqrt(max(1.0f - u, 0.0f)), radius * sin(phi));
        } else {
            float oneMinusMu = skyHenyeyGreensteinInverse(u, g);
            float sine = sqrt(max(oneMinusMu * (2.0f - oneMinusMu), 0.0f));
            direction = normalize((1.0f - oneMinusMu) * sun + sine * (cos(phi) * tangent + sin(phi) * bitangent));
        }
        if (direction.y <= 0.0f) {
            continue;
        }
        float nu = dot(direction, sun);
        // f / (N p_cosine + N p_phase) with f = L cos θ and p_cosine = cos θ / π.
        float weight = direction.y / (float(count) * (direction.y / skyPi + skyHenyeyGreenstein(nu, g)));
        SkyRay ray = {c.radii.z, direction.y, sun.y, nu};
        for (uint b = 0; b < skyBatches; ++b) {
            sums[b] += skyView(c, transmittance, multiple, ray, steps.view, b, float4(0)) * weight;
        }
    }
    for (uint b = 0; b < skyBatches; ++b) {
        partial[thread_index][b] = sums[b];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < skyBatches) {
        float4 total = float4(0);
        for (uint i = 0; i < threads; ++i) {
            total += partial[i][thread_index];
        }
        result[thread_index] = total;
    }
}

/// Linear sRGB sky radiance for each texel of a 2:1 panorama whose top row faces
/// +Y, center column faces +X and increasing U turns toward +Z.
kernel void frostfield_sky_panorama(
    texture2d<float, access::read> transmittance [[texture(0)]],
    texture2d_array<float, access::read> multiple [[texture(1)]],
    texture2d<float, access::write> output [[texture(2)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant SkySteps& steps [[buffer(1)]],
    device const float4* skyIrradiance [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 size = uint2(output.get_width(), output.get_height());
    if (any(gid >= size)) {
        return;
    }
    float2 uv = (float2(gid) + 0.5f) / float2(size);
    float theta = uv.y * skyPi;
    float phi = (uv.x - 0.5f) * 2.0f * skyPi;
    float3 direction = float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
    float3 sun = c.sun.xyz;
    SkyRay ray = {c.radii.z, direction.y, sun.y, dot(direction, sun)};
    float4 spectrum[4];
    for (uint b = 0; b < skyBatches; ++b) {
        spectrum[b] = skyView(c, transmittance, multiple, ray, steps.view, b, skyIrradiance[b]);
    }
    float3 rgb = max(skyLinearSRGB(c, spectrum), 0.0f);
    output.write(float4(select(float3(0), rgb, isfinite(rgb)), 1), gid);
}

// MARK: - Aerial perspective

struct FrostfieldAerialUniforms {
    float4 right;      // world camera right, tan of half the horizontal field of view
    float4 up;         // world camera up, tan of half the vertical field of view
    float4 forward;    // world view direction, maximum distance in meters
    float4 planet;     // local vertical at the camera, camera altitude in meters
    float4 transmittanceRed[4];
    float4 transmittanceGreen[4];
    float4 transmittanceBlue[4];
    float4 sunDisk;    // unit direction toward the Sun, cosine of its angular radius
    float4 clip;       // 1 / far, 1 / near, reversed-depth flag, meters per world unit
    float4 options;    // draw sky, draw Sun, sky from the camera table (else the panorama), unused
    float4 diskCenter[4];
    float4 limbDarkening[4];
};

static float3 aerialDirection(constant FrostfieldAerialUniforms& u, float2 uv) {
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    return normalize(u.forward.xyz + ndc.x * u.right.w * u.right.xyz + ndc.y * u.up.w * u.up.xyz);
}

/// Fills froxels of in-scattered radiance and transmittance along each screen
/// column's ray, one slice per `maximumDistance × (k / (slices − 1))²`.
kernel void frostfield_sky_aerial_volume(
    texture2d<float, access::read> transmittance [[texture(0)]],
    texture2d_array<float, access::read> multiple [[texture(1)]],
    texture3d<half, access::write> inscattering [[texture(2)]],
    texture3d<half, access::write> extinction [[texture(3)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant FrostfieldAerialUniforms& u [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint3 size = uint3(inscattering.get_width(), inscattering.get_height(), inscattering.get_depth());
    if (any(gid >= size.xy)) {
        return;
    }
    float3 direction = aerialDirection(u, (float2(gid) + 0.5f) / float2(size.xy));
    float3 vertical = u.planet.xyz;
    SkyRay ray = {u.planet.w, dot(direction, vertical), dot(c.sun.xyz, vertical), dot(direction, c.sun.xyz)};
    float4 radiance[4] = {float4(0), float4(0), float4(0), float4(0)};
    float4 throughput[4] = {float4(1), float4(1), float4(1), float4(1)};
    float maximum = u.forward.w;
    inscattering.write(half4(0, 0, 0, 1), uint3(gid, 0));
    extinction.write(half4(1), uint3(gid, 0));
    for (uint k = 1; k < size.z; ++k) {
        float w0 = float(k - 1) / float(size.z - 1);
        float w1 = float(k) / float(size.z - 1);
        for (uint b = 0; b < skyBatches; ++b) {
            skyIntegrate(c, transmittance, multiple, ray, maximum * w0 * w0, maximum * w1 * w1, 2, b, radiance[b], throughput[b]);
        }
        float3 rgb = max(skyLinearSRGB(c, radiance), 0.0f);
        float3 transmitted = float3(0);
        for (uint b = 0; b < skyBatches; ++b) {
            transmitted += float3(
                dot(throughput[b], u.transmittanceRed[b]),
                dot(throughput[b], u.transmittanceGreen[b]),
                dot(throughput[b], u.transmittanceBlue[b])
            );
        }
        inscattering.write(half4(half3(min(rgb, 65504.0f)), 1), uint3(gid, k));
        extinction.write(half4(half3(saturate(transmitted)), 1), uint3(gid, k));
    }
}

vertex float4 frostfield_sky_aerial_vertex(uint id [[vertex_id]]) {
    const float2 points[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(points[id], 0, 1);
}

constant uint aerialSkySteps = 48;

/// Zenith angle of the horizon seen from altitude h, just past 90°.
static float aerialHorizonZenith(float ground, float h) {
    float r = ground + h;
    return acos(-sqrt(h * (2.0f * ground + h)) / r);
}

/// Hillaire's sky-view rows: the upper half runs from the zenith to the horizon and
/// the lower half from the horizon to the nadir, each packed quadratically toward
/// the horizon, which falls between the two middle rows.
static float aerialSkyZenith(float v, float horizon) {
    if (v < 0.5f) {
        float a = 1.0f - 2.0f * v;
        return horizon * (1.0f - a * a);
    }
    float b = 2.0f * v - 1.0f;
    return horizon + (skyPi - horizon) * b * b;
}

/// Sky radiance seen from the camera's altitude, including the ground below the
/// horizon. Columns run over the azimuth from the Sun's vertical plane, 0 to π,
/// since a spherically symmetric sky is mirror-symmetric about that plane.
kernel void frostfield_sky_aerial_sky(
    texture2d<float, access::read> transmittance [[texture(0)]],
    texture2d_array<float, access::read> multiple [[texture(1)]],
    texture2d<float, access::write> output [[texture(5)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant FrostfieldAerialUniforms& u [[buffer(1)]],
    device const float4* skyIrradiance [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 size = uint2(output.get_width(), output.get_height());
    if (any(gid >= size)) {
        return;
    }
    float h = u.planet.w;
    float muSun = clamp(dot(c.sun.xyz, u.planet.xyz), -1.0f, 1.0f);
    float phi = skyPi * (float(gid.x) + 0.5f) / float(size.x);
    float zenith = aerialSkyZenith((float(gid.y) + 0.5f) / float(size.y), aerialHorizonZenith(c.radii.x, h));
    float mu = cos(zenith);
    float nu = mu * muSun + sin(zenith) * sqrt(max(1.0f - muSun * muSun, 0.0f)) * cos(phi);
    SkyRay ray = {h, mu, muSun, clamp(nu, -1.0f, 1.0f)};
    float4 spectrum[4];
    for (uint b = 0; b < skyBatches; ++b) {
        spectrum[b] = skyView(c, transmittance, multiple, ray, aerialSkySteps, b, skyIrradiance[b]);
    }
    float3 rgb = max(skyLinearSRGB(c, spectrum), 0.0f);
    output.write(float4(select(float3(0), rgb, isfinite(rgb)), 1), gid);
}

/// Reads the sky panorama, which is the sky seen from the world origin.
static float3 aerialPanorama(texture2d<float, access::read> panorama, float3 direction) {
    float2 uv = float2(
        atan2(direction.z, direction.x) / (2.0f * skyPi) + 0.5f,
        acos(clamp(direction.y, -1.0f, 1.0f)) / skyPi
    );
    int width = int(panorama.get_width());
    int height = int(panorama.get_height());
    float2 location = uv * float2(width, height) - 0.5f;
    int2 corner = int2(floor(location));
    float2 f = location - floor(location);
    int x0 = (corner.x % width + width) % width;
    int x1 = (x0 + 1) % width;
    int y0 = clamp(corner.y, 0, height - 1);
    int y1 = clamp(corner.y + 1, 0, height - 1);
    float3 north = mix(panorama.read(uint2(x0, y0)).rgb, panorama.read(uint2(x1, y0)).rgb, f.x);
    float3 south = mix(panorama.read(uint2(x0, y1)).rgb, panorama.read(uint2(x1, y1)).rgb, f.x);
    return mix(north, south, f.y);
}

/// Reads the camera's sky table along a world direction, interpolating only
/// within the direction's side of the horizon.
static float3 aerialSky(
    texture2d<float, access::read> table, constant FrostfieldSkyConstants& c,
    constant FrostfieldAerialUniforms& u, float3 direction
) {
    float3 up = u.planet.xyz;
    float mu = clamp(dot(direction, up), -1.0f, 1.0f);
    float3 sunAcross = c.sun.xyz - dot(c.sun.xyz, up) * up;
    float3 viewAcross = direction - mu * up;
    float phi = atan2(length(cross(sunAcross, viewAcross)), dot(sunAcross, viewAcross));
    float horizon = aerialHorizonZenith(c.radii.x, u.planet.w);
    float zenith = acos(mu);
    int width = int(table.get_width());
    int height = int(table.get_height());
    int middle = height / 2;
    float v;
    int first;
    int last;
    if (zenith <= horizon) {
        v = 0.5f * (1.0f - sqrt(max(1.0f - zenith / horizon, 0.0f)));
        first = 0;
        last = middle - 1;
    } else {
        v = 0.5f * (1.0f + sqrt(max((zenith - horizon) / (skyPi - horizon), 0.0f)));
        first = middle;
        last = height - 1;
    }
    float2 location = clamp(
        float2(phi / skyPi * float(width), v * float(height)) - 0.5f, float2(0, first), float2(width - 1, last)
    );
    int2 corner = int2(floor(location));
    float2 f = location - floor(location);
    int x1 = min(corner.x + 1, width - 1);
    int y1 = min(corner.y + 1, last);
    float3 top = mix(table.read(uint2(corner.x, corner.y)).rgb, table.read(uint2(x1, corner.y)).rgb, f.x);
    float3 bottom = mix(table.read(uint2(corner.x, y1)).rgb, table.read(uint2(x1, y1)).rgb, f.x);
    return mix(top, bottom, f.y);
}

fragment float4 frostfield_sky_aerial_fragment(
    float4 position [[position]],
    texture2d<float, access::read> color [[texture(0)]],
    depth2d<float, access::read> depth [[texture(1)]],
    texture3d<float> inscattering [[texture(2)]],
    texture3d<float> extinction [[texture(3)]],
    texture2d<float, access::read> panorama [[texture(4)]],
    texture2d<float, access::read> skyTable [[texture(5)]],
    constant FrostfieldSkyConstants& c [[buffer(0)]],
    constant FrostfieldAerialUniforms& u [[buffer(1)]]
) {
    uint2 pixel = uint2(position.xy);
    float2 size = float2(color.get_width(), color.get_height());
    float2 uv = position.xy / size;
    float4 source = color.read(pixel);
    float stored = depth.read(pixel);
    bool reversed = u.clip.z > 0.5f;
    bool background = reversed ? stored <= 0.0f : stored >= 1.0f;
    float3 direction = aerialDirection(u, uv);
    if (background) {
        if (u.options.x < 0.5f) {
            return source;
        }
        float3 sky = u.options.z > 0.5f ? aerialSky(skyTable, c, u, direction) : aerialPanorama(panorama, direction);
        float cosine = dot(direction, u.sunDisk.xyz);
        if (u.options.y > 0.5f && cosine >= u.sunDisk.w) {
            // Power-law limb darkening across the disk, per wavelength.
            float sine2 = max(1.0f - cosine * cosine, 0.0f);
            float edge2 = max(1.0f - u.sunDisk.w * u.sunDisk.w, 1e-12f);
            float mu = sqrt(max(1.0f - sine2 / edge2, 0.0f));
            float3 disk = float3(0);
            for (uint b = 0; b < skyBatches; ++b) {
                float4 value = u.diskCenter[b] * pow(max(float4(mu), 1e-6f), u.limbDarkening[b]);
                disk += float3(dot(value, c.red[b]), dot(value, c.green[b]), dot(value, c.blue[b]));
            }
            sky += max(disk, 0.0f);
        }
        return float4(min(sky, 65504.0f), 1);
    }

    float nearWeight = reversed ? stored : 1.0f - stored;
    float axial = 1.0f / mix(u.clip.x, u.clip.y, nearWeight);
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float stretch = length(float3(ndc.x * u.right.w, ndc.y * u.up.w, 1.0f));
    float meters = axial * stretch * u.clip.w;
    float slices = float(inscattering.get_depth());
    float w = sqrt(saturate(meters / u.forward.w));
    float3 coordinate = float3(uv, (w * (slices - 1.0f) + 0.5f) / slices);
    constexpr sampler volume(coord::normalized, filter::linear, address::clamp_to_edge);
    float3 scattered = inscattering.sample(volume, coordinate).rgb;
    float3 transmitted = extinction.sample(volume, coordinate).rgb;
    return float4(min(source.rgb * transmitted + scattered * source.a, 65504.0f), source.a);
}
