#include <metal_stdlib>
#include "FrostfieldSoftParticle.h"
using namespace metal;

constant bool frostfield_soft_particles [[function_constant(56)]];

static inline float frostfieldSoftParticleFade(
    float4 position, depth2d<float> sceneDepth, constant float4* softness
) {
    float depth = sceneDepth.read(uint2(position.xy));
    float4 settings = softness[1];
    if (!(depth >= 0.0f && depth <= 1.0f) || depth == settings.z) return 1.0f;
    float particleZ = frostfieldParticleViewZ(position.z, softness[0]);
    float sceneZ = frostfieldParticleViewZ(depth, softness[0]);
    return frostfieldParticleContrast(particleZ, sceneZ, settings.x, settings.y);
}

// Each particle occupies two float4 records shared with ParticleEmitter.swift.
kernel void frostfield_particles_update(
    device float4* particles [[buffer(0)]],
    constant float& deltaTime [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) {
        return;
    }

    float4 velocityLifetime = particles[id * 2 + 1];
    float age = fmod(particles[id * 2].w + deltaTime, velocityLifetime.w);
    float3 position = velocityLifetime.xyz * age + float3(0, -0.7f * age * age, 0);
    particles[id * 2] = float4(position, age);
}

struct ParticleVaryings {
    float4 position [[position]];
    float size [[point_size]];
    float lifetimeProgress;
};

vertex ParticleVaryings frostfield_particles_vertex(
    uint id [[vertex_id]],
    const device float4* particles [[buffer(0)]],
    constant float4x4& clipFromWorld [[buffer(1)]]
) {
    float4 positionAge = particles[id * 2];
    float progress = positionAge.w / particles[id * 2 + 1].w;
    return {clipFromWorld * float4(positionAge.xyz, 1), mix(10.0f, 4.0f, progress), progress};
}

fragment float4 frostfield_particles_fragment(
    ParticleVaryings input [[stage_in]],
    float2 point [[point_coord]],
    depth2d<float> sceneDepth [[texture(0), function_constant(frostfield_soft_particles)]],
    constant float4* softness [[buffer(1), function_constant(frostfield_soft_particles)]]
) {
    float radius = length(point * 2.0f - 1.0f);
    float edge = 1.0f - smoothstep(0.55f, 1.0f, radius);
    float progress = input.lifetimeProgress;
    float envelope = smoothstep(0.0f, 0.12f, progress) * (1.0f - smoothstep(0.65f, 1.0f, progress));
    float alpha = edge * envelope * 0.8f;
    if (frostfield_soft_particles) alpha *= frostfieldSoftParticleFade(input.position, sceneDepth, softness);
    float3 color = mix(float3(0.3f, 0.85f, 1.0f), float3(0.72f, 0.35f, 1.0f), progress);
    return float4(color * alpha, alpha);
}

// Each flow particle stores position/age, launch position/lifetime, and normal/drift.
static float3 frostfieldFlowField(float3 position, float frequency) {
    float3 phase = position * frequency;
    float3 sine = sin(phase);
    float3 cosine = cos(phase);
    // Each component is independent of its own axis, making the field divergence-free.
    return 0.5f * (sine.yzx - cosine.zxy);
}

kernel void frostfield_surface_flow_update(
    device float4* particles [[buffer(0)]],
    constant float& deltaTime [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float4& dynamics [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) {
        return;
    }

    uint offset = id * 3;
    float4 state = particles[offset];
    float4 launch = particles[offset + 1];
    float4 normal = particles[offset + 2];
    float age = state.w + deltaTime;
    float step = deltaTime;
    float3 position = state.xyz;
    if (age >= launch.w) {
        age -= launch.w;
        step = age;
        position = launch.xyz;
    }

    float3 drift = normal.xyz * (normal.w * dynamics.z);
    float3 velocity = frostfieldFlowField(position, dynamics.y) * dynamics.x + drift;
    float3 midpoint = position + 0.5f * step * velocity;
    position += step * (frostfieldFlowField(midpoint, dynamics.y) * dynamics.x + drift);
    particles[offset] = float4(position, age);
}

vertex ParticleVaryings frostfield_surface_flow_vertex(
    uint id [[vertex_id]],
    const device float4* particles [[buffer(0)]],
    constant float4x4& clipFromWorld [[buffer(1)]],
    constant float& pointSize [[buffer(2)]]
) {
    uint offset = id * 3;
    float4 positionAge = particles[offset];
    float progress = saturate(positionAge.w / particles[offset + 1].w);
    float size = max(1.0f, pointSize * mix(1.0f, 0.45f, progress));
    return {clipFromWorld * float4(positionAge.xyz, 1), size, progress};
}

fragment float4 frostfield_surface_flow_fragment(
    ParticleVaryings input [[stage_in]],
    float2 point [[point_coord]],
    constant float4& emission [[buffer(0)]],
    depth2d<float> sceneDepth [[texture(0), function_constant(frostfield_soft_particles)]],
    constant float4* softness [[buffer(1), function_constant(frostfield_soft_particles)]]
) {
    float radius = length(point * 2.0f - 1.0f);
    float halo = 1.0f - smoothstep(0.08f, 1.0f, radius);
    float core = 1.0f - smoothstep(0.05f, 0.43f, radius);
    float progress = input.lifetimeProgress;
    float fade = smoothstep(0.0f, 0.12f, progress) * (1.0f - smoothstep(0.55f, 1.0f, progress));
    float strength = fade * (0.55f * halo + 0.45f * core) * emission.w;
    float3 color = mix(emission.rgb, float3(1), 0.35f * core);
    float4 result = float4(color * strength, saturate(strength));
    if (frostfield_soft_particles) result *= frostfieldSoftParticleFade(input.position, sceneDepth, softness);
    return result;
}

struct ShaderParticleVaryings {
    float4 position [[position]];
    float2 corner;
    float3 color;
    float fade;
};

// axes: camera right, camera up, default color and size, field count.
vertex ShaderParticleVaryings frostfield_shader_particle_vertex(
    uint vertexID [[vertex_id]], uint instance [[instance_id]],
    const device float4* fields [[buffer(0)]],
    constant float4x4& clipFromWorld [[buffer(1)]],
    constant float4* axes [[buffer(2)]]
) {
    constexpr float2 corners[6] = { float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1) };
    uint count = uint(axes[3].x);
    float4 positionAge = fields[instance * count];
    float lifetime = fields[instance * count + 1].w;
    float4 colorSize = count > 2 ? fields[instance * count + 2] : axes[2];
    float progress = lifetime > 0.0f ? saturate(positionAge.w / lifetime) : 1.0f;
    float fade = lifetime > 0.0f ? smoothstep(0.0f, 0.1f, progress) * (1.0f - smoothstep(0.7f, 1.0f, progress)) : 0.0f;
    float2 corner = corners[vertexID];
    float radius = max(colorSize.w, 0.0f) * 0.5f;
    float3 world = positionAge.xyz + (axes[0].xyz * corner.x + axes[1].xyz * corner.y) * radius;
    ShaderParticleVaryings out;
    out.position = clipFromWorld * float4(world, 1);
    out.corner = corner;
    out.color = max(colorSize.xyz, float3(0));
    out.fade = fade;
    return out;
}

fragment float4 frostfield_shader_particle_fragment(
    ShaderParticleVaryings input [[stage_in]],
    depth2d<float> sceneDepth [[texture(0), function_constant(frostfield_soft_particles)]],
    constant float4* softness [[buffer(1), function_constant(frostfield_soft_particles)]]
) {
    float falloff = 1.0f - smoothstep(0.2f, 1.0f, length(input.corner));
    float alpha = falloff * input.fade;
    if (frostfield_soft_particles) alpha *= frostfieldSoftParticleFade(input.position, sceneDepth, softness);
    if (alpha <= 0.0f) { discard_fragment(); }
    return float4(min(input.color, float3(65504.0f)) * alpha, alpha);
}
