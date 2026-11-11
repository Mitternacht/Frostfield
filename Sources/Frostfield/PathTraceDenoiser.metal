#include <metal_stdlib>
using namespace metal;

// An à trous kernel, with edge stops derived from the pinhole first hit.
// Dammertz et al., Edge-Avoiding À-Trous Wavelet Transform, HPG 2010.
kernel void frostfield_path_denoise(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::read> normalDepth [[texture(1)]],
    texture2d<float, access::read> albedoRoughness [[texture(2)]],
    texture2d<float, access::write> output [[texture(3)]],
    constant uint& spacing [[buffer(0)]], uint2 pixel [[thread_position_in_grid]]
) {
    uint2 extent(output.get_width(), output.get_height());
    if (any(pixel >= extent)) { return; }
    float4 center = source.read(pixel);
    float4 geometry = normalDepth.read(pixel);
    if (geometry.w <= 0) {
        output.write(center, pixel);
        return;
    }

    constexpr float kernelWeights[] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    constexpr float3 luma(0.2126f, 0.7152f, 0.0722f);
    float4 surface = albedoRoughness.read(pixel);
    float centerLuma = log2(1.0f + max(dot(center.rgb, luma), 0.0f));
    float depthSigma = max(0.012f, 0.007f * geometry.w) * float(spacing);
    float colorSigma = mix(0.12f, 0.42f, saturate(surface.w));
    float3 sum = center.rgb * 36.0f;
    float total = 36.0f;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            if (x == 0 && y == 0) { continue; }
            int2 tap = int2(pixel) + int2(x, y) * int(spacing);
            if (any(tap < 0) || any(uint2(tap) >= extent)) { continue; }
            float4 hit = normalDepth.read(uint2(tap));
            if (hit.w <= 0) { continue; }
            float3 albedo = albedoRoughness.read(uint2(tap)).rgb;
            float3 rgb = source.read(uint2(tap)).rgb;
            float normalError = max(0.0f, 1.0f - dot(geometry.xyz, hit.xyz));
            float depthError = abs(geometry.w - hit.w) / depthSigma;
            float albedoError = distance(surface.rgb, albedo);
            float colorError = abs(centerLuma - log2(1.0f + max(dot(rgb, luma), 0.0f)));
            float weight = kernelWeights[x + 2] * kernelWeights[y + 2]
                * exp(-18.0f * normalError - depthError - 13.0f * albedoError - colorError / colorSigma);
            sum += rgb * weight;
            total += weight;
        }
    }
    output.write(float4(sum / total, center.a), pixel);
}
