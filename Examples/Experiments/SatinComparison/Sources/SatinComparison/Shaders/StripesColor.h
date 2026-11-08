// The custom shading both renderers share: UV as red and green, and six
// vertical blue bands. No derivatives or textures, so equal interpolated UVs
// give equal colors.
#ifndef STRIPES_COLOR_H
#define STRIPES_COLOR_H

static inline float4 stripesColor(float2 uv) {
    float band = step(0.5, fract(uv.x * 6.0));
    return float4(uv.x, uv.y, band, 1.0);
}

#endif
