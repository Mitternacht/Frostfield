extension ShaderProgram {
    static func proceduralNoiseFunctions(prefix: String) -> String {
        """
        static inline int \(prefix)_noise_permutation(int value) {
            return ((34 * value + 1) * value) % 289;
        }

        static inline float \(prefix)_simplex_corner(int3 index, float3 offset) {
            int hash = \(prefix)_noise_permutation(
                \(prefix)_noise_permutation(\(prefix)_noise_permutation(index.z) + index.y) + index.x
            );
            int square = hash % 49;
            float x = float(square / 7) * (2.0f / 7.0f) - 13.0f / 14.0f;
            float y = float(square % 7) * (2.0f / 7.0f) - 13.0f / 14.0f;
            float z = 1.0f - abs(x) - abs(y);
            float2 plane = float2(x, y);
            if (z <= 0.0f) { plane -= 2.0f * floor(plane) + 1.0f; }
            float weight = max(0.6f - dot(offset, offset), 0.0f);
            return weight * weight * weight * dot(normalize(float3(plane, z)), offset);
        }

        static inline float \(prefix)_simplex(float3 position) {
            float3 p = frostfield_graph_noise_point(position);
            float3 lattice = floor(p + dot(p, float3(1.0f / 3.0f)));
            float3 base = p - lattice + dot(lattice, float3(1.0f / 6.0f));
            float3 order = step(base.yzx, base);
            float3 opposite = 1.0f - order.zxy;
            float3 first = min(order, opposite);
            float3 second = max(order, opposite);
            int3 cell = (int3(lattice) % 289 + 289) % 289;
            float result = \(prefix)_simplex_corner(cell, base)
                + \(prefix)_simplex_corner(cell + int3(first), base - first + 1.0f / 6.0f)
                + \(prefix)_simplex_corner(cell + int3(second), base - second + 1.0f / 3.0f)
                + \(prefix)_simplex_corner(cell + 1, base - 0.5f);
            return 0.5f + 12.0f * result;
        }

        static inline float3 \(prefix)_simplex_field(float3 p) {
            return float3(
                \(prefix)_simplex(p * 2.0f - 1.0f),
                2.0f * \(prefix)_simplex(p.yzx + float3(-19.1f, 33.4f, 47.2f)) - 1.0f,
                2.0f * \(prefix)_simplex(p.zxy + float3(74.2f, -124.5f, 99.4f)) - 1.0f
            );
        }

        static inline float3 \(prefix)_curl(float3 p) {
            const float3 dx(0.1f, 0.0f, 0.0f), dy(0.0f, 0.1f, 0.0f), dz(0.0f, 0.0f, 0.1f);
            float3 x = \(prefix)_simplex_field(p + dx) - \(prefix)_simplex_field(p - dx);
            float3 y = \(prefix)_simplex_field(p + dy) - \(prefix)_simplex_field(p - dy);
            float3 z = \(prefix)_simplex_field(p + dz) - \(prefix)_simplex_field(p - dz);
            return float3(y.z - z.y, z.x - x.z, x.y - y.x) * 5.0f;
        }

        static inline float \(prefix)_voronoi(float2 position, float time) {
            float2 p = frostfield_graph_noise_point(float3(position, 0.0f)).xy;
            float2 lattice = floor(p), local = p - lattice;
            float nearest = 8.0f;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    float2 offset(x, y);
                    float2 cell = lattice + offset;
                    float2 phase = fract(sin(float2(dot(cell, float2(127.1f, 311.7f)),
                                                    dot(cell, float2(269.5f, 183.3f)))) * 18.5453f);
                    float2 delta = offset - local + sin(time + phase * 6.283185307179586f) * 0.5f + 0.5f;
                    nearest = min(nearest, dot(delta, delta));
                }
            }
            return nearest;
        }

        static inline float \(prefix)_voronoi(float3 position, float time) {
            float3 p = frostfield_graph_noise_point(position);
            float3 lattice = floor(p), local = p - lattice;
            float nearest = 8.0f;
            for (int z = -1; z <= 1; ++z) {
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        float3 offset(x, y, z);
                        float3 cell = lattice + offset;
                        float3 phase = fract(sin(float3(dot(cell, float3(127.1f, 311.7f, 74.7f)),
                                                        dot(cell, float3(269.5f, 183.3f, 246.1f)),
                                                        dot(cell, float3(113.5f, 271.9f, 124.6f)))) * 18.5453f);
                        float3 delta = offset - local + sin(time + phase * 6.283185307179586f) * 0.5f + 0.5f;
                        nearest = min(nearest, dot(delta, delta));
                    }
                }
            }
            return nearest;
        }
        """
    }
}
