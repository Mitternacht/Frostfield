import Frostfield
import Testing
import simd

struct DirectionalShadowMapTests {
    @Test
    func `Shadow region maps its center and light facing planes to Metal depth`() throws {
        let center = SIMD3<Float>(2, 3, -4)
        let result = try DirectionalShadowMap.projection(
            directionToLight: SIMD3<Float>(0, 0, 7), center: center, halfExtent: 2
        )
        let matrix = result.clipFromWorld
        #expect(result.directionToLight == SIMD3<Float>(0, 0, 1))
        #expect(matrix * SIMD4<Float>(center, 1) == SIMD4<Float>(0, 0, 0.5, 1))
        #expect(matrix * SIMD4<Float>(center + SIMD3<Float>(2, 2, 2), 1) == SIMD4<Float>(1, 1, 0, 1))
        #expect(matrix * SIMD4<Float>(center - SIMD3<Float>(2, 2, 2), 1) == SIMD4<Float>(-1, -1, 1, 1))
    }

    @Test
    func `A vertical light retains a usable perpendicular basis`() throws {
        let result = try DirectionalShadowMap.projection(
            directionToLight: SIMD3<Float>(0, -2, 0), center: .zero, halfExtent: 1
        )
        let matrix = result.clipFromWorld
        #expect(result.directionToLight == SIMD3<Float>(0, -1, 0))
        #expect(matrix * SIMD4<Float>(0, -1, 0, 1) == SIMD4<Float>(0, 0, 0, 1))
        #expect(matrix * SIMD4<Float>(1, 0, 0, 1) == SIMD4<Float>(0, 1, 0.5, 1))
    }

    @Test
    func `Shadow projection rejects invalid or unrepresentable regions`() {
        #expect(throws: DirectionalShadowMap.Failure.invalidDirection) {
            try DirectionalShadowMap.projection(directionToLight: .zero, center: .zero, halfExtent: 1)
        }
        #expect(throws: DirectionalShadowMap.Failure.invalidRegion) {
            try DirectionalShadowMap.projection(
                directionToLight: SIMD3<Float>(0, 0, 1), center: SIMD3<Float>(.nan, 0, 0), halfExtent: 1
            )
        }
        #expect(throws: DirectionalShadowMap.Failure.unrepresentableRegion) {
            try DirectionalShadowMap.projection(
                directionToLight: SIMD3<Float>(0, 0, 1), center: .zero, halfExtent: .leastNonzeroMagnitude
            )
        }
    }
}
