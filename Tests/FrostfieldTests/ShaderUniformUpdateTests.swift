@testable import Frostfield
import Testing
import simd

struct ShaderUniformUpdateTests {
    @Test("A frame hook and an object hook edit the values the renderer reads")
    func hooks() throws {
        let uniforms = try ShaderUniforms([ShaderUniform("Gain", Float(1))])
        #expect(uniforms.bufferSnapshot().values[0].x == 1)

        uniforms.setFrameUpdate { values in
            try? values.set("Gain", to: 4)
        }
        #expect(uniforms.bufferSnapshot().values[0].x == 4)

        uniforms.setFrameUpdate(nil)
        uniforms.setObjectUpdate { values, object in
            try? values.set("Gain", to: object.position.x)
        }
        let moved = simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(7, 0, 0, 1)
        ))
        let snapshot = uniforms.bufferSnapshot(object: ShaderObjectContext(world: moved))
        #expect(snapshot.values[0].x == 7)
        #expect(uniforms.bufferSnapshot().values[0].x == 7)
    }
}
