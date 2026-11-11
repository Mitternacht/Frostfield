import Foundation
import Metal
import simd

// Morph targets, instancing and vertex colors specialize the mesh vertex stages in
// Scene.metal. Raster vertex buffers 6 to 10 belong to them: 6 instance records, 7
// packed morph offsets, 8 the draw's nonzero weights, 9 the morph layout and 10 colors.
extension MeshRenderer {
    /// The function constants that select the specialized vertex stage's features.
    struct VertexSpecialization: Hashable {
        let skinned: Bool
        let morphed: Bool
        let instanced: Bool
        var colored = false

        /// Plain and skin-only meshes keep the original vertex functions.
        var isNeeded: Bool { morphed || instanced || colored }

        func function(_ name: String, in library: any MTLLibrary, objectNormals: Bool = false) throws -> any MTLFunction {
            let constants = MTLFunctionConstantValues()
            var skinned = skinned
            var morphed = morphed
            var instanced = instanced
            var colored = colored
            var objectNormals = objectNormals
            constants.setConstantValue(&skinned, type: .bool, index: 30)
            constants.setConstantValue(&morphed, type: .bool, index: 31)
            constants.setConstantValue(&instanced, type: .bool, index: 32)
            constants.setConstantValue(&colored, type: .bool, index: 33)
            constants.setConstantValue(&objectNormals, type: .bool, index: 34)
            do {
                return try library.makeFunction(name: name, constantValues: constants)
            } catch {
                throw Failure.function("\(name) (\(error.localizedDescription))")
            }
        }
    }

    /// Depth-only pipelines for a specialized vertex stage. Directional and punctual
    /// shadow passes share them because both use depth32Float and set their own depth state.
    struct ShadowPipelines {
        let opaque: any MTLRenderPipelineState
        let masked: (any MTLRenderPipelineState)?

        init(
            specialization: VertexSpecialization, masked: Bool, library: any MTLLibrary, device: any MTLDevice,
            indirectCommands: Bool
        ) throws {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "Frostfield specialized shadow depth"
            descriptor.vertexFunction = try specialization.function("frostfield_shadow_vertex_specialized", in: library)
            descriptor.depthAttachmentPixelFormat = .depth32Float
            descriptor.inputPrimitiveTopology = .triangle
            descriptor.supportIndirectCommandBuffers = indirectCommands
            do {
                opaque = try device.makeRenderPipelineState(descriptor: descriptor)
                if masked {
                    guard let mask = library.makeFunction(name: "frostfield_shadow_mask_specialized") else {
                        throw Failure.function("frostfield_shadow_mask_specialized")
                    }
                    descriptor.label = "Frostfield specialized masked shadow depth"
                    descriptor.fragmentFunction = mask
                    self.masked = try device.makeRenderPipelineState(descriptor: descriptor)
                } else {
                    self.masked = nil
                }
            } catch let failure as Failure {
                throw failure
            } catch {
                throw Failure.pipeline(error.localizedDescription)
            }
        }
    }

    /// The morph layout a draw binds at vertex buffer 9, matching `FrostfieldMorphLayout`
    /// in Scene.metal: five 32-bit values, 20 bytes.
    package struct PackedMorphLayout {
        package var activeCount: UInt32
        package var vertexCount: UInt32
        /// Element offsets of each attribute's section, or `UInt32.max` when no target has it.
        package var positionBase: UInt32
        package var normalBase: UInt32
        package var tangentBase: UInt32
    }

    /// One nonzero weight bound at vertex buffer 8, matching `FrostfieldMorphWeight`: 8 bytes.
    package struct PackedMorphWeight {
        package var target: UInt32
        package var weight: Float
    }

    struct MorphDraw {
        var layout: PackedMorphLayout
        let weights: [PackedMorphWeight]

        func bind(into encoder: any MTLRenderCommandEncoder, weightsIndex: Int = 8, layoutIndex: Int = 9) {
            weights.withUnsafeBytes { bytes in
                encoder.setVertexBytes(bytes.baseAddress!, length: bytes.count, index: weightsIndex)
            }
            var layout = layout
            encoder.setVertexBytes(&layout, length: MemoryLayout<PackedMorphLayout>.size, index: layoutIndex)
        }
    }

    /// Morph offsets uploaded once per mesh. Each attribute that any target uses
    /// gets a section of `targets × vertices` packed float3 values, ordered by
    /// target then vertex, with zeros for targets that lack the attribute.
    struct PreparedMorph {
        let offsets: any MTLBuffer
        let layout: PackedMorphLayout
        /// Per target, the box around its position offsets, or nil without position offsets.
        let positionBounds: [(lower: SIMD3<Float>, upper: SIMD3<Float>)?]

        init(_ targets: MeshMorphTargets, device: any MTLDevice) throws {
            let attributes: [KeyPath<MeshMorphTargets.Target, [SIMD3<Float>]?>] = [
                \.positionOffsets, \.normalOffsets, \.tangentOffsets,
            ]
            let used = attributes.map { path in targets.targets.contains { $0[keyPath: path] != nil } }
            let sectionLength = targets.vertexCount * targets.targets.count
            let (elementCount, overflow) = sectionLength.multipliedReportingOverflow(by: used.count(where: { $0 }))
            guard !overflow, elementCount < Int(UInt32.max), elementCount <= device.maxBufferLength / 12 else {
                throw Failure.resource("morph target offsets")
            }
            var floats: [Float] = []
            floats.reserveCapacity(elementCount * 3)
            var bases = [UInt32](repeating: .max, count: 3)
            for (index, path) in attributes.enumerated() where used[index] {
                bases[index] = UInt32(floats.count / 3)
                for target in targets.targets {
                    guard let values = target[keyPath: path] else {
                        floats.append(contentsOf: repeatElement(0, count: targets.vertexCount * 3))
                        continue
                    }
                    for value in values {
                        floats.append(value.x)
                        floats.append(value.y)
                        floats.append(value.z)
                    }
                }
            }
            let buffer = floats.withUnsafeBytes { bytes in
                bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared) }
            }
            guard let buffer else { throw Failure.resource("morph target offsets") }
            buffer.label = "mesh morph offsets"
            offsets = buffer
            layout = PackedMorphLayout(
                activeCount: 0, vertexCount: UInt32(targets.vertexCount),
                positionBase: bases[0], normalBase: bases[1], tangentBase: bases[2]
            )
            positionBounds = targets.targets.map { target in
                target.positionOffsets.map { values in
                    values.reduce((lower: values[0], upper: values[0])) { box, value in
                        (simd_min(box.lower, value), simd_max(box.upper, value))
                    }
                }
            }
        }

        /// The layout and weight list for a draw. Zero weights are skipped; an
        /// all-zero draw binds one unused entry, since Metal needs a nonempty buffer.
        func draw(weights: [Float]?) -> MorphDraw {
            var entries: [PackedMorphWeight] = []
            for (index, weight) in (weights ?? []).enumerated() where weight != 0 {
                entries.append(PackedMorphWeight(target: UInt32(index), weight: weight))
            }
            var layout = layout
            layout.activeCount = UInt32(entries.count)
            if entries.isEmpty { entries.append(PackedMorphWeight(target: 0, weight: 0)) }
            return MorphDraw(layout: layout, weights: entries)
        }

        /// A local box that encloses every morphed position: the base box plus each
        /// weighted target's offset box. Nil when the result leaves the Float range.
        func bounds(base: AxisAlignedBox, weights: [Float]?) -> AxisAlignedBox? {
            var lower = boundsDouble(base.minimum)
            var upper = boundsDouble(base.maximum)
            for (index, weight) in (weights ?? []).enumerated() where weight != 0 {
                guard let offsets = positionBounds[index] else { continue }
                let a = boundsDouble(offsets.lower) * Double(weight)
                let b = boundsDouble(offsets.upper) * Double(weight)
                lower += simd_min(a, b)
                upper += simd_max(a, b)
            }
            let box = AxisAlignedBox(validatedMinimum: boundsFloatBelow(lower), maximum: boundsFloatAbove(upper))
            return boundsFinite(box.minimum) && boundsFinite(box.maximum) ? box : nil
        }
    }

    static func validate(_ targets: MeshMorphTargets, geometry: MeshGeometry, material: MeshMaterial) throws {
        guard targets.vertexCount == geometry.vertices.count else {
            throw MeshMorphTargets.ValidationError.geometryVertexCount(
                expected: targets.vertexCount, actual: geometry.vertices.count
            )
        }
        guard targets.targets.count <= maximumMorphTargetCount else { throw Failure.morphTargetCount }
        if targets.targets.contains(where: { $0.tangentOffsets != nil }),
           geometry.vertices.contains(where: { $0.tangent == nil }) {
            throw MeshMorphTargets.ValidationError.missingBaseTangents
        }
        if case .physicallyBased(let material) = material, material.surfaceGraph?.hasVertexOutputs == true {
            throw Failure.morphedDeformation
        }
    }
}
