import Foundation
import Frostfield
import FrostfieldSwiftUI
import Metal
import SwiftUI
import simd

struct LabSwiftUIScene: View {
    private static let showsGraph = ProcessInfo.processInfo.arguments.contains("--swiftui-graph")
    static let isRequested = showsGraph || ProcessInfo.processInfo.arguments.contains("--swiftui-mesh")

    @State private var prepared: PreparedScene?
    @State private var errorMessage: String?
    @State private var rotationStep = 0

    var body: some View {
        Group {
            if let errorMessage {
                ContentUnavailableView(
                    "Mesh Unavailable",
                    systemImage: "exclamationmark.triangle",
                    description: Text(verbatim: errorMessage)
                )
            } else if let prepared {
                GeometryReader { geometry in
                    let aspect = Float(max(geometry.size.width, 1) / max(geometry.size.height, 1))
                    if let queue = prepared.graphQueue {
                        FrostfieldSceneView(
                            commandQueue: queue,
                            graph: prepared.graph(rotationStep: rotationStep, aspect: aspect),
                            onError: { errorMessage = $0.localizedDescription }
                        )
                    } else {
                        FrostfieldSceneView(
                            renderer: prepared.renderer,
                            scene: prepared.scene(rotationStep: rotationStep, aspect: aspect),
                            onError: { errorMessage = $0.localizedDescription }
                        )
                    }
                }
                .overlay(alignment: .bottom) {
                    Button(Self.showsGraph ? "Rotate Scene" : "Rotate Cube", systemImage: "rotate.right") {
                        rotationStep = (rotationStep + 1) % 8
                    }
                    .buttonStyle(.borderedProminent)
                    .padding()
                }
            } else {
                ProgressView("Preparing Mesh")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task { prepare() }
    }

    private func prepare() {
        guard prepared == nil, errorMessage == nil else { return }
        guard !ProcessInfo.processInfo.arguments.contains("--simulate-no-metal"),
              let device = MTLCreateSystemDefaultDevice() else {
            errorMessage = String(localized: "No Metal device is available.")
            return
        }

        do {
            let camera = try PerspectiveCamera(verticalFieldOfView: .pi / 4, near: 0.1, far: 100)
            let renderer = try MeshRenderer(device: device, colorPixelFormat: .bgra8Unorm_srgb, geometry: .cube())
            let queue = Self.showsGraph ? device.makeCommandQueue() : nil
            if Self.showsGraph && queue == nil { throw MeshRenderer.Failure.resource("command queue") }
            prepared = PreparedScene(renderer: renderer, camera: camera, graphQueue: queue)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private struct PreparedScene {
        let renderer: MeshRenderer
        let camera: PerspectiveCamera
        let graphQueue: (any MTLCommandQueue)?
        private let groupID = UUID()
        private let leftID = UUID()
        private let rightID = UUID()

        func scene(rotationStep: Int, aspect: Float) -> MeshScene {
            return MeshScene(
                camera: camera,
                transform: SceneTransform(rotation: rotation(rotationStep)),
                cameraTransform: SceneTransform(translation: SIMD3<Float>(0, 0, 2.8 / min(1, aspect))),
                materialParameters: SIMD4<Float>(0.72, 0.86, 1, 1)
            )
        }

        func graph(rotationStep: Int, aspect: Float) -> SceneGraph {
            let group = SceneNode(
                id: groupID, transform: SceneTransform(rotation: rotation(rotationStep)),
                children: [
                    SceneNode(
                        id: leftID, transform: SceneTransform(translation: SIMD3(-0.75, 0, 0)),
                        mesh: renderer, materialParameters: SIMD4(0.85, 0.54, 0.9, 1)
                    ),
                    SceneNode(
                        id: rightID, transform: SceneTransform(translation: SIMD3(0.75, 0, 0)),
                        mesh: renderer, materialParameters: SIMD4(0.35, 0.79, 0.64, 1)
                    ),
                ]
            )
            return SceneGraph(
                roots: [group], camera: camera,
                cameraTransform: SceneTransform(translation: SIMD3(0, 0, 4.6 / min(1, aspect)))
            )
        }

        private func rotation(_ step: Int) -> simd_quatf {
            let pitch = simd_quatf(angle: -0.38, axis: SIMD3<Float>(1, 0, 0))
            let yaw = simd_quatf(angle: 0.6 + Float(step) * .pi / 4, axis: SIMD3<Float>(0, 1, 0))
            return yaw * pitch
        }
    }
}
