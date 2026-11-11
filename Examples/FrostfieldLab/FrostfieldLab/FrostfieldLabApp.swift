#if os(iOS)
import SwiftUI
import UIKit

@main
final class FrostfieldLabApp: UIResponder, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        configurationForConnecting session: UISceneSession,
        options: UIScene.ConnectionOptions
    ) -> UISceneConfiguration {
        let configuration = UISceneConfiguration(name: nil, sessionRole: session.role)
        configuration.delegateClass = LabSceneDelegate.self
        return configuration
    }
}

final class LabSceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(
        _ scene: UIScene,
        willConnectTo session: UISceneSession,
        options connectionOptions: UIScene.ConnectionOptions
    ) {
        guard let scene = scene as? UIWindowScene else { return }
        let window = UIWindow(windowScene: scene)
        if LabSceneHost.isRequested {
            window.rootViewController = UIHostingController(rootView: LabSceneHost())
        } else if LabSwiftUIScene.isRequested {
            window.rootViewController = UIHostingController(rootView: LabSwiftUIScene())
        } else if LabOceanHost.isRequested {
            window.rootViewController = UIHostingController(rootView: LabOceanHost())
        } else if LabPathLightingHost.isRequested {
            window.rootViewController = UIHostingController(rootView: LabPathLightingHost())
        } else {
            window.rootViewController = TriangleViewController()
        }
        self.window = window
        window.makeKeyAndVisible()
    }
}
#elseif os(macOS) || os(visionOS)
import SwiftUI
#if os(macOS)
import Foundation
import Metal
#endif

@main
struct FrostfieldLabApp: App {
    #if os(macOS)
    init() {
        let args = CommandLine.arguments
        do {
            if try LabScenes.captureIfRequested(args) || LabScenes.benchmarkIfRequested(args) { exit(0) }
        } catch {
            FileHandle.standardError.write(Data("FrostfieldLab capture failed: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
        let captureModes = [
            "--capture-cyber-ocean", "--capture-pbr-studio", "--capture-procedural-camera", LabStudioFixture.assetStudioCaptureArgument,
            LabPathTracingFixture.captureArgument, LabPathTracingFixture.icebergCaptureArgument,
            LabPathTracingFixture.porscheCaptureArgument,
        ]
        guard let index = args.firstIndex(where: { captureModes.contains($0) }) else {
            return
        }
        do {
            let fields = args.dropFirst(index + 1)
            guard let path = fields.first, let device = MTLCreateSystemDefaultDevice() else {
                throw CocoaError(.fileWriteUnknown, userInfo: [
                    NSLocalizedDescriptionKey: "A capture requires an output PNG path and a Metal device."
                ])
            }
            let url = URL(fileURLWithPath: path)
            let iceberg = args[index] == LabPathTracingFixture.icebergCaptureArgument
            let porsche = args[index] == LabPathTracingFixture.porscheCaptureArgument
            if iceberg || porsche || args[index] == LabPathTracingFixture.captureArgument {
                var flags: Set<String> = [
                    "--studio-detail", "--path-raw", "--path-direct", "--path-light-room", "--shader-finish", "--path-look-split",
                ]
                if porsche { flags.insert("--path-preview") }
                let words = Array(fields.dropFirst().filter { !flags.contains($0) })
                let defaultSamples = porsche ? 256 : (iceberg ? 192 : 128)
                let defaultYaw = porsche ? LabPathTracingFixture.porscheYaw : (iceberg ? LabPathTracingFixture.icebergYaw : 0)
                let samples = words.isEmpty ? defaultSamples : Int(words[0])
                let yaw: Float? = words.count < 2 ? defaultYaw : Float(words[1])
                guard words.count <= 2, let samples, (1...2048).contains(samples), let yaw, yaw.isFinite else {
                    throw CocoaError(.fileWriteUnknown, userInfo: [
                        NSLocalizedDescriptionKey: "Use \(args[index]) OUTPUT.png [SAMPLES_PER_PIXEL] [YAW_RADIANS]."
                    ])
                }
                let subject: LabPathTracingFixture.Subject
                if porsche { subject = .porsche }
                else if iceberg { subject = .iceberg }
                else if fields.contains("--path-light-room") { subject = .lightRoom }
                else { subject = .rangefinder }
                let generatedFinish = fields.contains("--shader-finish")
                let splitLighting = fields.contains("--path-look-split")
                guard subject == .rangefinder || (!generatedFinish && !splitLighting) else {
                    throw CocoaError(.fileWriteUnknown, userInfo: [
                        NSLocalizedDescriptionKey: "The shader finish and split lighting preset apply to the camera studio."
                    ])
                }
                let fixture = try LabPathTracingFixture(
                    device: device, destinationFormat: .bgra8Unorm_srgb, detail: fields.contains("--studio-detail"),
                    subject: subject, generatedFinish: generatedFinish
                )
                if splitLighting { try fixture.setLighting(.split) }
                try fixture.capture(
                    to: url, samples: samples, yaw: yaw, bounces: fields.contains("--path-direct") ? 1 : 7,
                    denoise: !fields.contains("--path-raw"), preview: fields.contains("--path-preview")
                )
            } else if args[index] != "--capture-cyber-ocean" {
                let bakeOptions: Set<String> = [
                    "--studio-bake-reference", "--studio-bake-indirect",
                    "--studio-bake-shared-reference", "--studio-bake-shared-indirect", "--studio-bake-shared-shader",
                ]
                let bakeFlags = fields.dropFirst().filter { bakeOptions.contains($0) }
                let isAssetStudio = args[index] == LabStudioFixture.assetStudioCaptureArgument
                guard bakeFlags.count <= 1, isAssetStudio || bakeFlags.isEmpty else {
                    throw CocoaError(.fileWriteUnknown, userInfo: [
                        NSLocalizedDescriptionKey: "Choose at most one bake variant with --capture-asset-studio.",
                    ])
                }
                let sweepBake: LabAssetBaking.SweepBake = switch bakeFlags.first {
                case "--studio-bake-reference": .progressiveReference
                case "--studio-bake-indirect": .progressiveIndirect
                case "--studio-bake-shared-reference": .sharedReference
                case "--studio-bake-shared-indirect": .sharedIndirect
                case "--studio-bake-shared-shader": .sharedShader
                default: .staticDirect
                }
                let flags = bakeOptions.union([
                    "--studio-direct-only", "--studio-detail", "--studio-sky", "--studio-light-rig", "--studio-no-ao", "--shader-finish",
                ])
                let options = fields.dropFirst().filter { !flags.contains($0) && !$0.hasPrefix("--studio-asset=") }
                let angle = options.first.flatMap(Float.init)
                guard options.isEmpty || (options.count == 1 && angle?.isFinite == true) else {
                    throw CocoaError(.fileWriteUnknown, userInfo: [
                        NSLocalizedDescriptionKey: "Use \(args[index]) OUTPUT.png [YAW_RADIANS]."
                    ])
                }
                let fixture = try LabStudioFixture(
                    device: device, destinationFormat: .bgra8Unorm_srgb,
                    environmentEnabled: !fields.contains("--studio-direct-only"),
                    closeUp: fields.contains("--studio-detail"), showsEnvironment: fields.contains("--studio-sky"),
                    usesImportedLights: fields.contains("--studio-light-rig"),
                    occlusionEnabled: !fields.contains("--studio-no-ao"),
                    sourceURL: try LabStudioFixture.sourceURL(arguments: Array(fields.dropFirst())),
                    procedural: args[index] == "--capture-procedural-camera", generatedFinish: fields.contains("--shader-finish"),
                    bakesAssets: isAssetStudio, sweepBake: sweepBake
                )
                try fixture.capture(to: url, angle: angle)
            } else {
                let frames = fields.dropFirst().first.flatMap(Int.init) ?? 96
                guard (1...240).contains(frames) else {
                    throw CocoaError(.fileWriteUnknown, userInfo: [
                        NSLocalizedDescriptionKey: "Use --capture-cyber-ocean OUTPUT.png [FRAMES], with 1 to 240 frames."
                    ])
                }
                let fixture = try LabOceanFixture(device: device, destinationFormat: .bgra8Unorm_srgb)
                try fixture.capture(to: url, frames: frames)
            }
            exit(0)
        } catch {
            let message = "FrostfieldLab capture failed: \(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(message.utf8))
            exit(1)
        }
    }
    #endif
    #if os(visionOS)
    @State private var immersivePresentation = LabImmersivePresentation()
    #endif

    var body: some Scene {
        #if os(visionOS)
        WindowGroup("FrostfieldLab", id: LabImmersivePresentation.windowID) {
            LabWindowContent(immersivePresentation: immersivePresentation)
        }
        .defaultSize(width: 900, height: 640)

        LabImmersiveScene(presentation: immersivePresentation)
        #else
        WindowGroup("FrostfieldLab") {
            LabWindowContent()
        }
        .defaultSize(width: 900, height: 640)
        #endif
    }
}

private struct LabWindowContent: View {
    #if os(visionOS)
    let immersivePresentation: LabImmersivePresentation
    #endif

    var body: some View {
        #if os(visionOS)
        if LabImmersiveLauncher.isRequested {
            LabImmersiveLauncher(presentation: immersivePresentation)
        } else {
            windowContent
        }
        #else
        windowContent
        #endif
    }

    @ViewBuilder private var windowContent: some View {
        if LabSceneHost.isRequested {
            LabSceneHost()
        } else if LabSwiftUIScene.isRequested {
            LabSwiftUIScene()
        } else if LabOceanHost.isRequested {
            LabOceanHost()
        } else if LabPathLightingHost.isRequested {
            LabPathLightingHost()
        } else {
            TriangleHost()
        }
    }
}

#endif

#if os(macOS)
struct TriangleHost: NSViewControllerRepresentable {
    var oceanPaused: Bool? = nil
    var pathLighting: LabPathLightSettings? = nil
    var sceneControls: LabSceneControls? = nil
    /// Changes when a scene control changes, so SwiftUI forwards the edit.
    var sceneRevision = 0

    func makeNSViewController(context: Context) -> TriangleViewController {
        let controller = TriangleViewController()
        if let pathLighting { controller.setPathLighting(pathLighting) }
        if let sceneControls { controller.setSceneControls(sceneControls) }
        return controller
    }

    func updateNSViewController(_ controller: TriangleViewController, context: Context) {
        if let oceanPaused { controller.setOceanPaused(oceanPaused) }
        if let pathLighting { controller.setPathLighting(pathLighting) }
        if let sceneControls { controller.setSceneControls(sceneControls) }
    }
}
#else
struct TriangleHost: UIViewControllerRepresentable {
    var oceanPaused: Bool? = nil
    var pathLighting: LabPathLightSettings? = nil
    var sceneControls: LabSceneControls? = nil
    /// Changes when a scene control changes, so SwiftUI forwards the edit.
    var sceneRevision = 0

    func makeUIViewController(context: Context) -> TriangleViewController {
        let controller = TriangleViewController()
        if let pathLighting { controller.setPathLighting(pathLighting) }
        if let sceneControls { controller.setSceneControls(sceneControls) }
        return controller
    }

    func updateUIViewController(_ controller: TriangleViewController, context: Context) {
        if let oceanPaused { controller.setOceanPaused(oceanPaused) }
        if let pathLighting { controller.setPathLighting(pathLighting) }
        if let sceneControls { controller.setSceneControls(sceneControls) }
    }
}
#endif
