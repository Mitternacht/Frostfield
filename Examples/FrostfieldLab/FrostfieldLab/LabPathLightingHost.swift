import Foundation
import Frostfield
import simd
import SwiftUI

struct LabPathLightSettings: Equatable {
    enum Role: Int, CaseIterable, Identifiable {
        case key, fill, rim, front
        var id: Self { self }

        var title: String {
            switch self {
            case .key: "Key"
            case .fill: "Fill"
            case .rim: "Rim"
            case .front: "Front"
            }
        }
    }

    struct Light: Equatable {
        var power: Double = 1
        var size: Double = 1
        var angle: Double = 0
        var tint: Double = 0
    }

    private var lights = Array(repeating: Light(), count: Role.allCases.count)
    var exposure: Double = -0.22

    subscript(role: Role) -> Light {
        get { lights[role.rawValue] }
        set { lights[role.rawValue] = newValue }
    }

    static var split: Self {
        var settings = Self()
        settings[.key] = Light(power: 1.22, size: 0.88, angle: -23, tint: 0.7)
        settings[.fill] = Light(power: 0.34, size: 1.12, angle: 30, tint: -0.32)
        settings[.rim] = Light(power: 1.36, size: 0.76, angle: 18, tint: -0.78)
        settings[.front].power = 0.34
        settings.exposure = -0.12
        return settings
    }

    func makeLighting() throws -> PathTracer.Lighting {
        let panels: [(center: SIMD3<Float>, aim: SIMD3<Float>, size: SIMD2<Float>, radiance: SIMD3<Float>)] = [
            (SIMD3(-3.4, 5.1, 4.3), SIMD3(0, 1, 0), SIMD2(5.1, 2.4), SIMD3(11.5, 9.6, 7.6)),
            (SIMD3(3.9, 3, 1.6), SIMD3(0.1, 1.15, 0), SIMD2(1.6, 4.1), SIMD3(5.9, 7.4, 9.7)),
            (SIMD3(0.7, 4, -3.2), SIMD3(0, 1, 0), SIMD2(4.6, 1.55), SIMD3(9.6, 10.7, 13.4)),
            (SIMD3(-0.3, 1.4, 5.8), SIMD3(-0.3, 1.2, 0), SIMD2(1.9, 0.68), SIMD3(3.9, 3.6, 3.1)),
        ]
        let emitters = try zip(panels, lights).compactMap { panel, edit -> PathTracer.AreaLight? in
            guard edit.power > 0 else { return nil }
            let orientation = simd_quatf(angle: Float(edit.angle) * .pi / 180, axis: SIMD3(0, 1, 0))
            let center = panel.aim + orientation.act(panel.center - panel.aim)
            let extent = panel.size * Float(edit.size)
            let warm = exp2f(Float(edit.tint) * 0.85)
            let rgb = panel.radiance * SIMD3(warm, 1, 1 / warm)
            let initialArea = panel.size.x * panel.size.y
            let power = rgb * (Float.pi * initialArea * Float(edit.power))
            return try PathTracer.AreaLight(
                center: center, aim: panel.aim, size: extent, radiantPower: power, cameraVisible: false
            )
        }
        return try PathTracer.Lighting(lights: emitters, environment: SIMD3(0.014, 0.017, 0.022))
    }
}

struct LabPathLightingHost: View {
    static let argument = "--path-light-controls"

    static var isRequested: Bool {
        ProcessInfo.processInfo.arguments.contains(argument)
            && !ProcessInfo.processInfo.arguments.contains("--path-light-room")
            && !ProcessInfo.processInfo.arguments.contains(LabPathTracingFixture.icebergArgument)
            && !ProcessInfo.processInfo.arguments.contains(LabPathTracingFixture.porscheArgument)
    }

    @State private var lighting = LabPathLightSettings()
    @State private var selected = LabPathLightSettings.Role.key

    var body: some View {
        VStack(spacing: 0) {
            TriangleHost(pathLighting: lighting)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            LabPathLightInspector(lighting: $lighting, selected: $selected)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
        }
    }
}

private struct LabPathLightInspector: View {
    @Binding var lighting: LabPathLightSettings
    @Binding var selected: LabPathLightSettings.Role

    var body: some View {
        let light = Binding(get: { lighting[selected] }, set: { lighting[selected] = $0 })
        VStack(spacing: 8) {
            HStack {
                Picker("Emitter", selection: $selected) {
                    ForEach(LabPathLightSettings.Role.allCases) { role in
                        Text(role.title).tag(role)
                    }
                }
                .pickerStyle(.segmented)
                Button("Reset") { lighting = LabPathLightSettings() }
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 20)], spacing: 5) {
                LabPathLightSlider(label: "Power", value: light.power, limits: 0...2.5, step: 0.01)
                LabPathLightSlider(label: "Size", value: light.size, limits: 0.35...2.2, step: 0.01)
                LabPathLightSlider(label: "Angle", value: light.angle, limits: -100...100, step: 1)
                LabPathLightSlider(label: "Tint", value: light.tint, limits: -1...1, step: 0.01)
                LabPathLightSlider(label: "Exposure", value: $lighting.exposure, limits: -2...2, step: 0.02)
            }
        }
    }
}

private struct LabPathLightSlider: View {
    let label: String
    @Binding var value: Double
    let limits: ClosedRange<Double>
    let step: Double

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(minWidth: 62, alignment: .leading)
                .accessibilityHidden(true)
            Slider(value: $value, in: limits, step: step) { Text(label) }
                .labelsHidden()
            Text(value, format: .number.precision(.fractionLength(step == 1 ? 0 : 2)))
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(minWidth: 42, alignment: .trailing)
                .accessibilityHidden(true)
        }
        .font(.callout)
    }
}
