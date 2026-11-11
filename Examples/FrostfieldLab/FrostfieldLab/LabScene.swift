import Foundation
import Frostfield
import Metal
import Observation
import QuartzCore
import SwiftUI

/// A Lab scene that draws into the shared native Metal host and publishes its
/// editable values through `LabSceneControls`.
@MainActor
protocol LabSceneFixture: AnyObject {
    /// Redraws every display refresh instead of on demand.
    var animates: Bool { get }
    /// Asks the host to draw again after the frame completes, for progressive rendering.
    var needsMoreSamples: Bool { get }
    /// Caps the drawable's longest side in pixels, or nil for full resolution.
    var previewLimit: CGFloat? { get }
    func draw(drawable: any CAMetalDrawable, completed: @escaping @Sendable (String?) -> Void) throws
    func orbit(horizontal: Float, vertical: Float)
    func zoom(by scale: Float)
    /// Applies the current control values. Throwing reports the message in the inspector.
    func applyControls() throws
    #if os(macOS)
    func capture(to url: URL, width: Int, height: Int) throws
    /// Encodes one frame into an offscreen BGRA8 sRGB texture for benchmarks.
    func encodeOffscreen(into destination: any MTLTexture, commandBuffer: any MTLCommandBuffer) throws
    var commandQueue: any MTLCommandQueue { get }
    /// Scene-specific counts from the latest benchmark frame, stored under "details".
    var benchmarkDetails: [String: Any] { get }
    #endif
}

#if os(macOS)
extension LabSceneFixture {
    var benchmarkDetails: [String: Any] { [:] }
}
#endif

/// One editable value. Slider values and toggles share a numeric store.
struct LabControl: Identifiable {
    enum Kind {
        case slider(ClosedRange<Double>, step: Double)
        case toggle
        case choice([String])
    }

    let id: String
    let title: String
    let kind: Kind
    let initial: Double
}

struct LabControlGroup: Identifiable {
    let id: String
    let title: String
    let controls: [LabControl]
}

/// Values a Lab scene's inspector edits. The host forwards each change to the fixture.
@Observable
final class LabSceneControls {
    private(set) var groups: [LabControlGroup] = []
    private(set) var values: [String: Double] = [:]
    private(set) var revision = 0
    var selectedGroup: String?
    /// Inspectable source per control group, such as a graph's generated Metal.
    var sources: [String: String] = [:]
    var status: String?

    func register(_ groups: [LabControlGroup]) {
        self.groups = groups
        for control in groups.flatMap(\.controls) where values[control.id] == nil {
            values[control.id] = control.initial
        }
        if selectedGroup == nil { selectedGroup = groups.first?.id }
    }

    func value(_ id: String) -> Double { values[id] ?? 0 }
    func isOn(_ id: String) -> Bool { value(id) > 0.5 }

    func set(_ id: String, _ value: Double) {
        guard values[id] != value else { return }
        values[id] = value
        revision += 1
    }

    func reset() {
        for control in groups.flatMap(\.controls) { values[control.id] = control.initial }
        revision += 1
    }
}

enum LabScenes {
    enum Kind: String, CaseIterable {
        case shaderLab = "--shader-lab"
        case lightLab = "--light-lab"
        case animationLab = "--animation-lab"
        case effectsLab = "--effects-lab"
        case scaleLab = "--scale-lab"

        var title: String {
            switch self {
            case .shaderLab: "Shader graphs on generated objects"
            case .lightLab: "Editable light rig in a studio"
            case .animationLab: "Blended dolphin animation"
            case .effectsLab: "Plasma core and particle vortex"
            case .scaleLab: "Instanced field and material extensions"
            }
        }

        var captureArgument: String { "--capture" + rawValue.dropFirst() }
        var benchmarkArgument: String { "--benchmark" + rawValue.dropFirst() }
    }

    static func requested(_ arguments: [String] = ProcessInfo.processInfo.arguments) -> Kind? {
        Kind.allCases.first { arguments.contains($0.rawValue) }
    }

    static func make(
        _ kind: Kind, device: any MTLDevice, destinationFormat: MTLPixelFormat, controls: LabSceneControls
    ) throws -> any LabSceneFixture {
        switch kind {
        case .shaderLab: try LabShaderLab(device: device, destinationFormat: destinationFormat, controls: controls)
        case .lightLab: try LabLightLab(device: device, destinationFormat: destinationFormat, controls: controls)
        case .animationLab: try LabAnimationLab(device: device, destinationFormat: destinationFormat, controls: controls)
        case .effectsLab: try LabEffectsLab(device: device, destinationFormat: destinationFormat, controls: controls)
        case .scaleLab: try LabScaleLab(device: device, destinationFormat: destinationFormat, controls: controls)
        }
    }

    #if os(macOS)
    /// Handles `--benchmark-<scene> OUTPUT.json [FRAMES] [WIDTH HEIGHT] [--set id=value ...]`.
    /// Renders offscreen with one command buffer in flight and records each frame's
    /// CPU encode time, GPU time from command-buffer timestamps, and device allocation.
    static func benchmarkIfRequested(_ arguments: [String]) throws -> Bool {
        guard let kind = Kind.allCases.first(where: { arguments.contains($0.benchmarkArgument) }),
              let index = arguments.firstIndex(of: kind.benchmarkArgument) else { return false }
        let fields = Array(arguments.dropFirst(index + 1))
        guard let path = fields.first, let device = MTLCreateSystemDefaultDevice() else {
            throw CocoaError(.fileWriteUnknown, userInfo: [
                NSLocalizedDescriptionKey: "Use \(kind.benchmarkArgument) OUTPUT.json [FRAMES] [WIDTH HEIGHT] [--set ID=VALUE]."
            ])
        }
        let numbers = fields.dropFirst().prefix(3).compactMap(Int.init)
        let frames = numbers.first ?? 240
        let width = numbers.count == 3 ? numbers[1] : 1920
        let height = numbers.count == 3 ? numbers[2] : 1080
        let controls = LabSceneControls()
        let fixture = try make(kind, device: device, destinationFormat: .bgra8Unorm_srgb, controls: controls)
        for (offset, field) in fields.enumerated() where field == "--set" && offset + 1 < fields.count {
            let pair = fields[offset + 1].split(separator: "=", maxSplits: 1)
            guard pair.count == 2, let value = Double(pair[1]) else { continue }
            controls.set(String(pair[0]), value)
        }
        try fixture.applyControls()
        let target = try LabImageCapture.target(device: device, width: width, height: height)
        var cpu: [Double] = []
        var gpu: [Double] = []
        let warmup = min(30, frames / 4)
        for frame in 0..<(frames + warmup) {
            guard let buffer = fixture.commandQueue.makeCommandBuffer() else { throw MeshRenderer.Failure.resource("benchmark frame") }
            let start = CACurrentMediaTime()
            try fixture.encodeOffscreen(into: target, commandBuffer: buffer)
            let encoded = CACurrentMediaTime()
            buffer.commit()
            buffer.waitUntilCompleted()
            guard buffer.status == .completed else { throw buffer.error ?? MeshRenderer.Failure.resource("benchmark frame \(frame)") }
            if frame >= warmup {
                cpu.append((encoded - start) * 1000)
                gpu.append((buffer.gpuEndTime - buffer.gpuStartTime) * 1000)
            }
        }
        func percentile(_ values: [Double], _ p: Double) -> Double {
            let sorted = values.sorted()
            return sorted[min(sorted.count - 1, Int((Double(sorted.count - 1) * p).rounded()))]
        }
        var report: [String: Any] = [
            "scene": kind.rawValue, "frames": frames, "warmupFrames": warmup, "width": width, "height": height,
            "controls": controls.values, "device": device.name,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "cpuEncodeMs": ["p50": percentile(cpu, 0.5), "p95": percentile(cpu, 0.95), "p99": percentile(cpu, 0.99)],
            "gpuMs": ["p50": percentile(gpu, 0.5), "p95": percentile(gpu, 0.95), "p99": percentile(gpu, 0.99), "max": gpu.max() ?? 0],
            "deviceAllocatedMB": Double(device.currentAllocatedSize) / 1_048_576,
            "note": "Offscreen, one command buffer in flight, waits each frame; excludes presentation and display timing.",
        ]
        let details = fixture.benchmarkDetails
        if !details.isEmpty { report["details"] = details }
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
        return true
    }

    /// Handles `--capture-<scene> OUTPUT.png [WIDTH HEIGHT] [--set id=value ...]`.
    static func captureIfRequested(_ arguments: [String]) throws -> Bool {
        guard let kind = Kind.allCases.first(where: { arguments.contains($0.captureArgument) }),
              let index = arguments.firstIndex(of: kind.captureArgument) else { return false }
        let fields = Array(arguments.dropFirst(index + 1))
        guard let path = fields.first, let device = MTLCreateSystemDefaultDevice() else {
            throw CocoaError(.fileWriteUnknown, userInfo: [
                NSLocalizedDescriptionKey: "Use \(kind.captureArgument) OUTPUT.png [WIDTH HEIGHT] [--set ID=VALUE]."
            ])
        }
        let numbers = fields.dropFirst().prefix(2).compactMap(Int.init)
        let width = numbers.count == 2 ? numbers[0] : 1440
        let height = numbers.count == 2 ? numbers[1] : 960
        let controls = LabSceneControls()
        let fixture = try make(kind, device: device, destinationFormat: .bgra8Unorm_srgb, controls: controls)
        for (offset, field) in fields.enumerated() where field == "--set" && offset + 1 < fields.count {
            let pair = fields[offset + 1].split(separator: "=", maxSplits: 1)
            guard pair.count == 2, let value = Double(pair[1]) else { continue }
            controls.set(String(pair[0]), value)
        }
        try fixture.applyControls()
        try fixture.capture(to: URL(fileURLWithPath: path), width: width, height: height)
        return true
    }
    #endif
}

/// Hosts a Lab scene beside a native inspector generated from its controls.
struct LabSceneHost: View {
    static var isRequested: Bool { LabScenes.requested() != nil }

    @State private var controls = LabSceneControls()
    @State private var showsSource = false

    var body: some View {
        VStack(spacing: 0) {
            TriangleHost(sceneControls: controls, sceneRevision: controls.revision)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider()
            LabSceneInspector(controls: controls, showsSource: $showsSource)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
        }
        .sheet(isPresented: $showsSource) {
            LabSourceSheet(source: controls.sources[controls.selectedGroup ?? ""] ?? "")
        }
    }
}

private struct LabSceneInspector: View {
    let controls: LabSceneControls
    @Binding var showsSource: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                LabGroupPicker(controls: controls)
                if !controls.sources.isEmpty {
                    Button("Metal", systemImage: "chevron.left.forwardslash.chevron.right") { showsSource = true }
                        .disabled(controls.sources[controls.selectedGroup ?? ""] == nil)
                }
                Spacer(minLength: 0)
                Button("Reset") { controls.reset() }
            }
            if let group = controls.groups.first(where: { $0.id == controls.selectedGroup }) {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 20)], spacing: 5) {
                    ForEach(group.controls) { control in
                        LabControlRow(control: control, controls: controls)
                    }
                }
            }
            if let status = controls.status {
                Text(status)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
    }
}

/// Segments while the group titles fit, otherwise a menu.
private struct LabGroupPicker: View {
    let controls: LabSceneControls

    var body: some View {
        let selection = Binding(get: { controls.selectedGroup ?? "" }, set: { controls.selectedGroup = $0 })
        ViewThatFits(in: .horizontal) {
            picker(selection).pickerStyle(.segmented).fixedSize()
            picker(selection).pickerStyle(.menu)
        }
    }

    private func picker(_ selection: Binding<String>) -> some View {
        Picker("Group", selection: selection) {
            ForEach(controls.groups) { group in
                Text(group.title).tag(group.id)
            }
        }
        .labelsHidden()
    }
}

private struct LabControlRow: View {
    let control: LabControl
    let controls: LabSceneControls

    var body: some View {
        let binding = Binding(get: { controls.value(control.id) }, set: { controls.set(control.id, $0) })
        switch control.kind {
        case .slider(let limits, let step):
            HStack(spacing: 8) {
                Text(control.title)
                    .frame(minWidth: 70, alignment: .leading)
                    .accessibilityHidden(true)
                Slider(value: binding, in: limits, step: step) { Text(control.title) }
                    .labelsHidden()
                Text(binding.wrappedValue, format: .number.precision(.fractionLength(step >= 1 ? 0 : 2)))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 42, alignment: .trailing)
                    .accessibilityHidden(true)
            }
        case .toggle:
            Toggle(control.title, isOn: Binding(get: { binding.wrappedValue > 0.5 }, set: { binding.wrappedValue = $0 ? 1 : 0 }))
        case .choice(let options):
            Picker(control.title, selection: Binding(get: { Int(binding.wrappedValue) }, set: { binding.wrappedValue = Double($0) })) {
                ForEach(Array(options.enumerated()), id: \.element) { index, option in
                    Text(option).tag(index)
                }
            }
        }
    }
}

private struct LabSourceSheet: View {
    let source: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView([.vertical, .horizontal]) {
                Text(verbatim: source)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .padding()
            }
            .navigationTitle("Generated Metal")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .frame(minWidth: 520, minHeight: 420)
    }
}
