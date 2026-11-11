import SwiftUI

struct LabOceanHost: View {
    static let isRequested = ProcessInfo.processInfo.arguments.contains(LabOceanFixture.argument)

    @State private var isPaused = false

    var body: some View {
        TriangleHost(oceanPaused: isPaused)
            .overlay(alignment: .bottomTrailing) {
                let title: LocalizedStringKey = isPaused ? "Play" : "Pause"
                Button(title, systemImage: isPaused ? "play.fill" : "pause.fill") {
                    isPaused.toggle()
                }
                .buttonStyle(.borderedProminent)
                .accessibilityLabel(isPaused ? Text("Play Ocean") : Text("Pause Ocean"))
                .padding()
            }
    }
}
