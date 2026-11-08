// swift-tools-version: 6.2
import PackageDescription

// FF-05 spike: a clean Swift 6 consumer that renders the same transformed,
// indexed mesh with a custom MSL shader through Satin and through a minimal
// Metal renderer, exports both off-screen textures, and compares them. This
// package is isolated: Frostfield's manifest, library, and lab app never
// reference it, and it resolves its own Package.resolved.
let package = Package(
    name: "SatinComparison",
    platforms: [
        .macOS(.v26),
    ],
    dependencies: [
        // Fabric-Project fork (the Hi-Rez original is archived). Pinned to the
        // revision inspected on 2026-11-08: no build-tool plugins, no binary
        // targets, MIT license (copyright Hi-Rez).
        .package(url: "https://github.com/Fabric-Project/Satin.git", revision: "ff825e4945ca19277355eda69831b9b19aad1533"),
    ],
    targets: [
        .executableTarget(
            name: "SatinComparison",
            dependencies: [.product(name: "Satin", package: "Satin")],
            // Copied, not compiled: both renderers compile these sources at
            // run time (Satin through its own preprocessor), so the consumer
            // builds with plain `swift build`.
            resources: [.copy("Shaders")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
