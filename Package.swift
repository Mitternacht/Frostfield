// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Frostfield",
    platforms: [
        // iOS/iPadOS 26 is the first presentation target. macOS 26 is a
        // core-compilation target only; it does not imply a macOS adapter.
        .iOS(.v26),
        .macOS(.v26),
    ],
    products: [
        .library(name: "Frostfield", targets: ["Frostfield"]),
    ],
    targets: [
        // C declarations of CPU/GPU records, imported by Swift and included by
        // the Metal shaders so both compilers see one byte layout.
        .target(name: "FrostfieldShaderTypes"),
        // Core: Metal and simd only. Must not import UIKit or SwiftUI.
        // Xcode compiles Shaders/*.metal into the target's resource bundle.
        .target(name: "Frostfield", dependencies: ["FrostfieldShaderTypes"]),
        .testTarget(name: "FrostfieldTests", dependencies: ["Frostfield", "FrostfieldShaderTypes"]),
    ],
    swiftLanguageModes: [.v6]
)
