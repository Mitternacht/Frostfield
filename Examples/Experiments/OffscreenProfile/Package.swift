// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "FrostfieldOffscreenProfile",
    platforms: [.macOS("27.0")],
    dependencies: [.package(path: "../../..")],
    targets: [
        .executableTarget(
            name: "FrostfieldOffscreenProfile",
            dependencies: [.product(name: "Frostfield", package: "Frostfield")]
        )
    ],
    swiftLanguageModes: [.v6]
)
