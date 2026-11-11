// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "Frostfield",
    platforms: [
        .iOS("27.0"),
        .macOS("27.0"),
        .visionOS("27.0"),
    ],
    products: [
        .library(name: "Frostfield", targets: ["Frostfield"]),
        .library(name: "FrostfieldSwiftUI", targets: ["FrostfieldSwiftUI"]),
    ],
    targets: [
        .target(
            name: "FrostfieldDraco",
            cxxSettings: [
                .headerSearchPath("Vendor"),
                // Keep these private C++ symbols separate from a client's own Draco library.
                .define("draco", to: "frostfield_draco")
            ]
        ),
        .target(
            name: "FrostfieldBasis",
            cSettings: [.define("NDEBUG")],
            cxxSettings: [
                .define("NDEBUG"),
                .define("basist", to: "frostfield_basist"),
                .define("basisu", to: "frostfield_basisu"),
                .define("astc_helpers", to: "frostfield_basis_astc_helpers"),
                .define("BASISD_SUPPORT_KTX2_ZSTD", to: "1"),
            ]
        ),
        .target(
            name: "FrostfieldXAtlas",
            cxxSettings: [
                .define("NDEBUG"),
                .define("xatlas", to: "frostfield_xatlas"),
                .define("XA_MULTITHREADED", to: "0"),
                .define("FROSTFIELD_XATLAS_SINGLE_PAGE"),
            ]
        ),
        .target(name: "FrostfieldEXRPIZ"),
        .target(
            name: "FrostfieldJPEGXL",
            cxxSettings: [
                .headerSearchPath("Vendor/JPEGXL"),
                .headerSearchPath("Vendor/JPEGXL/lib/include"),
                .headerSearchPath("Vendor/Highway"),
                .headerSearchPath("Vendor/SkCMS"),
                .define("NDEBUG"),
                .define("JXL_INTERNAL_LIBRARY_BUILD"),
                .define("JPEGXL_ENABLE_BOXES", to: "0"),
                .define("JPEGXL_ENABLE_TRANSCODE_JPEG", to: "0"),
                .define("JPEGXL_ENABLE_SKCMS", to: "1"),
                .define("HWY_COMPILE_ONLY_STATIC"),
                .define("SKCMS_DISABLE_HSW"),
                .define("SKCMS_DISABLE_SKX"),
                .define("jxl", to: "frostfield_jxl"),
                .define("hwy", to: "frostfield_jxl_hwy"),
                .define("skcms_private", to: "frostfield_jxl_skcms_private"),
            ]
        ),
        .target(
            name: "Frostfield",
            dependencies: ["FrostfieldDraco", "FrostfieldBasis", "FrostfieldXAtlas", "FrostfieldEXRPIZ", "FrostfieldJPEGXL"],
            resources: [
                .process("Triangle.metal"),
                .process("Scene.metal"),
                .process("RasterMotionVectors.metal"),
                .process("CurveFlow.metal"),
                .process("MeshIndirectDraws.metal"),
                .process("Particles.metal"),
                .process("ColorGrade.metal"),
                .process("TextureDebugPass.metal"),
                .process("ScalarVolumeRenderer.metal"),
                .process("TextureDecompression.metal"),
                .process("Shadows.metal"),
                .process("PlanarProjectedShadow.metal"),
                .process("ContactShadows.metal"),
                .process("VarianceShadowMaps.metal"),
                .process("PhysicallyBased.metal"),
                .process("ClassicMaterials.metal"),
                .process("ScreenEffects.metal"),
                .process("HDRDisplay.metal"),
                .process("AntiAliasing.metal"),
                .process("SMAA.metal"),
                .process("EnvironmentLighting.metal"),
                .process("DepthOfField.metal"),
                .process("EnvironmentBackground.metal"),
                .process("AmbientOcclusion.metal"),
                .process("PathTracer.metal"),
                .process("PathTraceDenoiser.metal"),
                .process("ShaderCompute.metal"),
                .process("PostProcessSuite.metal"),
                .process("PostProcessOptics.metal"),
                .process("PostProcessMask.metal"),
                .process("PostProcessTexture.metal"),
                .process("Atmosphere.metal"),
                .process("Primitives.metal"),
                .process("ScreenSpaceReflections.metal"),
                .process("ScreenSpaceShadows.metal"),
                .process("ScreenSpaceGlobalIllumination.metal"),
                .process("SurfaceAttributes.metal"),
                .copy("FrostfieldScreenSpace.h"),
                .copy("FrostfieldShadowSampling.h"),
                .copy("FrostfieldLightProbe.h"),
                .copy("FrostfieldLogDepth.h"),
                .copy("FrostfieldDithering.h"),
                .copy("FrostfieldSoftParticle.h"),
                .copy("FrostfieldRayTriangle.h"),
                .copy("FrostfieldMeshBVH.h"),
                .copy("PhysicallyBasedShading.h"),
                .copy("RectangleLTC"),
                .copy("PathTracingShading.h"),
                .copy("PathBlueNoiseRanks.bin"),
                .copy("FrostfieldMaterialExtensions.h"),
                .copy("ClassicMaterialShading.h"),
                .copy("PrimitiveShading.h"),
            ]
        ),
        .target(name: "FrostfieldSwiftUI", dependencies: ["Frostfield"]),
        .testTarget(name: "FrostfieldTests", dependencies: ["Frostfield"]),
        .testTarget(name: "FrostfieldSwiftUITests", dependencies: ["Frostfield", "FrostfieldSwiftUI"]),
    ],
    swiftLanguageModes: [.v6],
    cxxLanguageStandard: .cxx17
)
