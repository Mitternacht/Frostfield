# Metal command backend

Keep the conventional `MTLCommandQueue` for the current Frostfield renderer.
The Metal 4 probe matched its pixels on one Mac. A backend change needs a
measured benefit or a feature the conventional queue cannot supply.

## FF-04 comparison

On 2026-11-09, an Apple M4 Max running macOS 27.2 with Xcode 27.2 (27B52)
reported `supportsFamily(.metal4) == true` and created an `MTL4CommandQueue`.
The probe used Frostfield's [triangle shader](../../Sources/Frostfield/Triangle.metal)
from the working tree based on `e6a7144` (SHA-256
`f7d46efd444a290129a38fa29e9657fac2cced143aac8db267978d5bc27d3117`).
Both paths used the same `MTLRenderPipelineState`, 256 × 256 shared
`.bgra8Unorm_srgb` targets, clear color, and three-vertex draw. The only
comparison variable was the command and render-pass API.

The probe waited for conventional command completion and Metal 4 commit
feedback before reading either texture. Four sequential Metal 4 submissions
matched all 262,144 bytes of the conventional result in every run. The center
was BGRA `[234, 172, 172, 255]` and the clear corner was
`[52, 39, 29, 255]`, which confirms that both paths drew geometry. Metal API
and GPU validation were enabled and reported no failures. The Xcode MCP
snippet build could not write index data on the full external volume. The
isolated probe built with the selected Xcode's `xcrun swiftc` and ran from
temporary storage. Build and run the [standalone probe](../../Examples/Experiments/Metal4Triangle.swift)
from the repository root.

```sh
probe_dir="$(mktemp -d)"
xcrun swiftc -module-cache-path "$probe_dir/cache" \
  Examples/Experiments/Metal4Triangle.swift -o "$probe_dir/metal4-triangle"
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  "$probe_dir/metal4-triangle" Sources/Frostfield/Triangle.metal
```

## Lifetime and adoption

Any future Metal 4 backend needs an OS availability check where applicable,
`supportsFamily(.metal4)` on the selected device, and successful queue creation.
Keep the conventional queue available when any requirement is unmet. This run
covered one supported GPU.
[Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
and [Metal feature detection](https://developer.apple.com/documentation/metal/detecting-gpu-features-and-metal-software-versions).

Metal 4 stores encoded commands in a caller-owned command allocator. Reset an
allocator only after completion of every submission that uses its memory. The
probe reused one command buffer and allocator after each commit feedback
callback. A live renderer would need to keep allocator ownership with each
in-flight slot and hold the corresponding resources through GPU completion.
The probe explicitly committed a residency set containing the render target
and attached it to the Metal 4 queue. Residency and GPU access ordering remain
separate responsibilities.
[Allocator reset](https://developer.apple.com/documentation/metal/mtl4commandallocator/reset())
and [residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets).

Drawable presentation, multiple in-flight frames, resize and resource reuse
under load, other devices, and matched timing and memory traces remain open.
Revisit Metal 4 after the same scene and settings have a matched Instruments
trace on a target device.
