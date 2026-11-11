# Lab scene benchmarks on one Mac

These are offscreen measurements from the Release FrostfieldLab app on an Apple
M4 Max Mac Studio with macOS 27.2 (26B123) and Xcode 27.2 (27B52). Each run
renders the scene's normal frame with one command buffer in flight, waits for
completion, skips a warm-up, and records CPU encode time and GPU time from
command-buffer timestamps. They exclude presentation, display timing, and
input latency. Metal API Validation was off for these timings.

| Scene and mode | Size | GPU p50 | GPU p95 | GPU p99 | CPU encode p50 | Device allocation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shader Lab, raster with deformation and post chain | 1920 × 1080 | 0.74 ms | 1.14 ms | 1.33 ms | 0.09 ms | 109 MB |
| Light Lab, raster with sun, spot, point, and rectangle shadows | 1920 × 1080 | 1.29 ms | 1.82 ms | 2.01 ms | 0.10 ms | 112 MB |
| Animation Lab, skinned dolphin and mixer | 1920 × 1080 | 0.24 ms | 0.41 ms | 0.50 ms | 0.12 ms | 52 MB |
| Effects Lab, 65,536 graph-simulated particles, plasma graph, bloom and post graphs | 1920 × 1080 | 0.83 ms | 1.30 ms | 1.45 ms | 0.06 ms | 96 MB |
| Shader Lab, path traced, 1 sample per pixel per frame | 1280 × 720 | 10.85 ms | 11.47 ms | 11.76 ms | 0.10 ms | 111 MB |
| Light Lab, path traced with HDR sky and four lights, 1 sample per pixel per frame | 1280 × 720 | 38.36 ms | 39.98 ms | 40.72 ms | 0.07 ms | 131 MB |

Raw results: [Shader raster](data/lab-bench-shader-raster.json) ·
[Shader path](data/lab-bench-shader-path.json) ·
[Light raster](data/lab-bench-light-raster.json) ·
[Light path](data/lab-bench-light-path.json) ·
[Animation raster](data/lab-bench-animation-raster.json) ·
[Effects raster](data/lab-bench-effects-raster.json)

Each scene ran once (300 raster frames or 120 path frames after warm-up), so
these numbers show the order of cost on this Mac, not a stable distribution.
They are not iPhone, iPad, or Vision Pro measurements, and no comparison with
Three.js was made. A fair comparison needs the same hardware, scene,
resolution, and quality settings in both engines.

```sh
FrostfieldLab.app/Contents/MacOS/FrostfieldLab --benchmark-shader-lab /tmp/shader.json 300 1920 1080
FrostfieldLab.app/Contents/MacOS/FrostfieldLab --benchmark-light-lab /tmp/light-path.json 120 1280 720 --set scene.path=1 --set scene.samples=100000
```
