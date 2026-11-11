# Mac offscreen profile

The standalone [profile executable](../../Examples/Experiments/OffscreenProfile)
loads the local Frostfield package through its public product. It renders the
combined effects fixture at 900 × 608 into private textures, with a Lambert
cube, 512 particles using seed 42, and a color grade at strength 0.72.
It paces one command buffer at a time at a requested 60 Hz on a CPU thread.

From the repository root, build the probe with Xcode 27 and its Metal toolchain.

```sh
profile_build="$(mktemp -d)"
cd Examples/Experiments/OffscreenProfile
xcodebuild -scheme FrostfieldOffscreenProfile -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath "$profile_build" build
"$profile_build/Build/Products/Release/FrostfieldOffscreenProfile"
```

The default run warms up for 30 seconds and observes for 600 seconds.
Add `--quick` for a one-second warmup and four-second sample.
CPU time spans encoding through commit and excludes waiting for completion.
GPU time comes from the completed command buffer's start and end timestamps.
A cadence overrun means CPU and GPU work plus wakeup delay crossed the next
scheduled tick. Process resident memory and thermal state are sampled once per
60 measured frames.

## Mac Studio, Apple M4 Max

2026-11-09. macOS 27.2 (26B123), Xcode 27.2 (27B52), Release, conventional
Metal. The Mac had 36 GB of unified memory and ran on AC power. Its console
was locked. Local builds and brief tests ran during the sustained observation.

The four-second smoke measured 240 CPU and GPU samples with zero cadence
overruns. CPU p50/p95/p99 was 0.212/0.746/1.572 ms and GPU command duration
was 0.135/0.699/0.871 ms. The reported process resident memory stayed at
16,384,000 bytes in that sample.

The sustained run collected 35,991 completed command buffers after warmup.

| Measurement | Result |
| --- | --- |
| CPU encode and commit p50 / p95 / p99 | 0.249 / 0.387 / 0.679 ms |
| GPU command buffer p50 / p95 / p99 | 0.061 / 0.377 / 0.739 ms |
| CPU-paced tick overruns | 18 |
| Maximum submitted buffers in flight | 1 |
| Sampled process resident bytes at start / peak / end | 15,908,864 / 15,958,016 / 14,319,616 |
| Sampled thermal state | Nominal |

This is a physical Mac GPU workload with an offscreen destination.
Presentation intervals, interaction latency, energy, repeated runs, and
device-specific thresholds require separate measurements.
[Windowed metrics recipe](render-metrics.md)
