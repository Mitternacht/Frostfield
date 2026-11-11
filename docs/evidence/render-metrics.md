# FF-14 windowed render metrics

Use a Release build of FrostfieldLab on macOS. Close other lab instances and
keep the test window visible and key. Set `LAB_APP` to the absolute path of the
built `FrostfieldLab.app`. Each launch writes its records to standard error.

Run the combined effects fixture for a 60-frame warmup and 240 measured frames.

```sh
open -n -a "$LAB_APP" --stderr /tmp/frostfield-windowed-short.log \
  --args --lit --particles --grade --continuous --render-metrics
```

Quit the lab before the next launch. For a sustained run, warmup starts with the
first continuous submission. The fixture warms up for 30 seconds and observes
the next 600 seconds.

```sh
open -n -a "$LAB_APP" --stderr /tmp/frostfield-windowed-sustained.log \
  --args --lit --particles --grade --continuous --render-metrics-sustained
```

For an on-demand idle probe, quit the continuous instance and launch again.

```sh
open -n -a "$LAB_APP" --stderr /tmp/frostfield-windowed-idle.log \
  --args --render-metrics
```

The `FrostfieldMetrics` records identify the run and report CPU submission
time in the display-link callback through command-buffer commit. GPU buffer
time comes from the completed mesh command buffer. Presentation intervals
use the timestamps reported by the drawable on macOS. The report gives sample
counts and p50, p95, and p99 in milliseconds. Sustained percentiles use
0.25 ms buckets with an overflow count at 250 ms. The summary includes
dimensions, drawable size changes, suspensions, drawable size mismatches, and
first and last thermal state and process resident memory. The idle probe reports
new submissions and completions over two seconds.

## Observed on macOS

On 2026-11-09, the Release metrics fixture built on an Apple M4 Max Mac Studio
running macOS 27.2 (26B123) with Xcode 27.2 (27B52). This launch preceded the
lab's display-link migration. With the console locked, the app reported
`event=idle mode=inactive seconds=2 new_submissions=0 completions=0`.
The sustained windowed fixture did not begin its warmup while the console was
locked. Foreground submission, GPU and presentation distributions remain
unmeasured. The simulator captures document rendering on the updated
host. [Render lifecycle evidence](render-lifecycle.md)

The memory values are process snapshots. The drawable mismatch counter counts
updates whose texture dimensions differ from the current layout. The remaining
qualification includes display intervals on physical hardware, frame misses,
allocation growth and energy, and repeated interaction runs.
[Mac offscreen profile](offscreen-profile.md) · [Validation recipe](../validation.md)
