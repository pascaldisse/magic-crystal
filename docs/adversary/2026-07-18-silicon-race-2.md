# ADVERSARY REPORT — SILICON RACE II · 2026-07-18

VERDICT: HOLDS

## CONCORDANCE

- Banned-word law upheld → probe source imports Foundation + Metal only; builder hunt created no prohibited-source artifact; only Metal-native runtime objects/tensors/encoder exercised. [source: tools/silicon-race-2/metal4-door.swift; proof/2026-07-18-silicon-race-2-metal4-door.txt]
- Offline law untouched → no service, network, or remote artifact used; package-builder, compiler, and probe were local machine tools only. [source: tools/silicon-race-2/probe-package-wall.sh; proof/2026-07-18-silicon-race-2-package-wall.txt]
- Builder identity/contract independently checked → `xcrun --find` resolves `metal-package-builder`; help requires a source-package input plus `-ml`, then writes `.mtlpackage`. [source: local command output, 2026-07-18]
- Apple workflow agrees → the builder converts its prerequisite source package into an `.mtlpackage`; no Metal/plain-source network authoring surface was found. [source: https://developer.apple.com/videos/play/wwdc2025/262/; local toolchain search, 2026-07-18]
- Runtime boundary agrees → pipeline state precedes network dispatch; the headers document dispatch/heap consumption, not package construction. [source: /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers/MTL4MachineLearningCommandEncoder.h; /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers/MTL4MachineLearningPipeline.h]
- Numbers cited → current empty-encoder control: `9.875 µs` encode, `0.035583 ms` timeline, `0.284667 ms` commit/wait; explicitly not network timing. [source: proof/2026-07-18-silicon-race-2-metal4-door.txt]

## REPRODUCED WALL

- Prior compiled artifact conversion → missing `Manifest.json`; builder tool exit `0`, output absent, probe exit `1`. [source: proof/2026-07-18-silicon-race-2-package-wall.txt]
- Verdict → **BUILDER FOUND; PACKAGE NOT PRODUCIBLE LAWFULLY.** The converter is installed, but its prerequisite source package cannot be authored in this lane without breaking the banned-word law. [source: local builder help, 2026-07-18; https://developer.apple.com/videos/play/wwdc2025/262/]
- Consequence → tiny `64–4096` package calls, `dispatchNetwork`, scheduler locus, and power residency stay **UNVERIFIED**; no timing or silicon claim inferred from the empty control. [source: proof/2026-07-18-silicon-race-2-metal4-door.txt]

## GAPS

- Lawful source package absent → dispatch/path latency and locus **UNVERIFIED**.
- Privileged power counters unavailable → residency **UNVERIFIED**. [source: docs/perf/2026-07-18-silicon-race-2.md]
