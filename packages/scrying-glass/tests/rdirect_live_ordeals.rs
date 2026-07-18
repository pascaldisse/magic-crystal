//! NEURAL-LIVE N0 GATE 1 — parity of the live-path net (MPSGraph batched GEMM
//! on the Metal device) vs the Rust CPU `Mlp::forward` reference, on the fixed
//! exported front pose (tools/metal4-probe/data). GPU ordeal: builds a real
//! MPSGraph forward, runs it on a live Metal command queue, reads back.
//!
//! Skips (does NOT fail) when no Metal device is present (CI/non-mac) — the
//! gate is meaningful only on the silicon it ships to.

#![cfg(target_os = "macos")]

use scrying_glass::rdirect_live::RdirectLive;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    // tests/ → package root → tools/metal4-probe/data
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.join("../../tools/metal4-probe/data")
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} not f32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn n0_gate1_live_net_matches_cpu_reference() {
    let dir = data_dir();
    let weights = std::fs::read(dir.join("rdirect-weights-v1.bin")).expect("weights blob");
    let features = read_f32(&dir.join("features.f32"));
    let expected = read_f32(&dir.join("expected.f32"));

    let live = match RdirectLive::from_system(&weights) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[n0-gate1] SKIP — no live Metal path: {e}");
            return;
        }
    };
    let in_f = live.in_features();
    let out_c = live.out_channels();
    let n = features.len() / in_f;
    assert_eq!(n * in_f, features.len(), "features not [N,{in_f}]");
    assert_eq!(expected.len(), n * out_c, "expected not [N,{out_c}]");

    // (a) live GPU forward.
    let got = live
        .forward_cpu_roundtrip(&features)
        .expect("live forward ran");
    assert_eq!(got.len(), expected.len(), "output length");

    // (b) independent CPU reference recomputed here (not just the committed
    //     artifact) — the true parity target.
    let cpu = live.cpu_ref();
    let mut cpu_ref = Vec::with_capacity(expected.len());
    for p in 0..n {
        let f = &features[p * in_f..(p + 1) * in_f];
        cpu_ref.extend_from_slice(&cpu.forward(f));
    }

    // Derived tolerance: f32 GEMM reassociation over the deepest accumulation
    // (INPUT_FEATURES..hidden_width across 6 layers). f32 eps ≈ 1.19e-7; a
    // ~1e-3 abs/rel envelope is orders above the spike's measured 1.6e-7 rel,
    // so a breach means a real wiring error, not float drift. Actuals printed.
    const TOL: f32 = 1.0e-3;
    let mut max_abs_vs_expected = 0f32;
    let mut max_rel_vs_expected = 0f32;
    let mut max_abs_vs_cpu = 0f32;
    for i in 0..expected.len() {
        let a = (got[i] - expected[i]).abs();
        let denom = expected[i].abs().max(1e-3);
        max_abs_vs_expected = max_abs_vs_expected.max(a);
        max_rel_vs_expected = max_rel_vs_expected.max(a / denom);
        max_abs_vs_cpu = max_abs_vs_cpu.max((got[i] - cpu_ref[i]).abs());
    }
    eprintln!(
        "[n0-gate1] N={n} px · live-vs-committed: abs {max_abs_vs_expected:.3e} rel \
         {max_rel_vs_expected:.3e} · live-vs-recomputed-CPU: abs {max_abs_vs_cpu:.3e}"
    );
    assert!(
        max_rel_vs_expected < TOL,
        "live net vs committed CPU ref rel error {max_rel_vs_expected:.3e} ≥ {TOL:.0e}"
    );
    assert!(
        max_abs_vs_cpu < TOL,
        "live net vs recomputed CPU ref abs error {max_abs_vs_cpu:.3e} ≥ {TOL:.0e}"
    );

    // Determinism: the same feed twice is byte-identical.
    let got2 = live
        .forward_cpu_roundtrip(&features)
        .expect("second forward");
    assert_eq!(got, got2, "live forward is not deterministic run-to-run");
}
