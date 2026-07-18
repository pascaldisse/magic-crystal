//! THE REAL-IMAGE GATE ORDEAL (committed like rite5).
//!
//! Law (Architect, 2026-07-18): REAL OR BLACK. The app presents a neural frame
//! ONLY when the shipped weights carry a PASS stamp from `real_image_ordeal`.
//! These tests pin the gate that `main.rs`'s rig build calls
//! (`rdirect::verify_stamp`): no stamp / wrong sha / FAIL → the window is BLACK.

use std::path::Path;

use scrying_glass::rdirect::{blob_sha256, stamp_pass_text, stamp_path_for, verify_stamp};

fn crate_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// A weights file with NO stamp beside it must be DENIED (present black).
/// This is the v2/v1 case until an ordeal stamps a passing net.
#[test]
fn unstamped_weights_are_denied_black() {
    for rel in ["data/rdirect-weights-v1.bin", "data/rdirect-weights-v2.bin"] {
        let wpath = crate_path(rel);
        if !wpath.exists() {
            continue;
        }
        let bytes = std::fs::read(&wpath).unwrap();
        let stamp = stamp_path_for(&wpath);
        // If a stamp exists it MUST be a genuine PASS matching the sha; if it is
        // absent the gate must deny. Either way the gate must never accept an
        // unstamped/foreign blob.
        if !stamp.exists() {
            assert!(
                !verify_stamp(&bytes, &stamp),
                "{rel}: no stamp yet the gate accepted it — BLACK law violated"
            );
        }
    }
}

/// A genuine PASS stamp for the exact bytes is accepted; tamper/FAIL denied.
#[test]
fn gate_accepts_only_genuine_pass() {
    let dir = std::env::temp_dir().join(format!("gaia-gate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wpath = dir.join("weights.bin");
    let bytes = vec![7u8, 7, 7, 42, 1, 2, 3];
    std::fs::write(&wpath, &bytes).unwrap();
    let stamp = stamp_path_for(&wpath);

    assert!(!verify_stamp(&bytes, &stamp), "missing stamp accepted");

    std::fs::write(&stamp, stamp_pass_text(&bytes, &[("resid_still", 0.02)])).unwrap();
    assert!(verify_stamp(&bytes, &stamp), "genuine PASS rejected");

    // Different weights, same stamp → sha mismatch → deny.
    assert!(!verify_stamp(&[0u8, 0, 0], &stamp), "sha mismatch accepted");

    // FAIL status → deny even if sha matches.
    std::fs::write(
        &stamp,
        format!("GAIA-REAL-IMAGE-ORDEAL v1\nweights_sha256={}\nstatus=FAIL\n", blob_sha256(&bytes)),
    )
    .unwrap();
    assert!(!verify_stamp(&bytes, &stamp), "FAIL stamp accepted");

    let _ = std::fs::remove_dir_all(&dir);
}
