//! RITE VIII-3 — resolution/scale plumbing shared by the upscaler's training
//! harness (`examples/viii3_train.rs`), its ordeals (`tests/viii3_ordeals.rs`)
//! and its proof (`examples/viii3_upscale.rs`), factored into ONE source so
//! they can never silently drift (VIII-1's A5 lesson). NOT part of the net.
//! This file carries the `// BAN-SCOPED` marker so the VIII-0 grep-gate scans
//! it whole — it is plain resolution/pose setup with no cross-frame machinery
//! and passes trivially.
//!
//! POSES: reused VERBATIM from `denoiser_dataset` — the SAME five naruko
//! views, the SAME train/validation split (TRAIN = {front, wide, orbit_+20},
//! held-out VALIDATION = {orbit_-20, orbit_+40}; front is TRAIN, orbit poses
//! validate — established 07-17). The upscaler and denoiser therefore share a
//! dataset scope, keeping the VIII lineage honest and comparable.
//!
//! SCALE (hardcode law): [`UPSCALE_SCALE`] is the ONE magnitude — the integer
//! factor from internal (low, traced) resolution to native (target, upscaled)
//! resolution, a parameter with a default. Every resolution here is DERIVED
//! from a low resolution × the scale; NO target pixel count is frozen. The
//! Architect's ruling (07-17): render at 640×480 ("the resolution of God") →
//! neural upscale to the window. So 640×480 is the INTERNAL/low resolution
//! ([`PRODUCTION_LOW_WIDTH`]×[`PRODUCTION_LOW_HEIGHT`]) and the window-res
//! family is `low × UPSCALE_SCALE`. The training/ordeal dataset renders a
//! MUCH smaller low resolution (CPU per-pixel MLP budget) at the SAME scale.

// BAN-SCOPED

// Re-export the shared pose machinery so downstream code has ONE import
// surface and the split can never diverge from the denoiser's.
pub use crate::denoiser_dataset::{
    DATASET_REF_FRAMES, TRAIN_POSE_NAMES, VALIDATION_POSE_NAMES, law_poses, naruko_params,
};

/// The internal→native scale factor (the ONLY magnitude — parameter, default).
/// Default 2× (the window-res family is `internal × 2`); the derivation below
/// works for any factor ≥ 1 (scale 1 degenerates the resampler to identity,
/// the proposal's §VIII-3 ordeal).
pub const UPSCALE_SCALE: u32 = 2;

/// PRODUCTION internal (low, traced) resolution — the Architect's 640×480,
/// "the resolution of God". The production TARGET is this × [`UPSCALE_SCALE`]
/// (the window-res family), derived by [`target_dims`], never frozen.
pub const PRODUCTION_LOW_WIDTH: u32 = 640;
pub const PRODUCTION_LOW_HEIGHT: u32 = 480;

/// The DATASET internal (low, traced) resolution — small enough that CPU
/// per-pixel MLP training over 5 poses × converged references finishes in a
/// forge-time budget a builder actually runs (VIII-1 precedent: the dataset
/// resolution is a training convenience, not the production resolution). The
/// dataset TARGET is this × [`UPSCALE_SCALE`], derived below. Chosen so the
/// target lands at the denoiser dataset's 96×64 (48×2, 32×2) — the two rites
/// then validate at the same target resolution.
pub const DATASET_LOW_WIDTH: u32 = 48;
pub const DATASET_LOW_HEIGHT: u32 = 32;

/// Derive the native (target, upscaled) resolution from a low resolution and
/// the scale factor — the hardcode-law derivation used everywhere.
pub fn target_dims(low_w: u32, low_h: u32, scale: u32) -> (u32, u32) {
    (low_w * scale, low_h * scale)
}

/// The dataset's (low, target) resolution pair under [`UPSCALE_SCALE`].
pub fn dataset_dims() -> (u32, u32, u32, u32) {
    let (tw, th) = target_dims(DATASET_LOW_WIDTH, DATASET_LOW_HEIGHT, UPSCALE_SCALE);
    (DATASET_LOW_WIDTH, DATASET_LOW_HEIGHT, tw, th)
}

/// The production (low, target) resolution pair under [`UPSCALE_SCALE`] —
/// documented for the live-window seam (out of scope for wave (a)), so the
/// window lane inherits the derived family instead of re-freezing it.
pub fn production_dims() -> (u32, u32, u32, u32) {
    let (tw, th) = target_dims(PRODUCTION_LOW_WIDTH, PRODUCTION_LOW_HEIGHT, UPSCALE_SCALE);
    (PRODUCTION_LOW_WIDTH, PRODUCTION_LOW_HEIGHT, tw, th)
}
