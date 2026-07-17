//! RITE VIII-3 — THE UPSCALER: a tiny per-pixel MLP that reconstructs a
//! HIGHER-resolution frame from ONE low-resolution traced (+denoised) frame
//! plus THIS SAME frame's higher-resolution auxiliary buffers
//! (albedo/normal/depth). See docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md
//! §VIII-3. Wave (a): CPU reference; the GPU/WGSL port is a separate later
//! wave, exactly as VIII-1 → VIII-2.
//!
//! THE BAN (identical in spirit to VIII-1's, machine-checked the same way —
//! see `tests/viii3_ordeals.rs`): every public function here takes
//! CURRENT-FRAME buffers ONLY. Space is reconstructed WITHIN one honestly
//! traced frame; no frame is ever dreamt from another. There is no
//! cross-frame parameter or state of any kind anywhere in this module's
//! public API. This file carries the `// BAN-SCOPED` marker so the VIII-0
//! grep-gate's forward-proof scope mechanism scans it whole, and the VIII-3
//! ordeals additionally scan every `pub fn` signature here for forbidden
//! parameter-name substrings.
//!
//! THE ARCHITECTURE (the honest weld, Reading B of the proposal's OPEN 1):
//! the auxiliary geometry buffers (albedo/normal/depth) are cheap and are
//! produced at the FULL target resolution; only the expensive path-traced
//! RADIANCE is traced at the low internal resolution. The net upsamples the
//! expensive lighting GUIDED by the cheap high-resolution geometry — this is
//! what lets it beat a naive bilinear upsample: at a target-resolution edge,
//! bilinear blurs radiance across the geometric boundary, but the net, given
//! the SHARP high-resolution albedo/normal/depth of THIS target pixel, can
//! pull the radiance toward the correct side.
//!
//! Per TARGET pixel the network predicts a RESIDUAL over the bilinearly
//! upsampled radiance, worked in the SAME albedo-demodulated log-radiance
//! space VIII-1 established:
//!   base_dl  = log-demod(bilinear_upsample(low_radiance) at this pixel)
//!   residual = MLP(features)               (3 scalars)
//!   out_dl   = base_dl + residual
//!   out_rgb  = invert(out_dl)              (expm1, re-modulate by albedo)
//! The residual head is ZERO-INITIALIZED (last layer weights+biases = 0), so
//! an untrained net is EXACTLY naive bilinear (the log/demod roundtrip
//! cancels): training can only teach it to improve on bilinear, never to
//! start behind it.
//!
//! Features per target pixel (all CURRENT-FRAME): the 2×2 low-resolution
//! radiance TAPS around the sample point, albedo-demodulated + log-
//! transformed (4×3 = 12) + the subpixel fractional offset within the low
//! texel (2) + this target pixel's high-resolution albedo (3) + normal (3) +
//! log depth (1) = 21 scalars. All the same standard, current-frame-only
//! feature engineering VIII-1 documents (demodulation separates "how much
//! light landed" from "surface colour"; the log keeps HDR + depth on a
//! comparable scale without any per-frame normalization constant, which
//! would itself be forbidden cross-frame state).
//!
//! Pure Rust, f32 inference, FIXED index-ordered loops — byte-deterministic
//! by construction. No threading in the reference path.

// BAN-SCOPED

use crate::denoiser::sha256_hex;
use glam::Vec3;

/// Low-resolution radiance taps gathered per target pixel: a 2×2 bilinear
/// neighbourhood.
pub const RADIANCE_TAPS: usize = 4;
/// Per-target-pixel input feature count: 2×2 demod-log radiance taps (12) +
/// subpixel offset (2) + high-res albedo (3) + normal (3) + log depth (1).
pub const INPUT_FEATURES: usize = RADIANCE_TAPS * 3 + 2 + 3 + 3 + 1;
/// Per-pixel output channel count: the residual radiance (3).
pub const OUTPUT_CHANNELS: usize = 3;

/// Numerical floor added under albedo before dividing (mirrors VIII-1's
/// `ALBEDO_DEMOD_EPS`): avoids a divide by literal zero on dark-but-real
/// surfaces, small enough not to perturb a real material's demodulation.
pub const ALBEDO_DEMOD_EPS: f32 = 1e-3;

/// Below this squared albedo length a pixel is a NO-HIT (sky) primary-ray
/// miss (the AOV export writes exactly zero albedo there): demodulation is
/// skipped (divisor 1.0), never amplified by the tiny eps. Same rule and
/// rationale as VIII-1's denoiser.
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

fn demod_divisor(albedo: Vec3) -> Vec3 {
    if albedo.length_squared() > NO_HIT_ALBEDO_THRESHOLD_SQ {
        albedo + Vec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        Vec3::ONE
    }
}

fn log_demod(radiance: Vec3, divisor: Vec3) -> Vec3 {
    let d = radiance / divisor;
    Vec3::new(
        (d.x.max(0.0) + 1.0).ln(),
        (d.y.max(0.0) + 1.0).ln(),
        (d.z.max(0.0) + 1.0).ln(),
    )
}

fn undo_log_demod(dl: Vec3, divisor: Vec3) -> Vec3 {
    let expm1 = Vec3::new(dl.x.exp() - 1.0, dl.y.exp() - 1.0, dl.z.exp() - 1.0);
    Vec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0)) * divisor
}

/// The upscaler's scale/shape configuration. `scale` is the integer factor
/// from internal (low) resolution to native (target) resolution — the ONLY
/// magnitude here, a parameter with a default, never a frozen pixel count
/// (the caller derives low = target / scale, per the hardcode law). Hidden
/// shape defaults land in the Breda/NRC-class range VIII-1 uses, widened one
/// step (32 → 64) because the upscaler's input is larger (21 vs 10) and it
/// must resolve spatial edges the denoiser never had to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpscaleConfig {
    pub hidden_layers: usize,
    pub hidden_width: usize,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 4,
            hidden_width: 64,
        }
    }
}

impl UpscaleConfig {
    fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![INPUT_FEATURES];
        for _ in 0..self.hidden_layers {
            sizes.push(self.hidden_width);
        }
        sizes.push(OUTPUT_CHANNELS);
        sizes
    }
}

/// One dense layer: `out = w * in + b`, row-major `w` (rows = out_dim, cols =
/// in_dim).
#[derive(Debug, Clone)]
struct Layer {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Layer {
    fn zeros(in_dim: usize, out_dim: usize) -> Self {
        Self {
            in_dim,
            out_dim,
            w: vec![0.0; in_dim * out_dim],
            b: vec![0.0; out_dim],
        }
    }
}

/// Deterministic, dependency-free PRNG (SplitMix64) — weight INIT only
/// (forge-time), never the inference path. Same seed => same init, always.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_signed_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let unit = (bits as f32) / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }
}

/// The per-pixel upscaler MLP: feed-forward, ReLU hidden, LINEAR output (the
/// output is a signed log-radiance RESIDUAL over the bilinear base).
#[derive(Debug, Clone)]
pub struct Mlp {
    config: UpscaleConfig,
    layers: Vec<Layer>,
}

impl Mlp {
    /// He-initialized network EXCEPT the last (residual) layer, which is
    /// ZEROED: an untrained net produces zero residual, i.e. EXACTLY naive
    /// bilinear (the log/demod roundtrip cancels). Deterministic given
    /// `seed`; training itself is not promised bit-reproducible (proposal
    /// OPEN 4), but a fixed, bilinear-equivalent starting point is cheap
    /// honesty and makes "beats bilinear" a pure improvement claim.
    pub fn new_bilinear_start(config: UpscaleConfig, seed: u64) -> Self {
        let sizes = config.layer_sizes();
        let mut rng = SplitMix64::new(seed);
        let mut layers = Vec::with_capacity(sizes.len() - 1);
        let last = sizes.len() - 2; // index of the final layer in the windows walk
        for (li, pair) in sizes.windows(2).enumerate() {
            let (in_dim, out_dim) = (pair[0], pair[1]);
            let mut layer = Layer::zeros(in_dim, out_dim);
            if li != last {
                let scale = (2.0 / in_dim.max(1) as f32).sqrt();
                for w in layer.w.iter_mut() {
                    *w = rng.next_signed_unit() * scale;
                }
            }
            // last layer stays all zeros (weights AND biases) => zero residual.
            layers.push(layer);
        }
        Self { config, layers }
    }

    pub fn config(&self) -> UpscaleConfig {
        self.config
    }

    pub fn layer_dims(&self) -> Vec<(u32, u32)> {
        self.layers
            .iter()
            .map(|l| (l.in_dim as u32, l.out_dim as u32))
            .collect()
    }

    /// Flatten into one contiguous f32 array for a future compute-buffer
    /// upload (the GPU port will transcribe THESE numbers, never re-derive
    /// them): per layer in evaluation order, `in*out` row-major weights then
    /// `out` biases.
    pub fn flat_weights(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for l in &self.layers {
            out.extend_from_slice(&l.w);
            out.extend_from_slice(&l.b);
        }
        out
    }

    /// Forward pass for one pixel's feature vector. Fixed loop order — byte-
    /// deterministic. `needless_range_loop` silenced deliberately (the index
    /// order is the point, VIII-1 precedent).
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, input: &[f32]) -> [f32; OUTPUT_CHANNELS] {
        let mut activation = input.to_vec();
        for (li, layer) in self.layers.iter().enumerate() {
            let is_last = li == self.layers.len() - 1;
            let mut next = vec![0.0f32; layer.out_dim];
            for o in 0..layer.out_dim {
                let mut sum = layer.b[o];
                let row = o * layer.in_dim;
                for i in 0..layer.in_dim {
                    sum += layer.w[row + i] * activation[i];
                }
                next[o] = if is_last { sum } else { sum.max(0.0) };
            }
            activation = next;
        }
        [activation[0], activation[1], activation[2]]
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_train(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut activations = vec![input.to_vec()];
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        for (li, layer) in self.layers.iter().enumerate() {
            let is_last = li == self.layers.len() - 1;
            let earlier_act = activations.last().unwrap();
            let mut pre = vec![0.0f32; layer.out_dim];
            let mut act = vec![0.0f32; layer.out_dim];
            for o in 0..layer.out_dim {
                let mut sum = layer.b[o];
                let row = o * layer.in_dim;
                for i in 0..layer.in_dim {
                    sum += layer.w[row + i] * earlier_act[i];
                }
                pre[o] = sum;
                act[o] = if is_last { sum } else { sum.max(0.0) };
            }
            pre_activations.push(pre);
            activations.push(act);
        }
        (pre_activations, activations)
    }

    /// One backprop pass for a single pixel's (input, residual-target) pair
    /// under MSE loss on the residual output. Index-ordered throughout.
    #[allow(clippy::needless_range_loop)]
    fn backward(
        &self,
        input: &[f32],
        residual_target: &[f32; OUTPUT_CHANNELS],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (pre_activations, activations) = self.forward_train(input);
        let n_layers = self.layers.len();
        let mut w_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut b_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();

        let output = activations.last().unwrap();
        let mut delta: Vec<f32> = (0..OUTPUT_CHANNELS)
            .map(|c| 2.0 * (output[c] - residual_target[c]) / OUTPUT_CHANNELS as f32)
            .collect();

        for li in (0..n_layers).rev() {
            let layer = &self.layers[li];
            let is_last = li == n_layers - 1;
            if !is_last {
                for o in 0..layer.out_dim {
                    if pre_activations[li][o] <= 0.0 {
                        delta[o] = 0.0;
                    }
                }
            }
            let earlier_activation = &activations[li];
            for o in 0..layer.out_dim {
                b_grads[li][o] += delta[o];
                let row = o * layer.in_dim;
                for i in 0..layer.in_dim {
                    w_grads[li][row + i] += delta[o] * earlier_activation[i];
                }
            }
            if li > 0 {
                let mut earlier_delta = vec![0.0f32; layer.in_dim];
                for o in 0..layer.out_dim {
                    let row = o * layer.in_dim;
                    for i in 0..layer.in_dim {
                        earlier_delta[i] += layer.w[row + i] * delta[o];
                    }
                }
                delta = earlier_delta;
            }
        }
        (w_grads, b_grads)
    }
}

/// Minimal in-repo Adam optimizer (matches VIII-1's) — no new dependency,
/// index-ordered, forge-time only.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u32,
    m_w: Vec<Vec<f32>>,
    v_w: Vec<Vec<f32>>,
    m_b: Vec<Vec<f32>>,
    v_b: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(mlp: &Mlp, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m_w: mlp.layers.iter().map(|l| vec![0.0; l.w.len()]).collect(),
            v_w: mlp.layers.iter().map(|l| vec![0.0; l.w.len()]).collect(),
            m_b: mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect(),
            v_b: mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect(),
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self, mlp: &mut Mlp, w_grads: &[Vec<f32>], b_grads: &[Vec<f32>]) {
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);
        for li in 0..mlp.layers.len() {
            for i in 0..mlp.layers[li].w.len() {
                let g = w_grads[li][i];
                self.m_w[li][i] = self.beta1 * self.m_w[li][i] + (1.0 - self.beta1) * g;
                self.v_w[li][i] = self.beta2 * self.v_w[li][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m_w[li][i] / bc1;
                let v_hat = self.v_w[li][i] / bc2;
                mlp.layers[li].w[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            for i in 0..mlp.layers[li].b.len() {
                let g = b_grads[li][i];
                self.m_b[li][i] = self.beta1 * self.m_b[li][i] + (1.0 - self.beta1) * g;
                self.v_b[li][i] = self.beta2 * self.v_b[li][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m_b[li][i] / bc1;
                let v_hat = self.v_b[li][i] / bc2;
                mlp.layers[li].b[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

// ─────────────────────── resampling (current-frame) ───────────────────────

/// Map a target-resolution pixel index to its continuous LOW-resolution
/// sample coordinate under pixel-center alignment: `(idx + 0.5) * low /
/// target - 0.5`. When `low == target` this is exactly `idx` (identity — see
/// [`bilinear_upsample`]'s scale-1 property).
fn low_coord(target_idx: u32, low_dim: u32, target_dim: u32) -> f32 {
    (target_idx as f32 + 0.5) * (low_dim as f32) / (target_dim as f32) - 0.5
}

/// The 2×2 low-resolution neighbourhood for a target pixel: the four integer
/// tap coordinates (clamped to the low image) and the fractional offset
/// (dx, dy) ∈ [0,1]² within the cell. Pure function of THIS frame's
/// resolutions — no cross-frame anything.
fn bilinear_taps(
    tx: u32,
    ty: u32,
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
) -> ([usize; RADIANCE_TAPS], f32, f32) {
    let fx = low_coord(tx, low_w, target_w);
    let fy = low_coord(ty, low_h, target_h);
    let x0 = fx.floor();
    let y0 = fy.floor();
    let dx = fx - x0;
    let dy = fy - y0;
    let clampi = |v: f32, hi: u32| -> usize { (v.max(0.0) as u32).min(hi - 1) as usize };
    let x0i = clampi(x0, low_w);
    let x1i = clampi(x0 + 1.0, low_w);
    let y0i = clampi(y0, low_h);
    let y1i = clampi(y0 + 1.0, low_h);
    let idx = |x: usize, y: usize| y * low_w as usize + x;
    (
        [idx(x0i, y0i), idx(x1i, y0i), idx(x0i, y1i), idx(x1i, y1i)],
        dx,
        dy,
    )
}

/// Bilinearly upsample a low-resolution radiance image to the target
/// resolution — the NAIVE baseline the neural upscaler must beat, and the
/// base the net predicts a residual over. When `low_w == target_w &&
/// low_h == target_h` this is the EXACT identity (scale-1 degeneracy ordeal,
/// proposal §VIII-3): every fractional offset is zero, so each output pixel
/// equals its single coincident input tap. Current-frame only.
pub fn bilinear_upsample(
    low_radiance: &[Vec3],
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
) -> Vec<Vec3> {
    assert_eq!(low_radiance.len(), (low_w * low_h) as usize);
    let mut out = Vec::with_capacity((target_w * target_h) as usize);
    for ty in 0..target_h {
        for tx in 0..target_w {
            let (taps, dx, dy) = bilinear_taps(tx, ty, low_w, low_h, target_w, target_h);
            let c00 = low_radiance[taps[0]];
            let c10 = low_radiance[taps[1]];
            let c01 = low_radiance[taps[2]];
            let c11 = low_radiance[taps[3]];
            let top = c00 * (1.0 - dx) + c10 * dx;
            let bot = c01 * (1.0 - dx) + c11 * dx;
            out.push(top * (1.0 - dy) + bot * dy);
        }
    }
    out
}

/// Build one target pixel's `INPUT_FEATURES`-long feature vector from
/// CURRENT-FRAME buffers only: the 2×2 low-res demod-log radiance taps, the
/// subpixel offset, and this target pixel's high-res albedo/normal/depth.
/// (One of the `pub fn`s the ban ordeal's signature scan checks.)
#[allow(clippy::too_many_arguments)]
pub fn pixel_features(
    low_radiance: &[Vec3],
    low_w: u32,
    low_h: u32,
    target_w: u32,
    target_h: u32,
    tx: u32,
    ty: u32,
    hi_albedo: Vec3,
    hi_normal: Vec3,
    hi_depth: f32,
) -> [f32; INPUT_FEATURES] {
    let (taps, dx, dy) = bilinear_taps(tx, ty, low_w, low_h, target_w, target_h);
    let divisor = demod_divisor(hi_albedo);
    let mut f = [0.0f32; INPUT_FEATURES];
    let mut k = 0usize;
    for &tap in &taps {
        let dl = log_demod(low_radiance[tap], divisor);
        f[k] = dl.x;
        f[k + 1] = dl.y;
        f[k + 2] = dl.z;
        k += 3;
    }
    f[k] = dx;
    f[k + 1] = dy;
    k += 2;
    f[k] = hi_albedo.x;
    f[k + 1] = hi_albedo.y;
    f[k + 2] = hi_albedo.z;
    k += 3;
    f[k] = hi_normal.x;
    f[k + 1] = hi_normal.y;
    f[k + 2] = hi_normal.z;
    k += 3;
    f[k] = (hi_depth.max(0.0) + 1.0).ln();
    f
}

/// THE SEAM (live-window wiring binds to THIS; out of scope for wave (a)):
/// upscale a whole low-resolution radiance image to the target resolution,
/// guided by the target-resolution AOV buffers. One MLP forward pass per
/// TARGET pixel, FIXED index order, no threading — byte-deterministic.
/// Inputs are ALL current-frame buffers (low radiance + high-res albedo/
/// normal/depth); no cross-frame state, no frame index. Current-frame only.
///
/// ADVISORY (parked, not blocking wave (a)): this `#[allow(too_many_arguments)]`
/// signature is a candidate for a params-struct refactor (bundle the
/// low/target dims + AOV triple); deferred to the GPU port (VIII-3 wave b),
/// where the WGSL binding layout will force the same grouping anyway —
/// refactor once, not twice.
#[allow(clippy::too_many_arguments)]
pub fn upscale_image(
    mlp: &Mlp,
    low_radiance: &[Vec3],
    low_w: u32,
    low_h: u32,
    hi_albedo: &[Vec3],
    hi_normal: &[Vec3],
    hi_depth: &[f32],
    target_w: u32,
    target_h: u32,
) -> Vec<Vec3> {
    let n = (target_w * target_h) as usize;
    assert_eq!(low_radiance.len(), (low_w * low_h) as usize);
    assert_eq!(hi_albedo.len(), n);
    assert_eq!(hi_normal.len(), n);
    assert_eq!(hi_depth.len(), n);
    let base = bilinear_upsample(low_radiance, low_w, low_h, target_w, target_h);
    let mut out = Vec::with_capacity(n);
    for ty in 0..target_h {
        for tx in 0..target_w {
            let i = (ty * target_w + tx) as usize;
            let divisor = demod_divisor(hi_albedo[i]);
            let base_dl = log_demod(base[i], divisor);
            let features = pixel_features(
                low_radiance,
                low_w,
                low_h,
                target_w,
                target_h,
                tx,
                ty,
                hi_albedo[i],
                hi_normal[i],
                hi_depth[i],
            );
            let residual = mlp.forward(&features);
            let out_dl = Vec3::new(
                base_dl.x + residual[0],
                base_dl.y + residual[1],
                base_dl.z + residual[2],
            );
            out.push(undo_log_demod(out_dl, divisor));
        }
    }
    out
}

// ──────────────────────────── training ────────────────────────────────────

/// One training frame: a low-resolution noisy radiance image + the target-
/// resolution AOVs + the target-resolution converged reference. Whole-frame
/// grouping keeps the train/validation split PER FRAME (never per pixel).
pub struct TrainingFrame {
    pub low_radiance: Vec<Vec3>,
    pub low_w: u32,
    pub low_h: u32,
    pub hi_albedo: Vec<Vec3>,
    pub hi_normal: Vec<Vec3>,
    pub hi_depth: Vec<f32>,
    pub reference: Vec<Vec3>,
    pub target_w: u32,
    pub target_h: u32,
}

/// The residual TARGET for one pixel: `log-demod(reference) -
/// log-demod(bilinear_base)`, both demodulated by the same high-res albedo.
/// Training drives the net's residual output toward this, so an improvement
/// over bilinear is exactly a nonzero learned residual in the right
/// direction.
fn residual_target(reference: Vec3, base: Vec3, divisor: Vec3) -> [f32; OUTPUT_CHANNELS] {
    let ref_dl = log_demod(reference, divisor);
    let base_dl = log_demod(base, divisor);
    [
        ref_dl.x - base_dl.x,
        ref_dl.y - base_dl.y,
        ref_dl.z - base_dl.z,
    ]
}

/// Train `mlp` for one epoch over the frames' pixels (index order fixed: the
/// frame order given, then row-major pixels; no shuffle — callers wanting
/// shuffled minibatches pre-shuffle deterministically). Returns the epoch's
/// mean MSE in residual space for progress reporting.
pub fn train_epoch(
    mlp: &mut Mlp,
    adam: &mut Adam,
    dataset: &[TrainingFrame],
    batch_size: usize,
) -> f64 {
    // Assemble (features, residual_target) pairs once per epoch — the base
    // bilinear image is a pure function of the frame, so recomputing it here
    // keeps training self-contained and deterministic.
    let mut inputs: Vec<[f32; INPUT_FEATURES]> = Vec::new();
    let mut targets: Vec<[f32; OUTPUT_CHANNELS]> = Vec::new();
    for fr in dataset {
        let base = bilinear_upsample(
            &fr.low_radiance,
            fr.low_w,
            fr.low_h,
            fr.target_w,
            fr.target_h,
        );
        for ty in 0..fr.target_h {
            for tx in 0..fr.target_w {
                let i = (ty * fr.target_w + tx) as usize;
                let divisor = demod_divisor(fr.hi_albedo[i]);
                inputs.push(pixel_features(
                    &fr.low_radiance,
                    fr.low_w,
                    fr.low_h,
                    fr.target_w,
                    fr.target_h,
                    tx,
                    ty,
                    fr.hi_albedo[i],
                    fr.hi_normal[i],
                    fr.hi_depth[i],
                ));
                targets.push(residual_target(fr.reference[i], base[i], divisor));
            }
        }
    }

    let mut total_loss = 0.0f64;
    let mut i = 0usize;
    while i < inputs.len() {
        let end = (i + batch_size).min(inputs.len());
        let mut w_grads: Vec<Vec<f32>> = mlp.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut b_grads: Vec<Vec<f32>> = mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        let batch_len = (end - i) as f32;
        for j in i..end {
            let (wg, bg) = mlp.backward(&inputs[j], &targets[j]);
            for li in 0..w_grads.len() {
                for k in 0..w_grads[li].len() {
                    w_grads[li][k] += wg[li][k] / batch_len;
                }
                for k in 0..b_grads[li].len() {
                    b_grads[li][k] += bg[li][k] / batch_len;
                }
            }
            let pred = mlp.forward(&inputs[j]);
            for c in 0..OUTPUT_CHANNELS {
                let d = (pred[c] - targets[j][c]) as f64;
                total_loss += d * d;
            }
        }
        adam.step(mlp, &w_grads, &b_grads);
        i = end;
    }
    total_loss / (inputs.len() * OUTPUT_CHANNELS).max(1) as f64
}

// ─────────────────────────── serialization ────────────────────────────────

const WEIGHTS_MAGIC: &[u8; 8] = b"GAIAUPS1";

/// Serialize `mlp`'s weights: magic (8B), layer count (u32 LE), hidden
/// layers + width (u32 LE each), then per layer (in_dim u32, out_dim u32,
/// weights f32*in*out LE, biases f32*out LE). Deterministic byte layout.
pub fn serialize_weights(mlp: &Mlp) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(WEIGHTS_MAGIC);
    buf.extend_from_slice(&(mlp.layers.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(mlp.config.hidden_layers as u32).to_le_bytes());
    buf.extend_from_slice(&(mlp.config.hidden_width as u32).to_le_bytes());
    for layer in &mlp.layers {
        buf.extend_from_slice(&(layer.in_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(layer.out_dim as u32).to_le_bytes());
        for w in &layer.w {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for b in &layer.b {
            buf.extend_from_slice(&b.to_le_bytes());
        }
    }
    buf
}

/// Deserialize weights written by [`serialize_weights`]. `None` on any
/// malformed input (wrong magic, truncated) rather than panicking.
pub fn deserialize_weights(bytes: &[u8]) -> Option<Mlp> {
    if bytes.len() < 20 || &bytes[0..8] != WEIGHTS_MAGIC {
        return None;
    }
    let mut cursor = 8usize;
    let read_u32 = |cursor: &mut usize, bytes: &[u8]| -> Option<u32> {
        let v = u32::from_le_bytes(bytes.get(*cursor..*cursor + 4)?.try_into().ok()?);
        *cursor += 4;
        Some(v)
    };
    let layer_count = read_u32(&mut cursor, bytes)? as usize;
    let hidden_layers = read_u32(&mut cursor, bytes)? as usize;
    let hidden_width = read_u32(&mut cursor, bytes)? as usize;
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let in_dim = read_u32(&mut cursor, bytes)? as usize;
        let out_dim = read_u32(&mut cursor, bytes)? as usize;
        let mut w = Vec::with_capacity(in_dim * out_dim);
        for _ in 0..(in_dim * out_dim) {
            let b4: [u8; 4] = bytes.get(cursor..cursor + 4)?.try_into().ok()?;
            w.push(f32::from_le_bytes(b4));
            cursor += 4;
        }
        let mut b = Vec::with_capacity(out_dim);
        for _ in 0..out_dim {
            let b4: [u8; 4] = bytes.get(cursor..cursor + 4)?.try_into().ok()?;
            b.push(f32::from_le_bytes(b4));
            cursor += 4;
        }
        layers.push(Layer {
            in_dim,
            out_dim,
            w,
            b,
        });
    }
    Some(Mlp {
        config: UpscaleConfig {
            hidden_layers,
            hidden_width,
        },
        layers,
    })
}

/// Hash-pin helper: sha256 of a weights byte buffer (reuses VIII-1's audited
/// pure-Rust SHA-256 so there is ONE implementation in the crate).
pub fn weights_sha256(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_feature_count_is_twentyone() {
        assert_eq!(INPUT_FEATURES, 21);
    }

    #[test]
    fn bilinear_upsample_is_exact_identity_at_scale_one() {
        // low == target => every output pixel equals its coincident input.
        let img = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.9, 0.0, 0.5),
            Vec3::new(0.4, 0.7, 0.2),
            Vec3::new(0.0, 0.3, 0.8),
        ];
        let out = bilinear_upsample(&img, 2, 2, 2, 2);
        assert_eq!(out, img);
    }

    #[test]
    fn untrained_net_is_exactly_bilinear() {
        // Zero residual head => upscale == bilinear base (log/demod roundtrip
        // cancels). Checked on a non-trivial 2x upscale.
        let (lw, lh, tw, th) = (2u32, 2u32, 4u32, 4u32);
        let low = vec![
            Vec3::new(0.2, 0.4, 0.1),
            Vec3::new(0.8, 0.1, 0.6),
            Vec3::new(0.5, 0.9, 0.3),
            Vec3::new(0.1, 0.2, 0.7),
        ];
        let n = (tw * th) as usize;
        let hi_albedo = vec![Vec3::new(0.6, 0.5, 0.4); n];
        let hi_normal = vec![Vec3::new(0.0, 1.0, 0.0); n];
        let hi_depth = vec![10.0f32; n];
        let mlp = Mlp::new_bilinear_start(UpscaleConfig::default(), 1);
        let base = bilinear_upsample(&low, lw, lh, tw, th);
        let up = upscale_image(
            &mlp, &low, lw, lh, &hi_albedo, &hi_normal, &hi_depth, tw, th,
        );
        for (a, b) in base.iter().zip(up.iter()) {
            assert!(
                (*a - *b).length() < 1e-5,
                "untrained upscale drifted from bilinear base: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn forward_is_pure_and_repeatable() {
        let mlp = Mlp::new_bilinear_start(UpscaleConfig::default(), 7);
        let input = vec![0.2f32; INPUT_FEATURES];
        assert_eq!(mlp.forward(&input), mlp.forward(&input));
    }

    #[test]
    fn weights_roundtrip_through_serialization() {
        let mlp = Mlp::new_bilinear_start(UpscaleConfig::default(), 42);
        let bytes = serialize_weights(&mlp);
        let restored = deserialize_weights(&bytes).expect("deserialize");
        let input = vec![0.15f32; INPUT_FEATURES];
        assert_eq!(mlp.forward(&input), restored.forward(&input));
    }

    #[test]
    fn training_reduces_residual_loss_on_a_learnable_task() {
        // A synthetic upscale where the reference differs from bilinear in a
        // way the geometry features predict: reference = bilinear scaled by a
        // factor keyed to the (constant here) albedo — the net must learn a
        // nonzero residual to reduce loss.
        let (lw, lh, tw, th) = (4u32, 4u32, 8u32, 8u32);
        let low: Vec<Vec3> = (0..(lw * lh))
            .map(|i| Vec3::splat(0.2 + 0.05 * (i % 5) as f32))
            .collect();
        let base = bilinear_upsample(&low, lw, lh, tw, th);
        let n = (tw * th) as usize;
        let hi_albedo = vec![Vec3::new(0.5, 0.5, 0.5); n];
        let hi_normal = vec![Vec3::new(0.0, 1.0, 0.0); n];
        let hi_depth = vec![8.0f32; n];
        // target = base * 1.3 (a residual the net can fit from the taps).
        let reference: Vec<Vec3> = base.iter().map(|c| *c * 1.3).collect();
        let frame = TrainingFrame {
            low_radiance: low,
            low_w: lw,
            low_h: lh,
            hi_albedo,
            hi_normal,
            hi_depth,
            reference,
            target_w: tw,
            target_h: th,
        };
        let mut mlp = Mlp::new_bilinear_start(
            UpscaleConfig {
                hidden_layers: 2,
                hidden_width: 32,
            },
            9,
        );
        let mut adam = Adam::new(&mlp, 0.005, 0.9, 0.999, 1e-8);
        let first = train_epoch(&mut mlp, &mut adam, std::slice::from_ref(&frame), 32);
        let mut last = first;
        for _ in 0..80 {
            last = train_epoch(&mut mlp, &mut adam, std::slice::from_ref(&frame), 32);
        }
        assert!(
            last < first,
            "training did not reduce residual loss: first={first} last={last}"
        );
    }
}
