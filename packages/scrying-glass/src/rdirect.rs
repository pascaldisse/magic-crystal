//! R-DIRECT — THE NET IS THE RENDERER (spike, 07-18, the Architect's ruling).
//!
//! One net that RENDERS DIRECTLY: input = full-res G-buffer geometry
//! (albedo/normal/depth/motion) + sparse traced radiance (1-spp, at the LOW
//! internal resolution); output = THE IMAGE at native resolution. Not a
//! post-process, not a residual over a bilinear base (that is the VIII-3
//! upscaler's shape) — this net EMITS the final radiance directly. The
//! traced rays are the teacher (ground-truth long-accumulation reference) and
//! the guide signal (the 1-spp taps fed in). It FUSES what the shipped chain
//! does in two nets (VIII-1 denoise at low res → VIII-3 upscale to native)
//! into ONE forward pass — the honest thing to weigh against that chain at
//! the SAME ray budget.
//!
//! THE BAN (VIII-1/VIII-3 precedent, same spirit): every public function here
//! takes CURRENT-FRAME buffers ONLY. Space is reconstructed within one
//! honestly traced frame; no frame is dreamt from another. The `motion`
//! channels are a CURRENT-FRAME G-buffer aux (screen-space geometry flow
//! computable from this frame's camera+transform alone, NOT a read of any
//! earlier frame's pixels) — carried per the standard deferred G-buffer the
//! ruling names; in a STATIC-pose dataset they are zero-valued (an honest
//! gap: the signal they carry is only exercised by a moving-camera wave, out
//! of this current-frame spike's static scope).
//!
//! Output space: albedo-demodulated log-radiance (VIII-1's space — HDR-safe,
//! separates "how much light landed" from "surface colour"), inverted (expm1,
//! re-modulate by THIS pixel's high-res albedo) to native RGB. Direct
//! absolute prediction — NOT a residual, so the net truly emits the image.
//!
//! Pure Rust, f32 inference, FIXED index-ordered loops — byte-deterministic.

// BAN-SCOPED

use crate::denoiser::sha256_hex;
use glam::{Vec2, Vec3};

/// Low-res radiance taps gathered per target pixel (2×2 bilinear neighbourhood).
pub const RADIANCE_TAPS: usize = 4;
/// Per-target-pixel input feature count: 2×2 demod-log radiance taps (12) +
/// subpixel offset (2) + high-res albedo (3) + normal (3) + log depth (1) +
/// screen-space motion (2).
pub const INPUT_FEATURES: usize = RADIANCE_TAPS * 3 + 2 + 3 + 3 + 1 + 2;
/// Output channels: the final demod-log radiance (3).
pub const OUTPUT_CHANNELS: usize = 3;

/// Numerical floor under albedo before dividing (VIII-1's `ALBEDO_DEMOD_EPS`).
pub const ALBEDO_DEMOD_EPS: f32 = 1e-3;
/// Below this squared albedo length a pixel is a NO-HIT (sky) primary-ray
/// miss (AOV writes exactly zero albedo there): demodulation skipped.
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

/// Shape config. `scale` (integer low→native factor) is the ONLY magnitude,
/// a parameter, never a frozen pixel count. Hidden defaults land one step
/// deeper/wider than the VIII-3 upscaler (4×64) because this net must do
/// denoise AND upscale in one pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdirectConfig {
    pub hidden_layers: usize,
    pub hidden_width: usize,
}

impl Default for RdirectConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 5,
            hidden_width: 64,
        }
    }
}

impl RdirectConfig {
    fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![INPUT_FEATURES];
        for _ in 0..self.hidden_layers {
            sizes.push(self.hidden_width);
        }
        sizes.push(OUTPUT_CHANNELS);
        sizes
    }
}

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

/// Deterministic dependency-free PRNG (SplitMix64) — weight INIT only.
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

/// The per-target-pixel direct-render MLP: feed-forward, ReLU hidden, LINEAR
/// output (a signed log-radiance value, direct — no output nonlinearity).
#[derive(Debug, Clone)]
pub struct Mlp {
    config: RdirectConfig,
    layers: Vec<Layer>,
}

impl Mlp {
    /// He-initialized network (forge-time only). Deterministic given `seed`;
    /// training is not promised bit-reproducible (a fixed start is cheap
    /// honesty). Unlike the upscaler, the last layer is NOT zeroed — this net
    /// emits the image directly, it has no bilinear base to fall back to.
    pub fn new_random(config: RdirectConfig, seed: u64) -> Self {
        let sizes = config.layer_sizes();
        let mut rng = SplitMix64::new(seed);
        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for pair in sizes.windows(2) {
            let (in_dim, out_dim) = (pair[0], pair[1]);
            let mut layer = Layer::zeros(in_dim, out_dim);
            let scale = (2.0 / in_dim.max(1) as f32).sqrt();
            for w in layer.w.iter_mut() {
                *w = rng.next_signed_unit() * scale;
            }
            layers.push(layer);
        }
        Self { config, layers }
    }

    pub fn config(&self) -> RdirectConfig {
        self.config
    }

    pub fn layer_dims(&self) -> Vec<(u32, u32)> {
        self.layers
            .iter()
            .map(|l| (l.in_dim as u32, l.out_dim as u32))
            .collect()
    }

    /// Total multiply-accumulates per forward pass (for cost accounting).
    pub fn macs(&self) -> u64 {
        self.layers.iter().map(|l| (l.in_dim * l.out_dim) as u64).sum()
    }

    pub fn flat_weights(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for l in &self.layers {
            out.extend_from_slice(&l.w);
            out.extend_from_slice(&l.b);
        }
        out
    }

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

    #[allow(clippy::needless_range_loop)]
    fn backward(
        &self,
        input: &[f32],
        target: &[f32; OUTPUT_CHANNELS],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (pre_activations, activations) = self.forward_train(input);
        let n_layers = self.layers.len();
        let mut w_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut b_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();

        let output = activations.last().unwrap();
        let mut delta: Vec<f32> = (0..OUTPUT_CHANNELS)
            .map(|c| 2.0 * (output[c] - target[c]) / OUTPUT_CHANNELS as f32)
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

/// Minimal in-repo Adam (VIII-1/VIII-3 precedent) — forge-time only.
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

    /// Set the learning rate (for schedule decay — NRC lesson: lr-decay
    /// stabilizes the descent).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
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

fn low_coord(target_idx: u32, low_dim: u32, target_dim: u32) -> f32 {
    (target_idx as f32 + 0.5) * (low_dim as f32) / (target_dim as f32) - 0.5
}

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

/// Bilinearly upsample a low-resolution radiance image (the naive baseline
/// AND the proof's "1-spp input" panel). Current-frame only.
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

/// Build one target pixel's feature vector from CURRENT-FRAME buffers only:
/// 2×2 low-res demod-log radiance taps, subpixel offset, this pixel's
/// high-res albedo/normal/log-depth, and its screen-space motion.
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
    hi_motion: Vec2,
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
    k += 1;
    f[k] = hi_motion.x;
    f[k + 1] = hi_motion.y;
    f
}

/// Direct-render a whole native image from a LOW-res 1-spp radiance frame
/// guided by the native-res G-buffer. One MLP forward per TARGET pixel, fixed
/// index order — byte-deterministic. All current-frame. This is THE renderer:
/// output is the final image, not a correction of a base.
#[allow(clippy::too_many_arguments)]
pub fn direct_render_image(
    mlp: &Mlp,
    low_radiance: &[Vec3],
    low_w: u32,
    low_h: u32,
    hi_albedo: &[Vec3],
    hi_normal: &[Vec3],
    hi_depth: &[f32],
    hi_motion: &[Vec2],
    target_w: u32,
    target_h: u32,
) -> Vec<Vec3> {
    let n = (target_w * target_h) as usize;
    assert_eq!(low_radiance.len(), (low_w * low_h) as usize);
    assert_eq!(hi_albedo.len(), n);
    assert_eq!(hi_normal.len(), n);
    assert_eq!(hi_depth.len(), n);
    assert_eq!(hi_motion.len(), n);
    let mut out = Vec::with_capacity(n);
    for ty in 0..target_h {
        for tx in 0..target_w {
            let i = (ty * target_w + tx) as usize;
            let divisor = demod_divisor(hi_albedo[i]);
            let features = pixel_features(
                low_radiance, low_w, low_h, target_w, target_h, tx, ty, hi_albedo[i],
                hi_normal[i], hi_depth[i], hi_motion[i],
            );
            let out_dl = mlp.forward(&features);
            out.push(undo_log_demod(Vec3::new(out_dl[0], out_dl[1], out_dl[2]), divisor));
        }
    }
    out
}

// ──────────────────────────── training ────────────────────────────────────

/// One training frame: a low-res 1-spp radiance image + native-res G-buffer +
/// native-res converged reference (the truth). Whole-frame grouping keeps the
/// train/validation split PER FRAME (never per pixel).
pub struct TrainingFrame {
    pub low_radiance: Vec<Vec3>,
    pub low_w: u32,
    pub low_h: u32,
    pub hi_albedo: Vec<Vec3>,
    pub hi_normal: Vec<Vec3>,
    pub hi_depth: Vec<f32>,
    pub hi_motion: Vec<Vec2>,
    pub reference: Vec<Vec3>,
    pub target_w: u32,
    pub target_h: u32,
}

/// The absolute TARGET for one pixel: `log-demod(reference)` by the high-res
/// albedo. Direct — the net predicts THIS, not a residual over any base.
fn absolute_target(reference: Vec3, divisor: Vec3) -> [f32; OUTPUT_CHANNELS] {
    let dl = log_demod(reference, divisor);
    [dl.x, dl.y, dl.z]
}

/// Assemble all (features, target) pairs for a dataset once (pure function of
/// the frames — deterministic).
fn assemble_pairs(
    dataset: &[TrainingFrame],
) -> (Vec<[f32; INPUT_FEATURES]>, Vec<[f32; OUTPUT_CHANNELS]>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for fr in dataset {
        for ty in 0..fr.target_h {
            for tx in 0..fr.target_w {
                let i = (ty * fr.target_w + tx) as usize;
                let divisor = demod_divisor(fr.hi_albedo[i]);
                inputs.push(pixel_features(
                    &fr.low_radiance, fr.low_w, fr.low_h, fr.target_w, fr.target_h, tx, ty,
                    fr.hi_albedo[i], fr.hi_normal[i], fr.hi_depth[i], fr.hi_motion[i],
                ));
                targets.push(absolute_target(fr.reference[i], divisor));
            }
        }
    }
    (inputs, targets)
}

/// Train `mlp` for one epoch over the frames' pixels (fixed index order: the
/// frame order given, then row-major pixels; no shuffle). Returns the epoch's
/// mean MSE (output space) for progress reporting.
pub fn train_epoch(
    mlp: &mut Mlp,
    adam: &mut Adam,
    dataset: &[TrainingFrame],
    batch_size: usize,
) -> f64 {
    let (inputs, targets) = assemble_pairs(dataset);
    train_epoch_prepared(mlp, adam, &inputs, &targets, batch_size)
}

/// Train one epoch over PRE-ASSEMBLED pairs (avoids re-featurizing every
/// epoch — the caller assembles once with [`assemble_dataset_pairs`]).
#[allow(clippy::needless_range_loop)]
pub fn train_epoch_prepared(
    mlp: &mut Mlp,
    adam: &mut Adam,
    inputs: &[[f32; INPUT_FEATURES]],
    targets: &[[f32; OUTPUT_CHANNELS]],
    batch_size: usize,
) -> f64 {
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

/// Public re-export for callers that want to assemble once and loop epochs.
pub fn assemble_dataset_pairs(
    dataset: &[TrainingFrame],
) -> (Vec<[f32; INPUT_FEATURES]>, Vec<[f32; OUTPUT_CHANNELS]>) {
    assemble_pairs(dataset)
}

// ─────────────────────────── serialization ────────────────────────────────

const WEIGHTS_MAGIC: &[u8; 8] = b"GAIARDR1";

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
        layers.push(Layer { in_dim, out_dim, w, b });
    }
    Some(Mlp {
        config: RdirectConfig { hidden_layers, hidden_width },
        layers,
    })
}

pub fn weights_sha256(mlp: &Mlp) -> String {
    sha256_hex(&serialize_weights(mlp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_pure_and_repeatable() {
        let mlp = Mlp::new_random(RdirectConfig::default(), 7);
        let input = vec![0.2f32; INPUT_FEATURES];
        assert_eq!(mlp.forward(&input), mlp.forward(&input));
    }

    #[test]
    fn weights_roundtrip_through_serialization() {
        let mlp = Mlp::new_random(RdirectConfig::default(), 42);
        let bytes = serialize_weights(&mlp);
        let restored = deserialize_weights(&bytes).expect("deserialize");
        let input = vec![0.13f32; INPUT_FEATURES];
        assert_eq!(mlp.forward(&input), restored.forward(&input));
    }

    #[test]
    fn feature_count_is_stable() {
        // 12 taps + 2 subpixel + 3 albedo + 3 normal + 1 depth + 2 motion.
        assert_eq!(INPUT_FEATURES, 23);
    }

    #[test]
    fn training_reduces_loss_on_a_tiny_direct_task() {
        let mut mlp = Mlp::new_random(
            RdirectConfig { hidden_layers: 3, hidden_width: 32 },
            123,
        );
        let mut adam = Adam::new(&mlp, 0.01, 0.9, 0.999, 1e-8);
        // One 4×4 native frame from a 2×2 low frame: constant radiance, so the
        // learnable target is a fixed demod-log value everywhere.
        let low = vec![Vec3::splat(0.5); 4];
        let refimg = vec![Vec3::splat(0.5); 16];
        let frame = TrainingFrame {
            low_radiance: low,
            low_w: 2,
            low_h: 2,
            hi_albedo: vec![Vec3::splat(0.5); 16],
            hi_normal: vec![Vec3::new(0.0, 1.0, 0.0); 16],
            hi_depth: vec![10.0; 16],
            hi_motion: vec![Vec2::ZERO; 16],
            reference: refimg,
            target_w: 4,
            target_h: 4,
        };
        let ds = vec![frame];
        let first = train_epoch(&mut mlp, &mut adam, &ds, 8);
        let mut last = first;
        for _ in 0..80 {
            last = train_epoch(&mut mlp, &mut adam, &ds, 8);
        }
        assert!(last < first, "training did not reduce loss: first={first} last={last}");
    }
}
