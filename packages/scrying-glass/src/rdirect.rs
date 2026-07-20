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

// ── N5 SIGNED EVIDENCE: split-radiance feature layout ──────────────────────
// The N5 net takes the radiance evidence as TWO channels instead of one: E
// (direct/specular-chain, sharp) and D (post-diffuse-bounce, noisy). Each is a
// 2×2 demod-log tap set (12 features), so the split base is 24 radiance +
// 2 subpixel + 3 albedo + 3 normal + 1 depth + 2 motion = 35, and the
// recurrent split net widens to 35 + 4 history = 39. The teacher TARGET is
// unchanged (the converged total, 3 demod-log channels) — only the INPUT
// widens. Kept as SEPARATE constants so the 23/27 nets and their parity
// ordeals are byte-untouched; the loader dispatches on the first layer in_dim.
pub const INPUT_FEATURES_SPLIT: usize = RADIANCE_TAPS * 3 * 2 + 2 + 3 + 3 + 1 + 2;
/// N5 recurrent split feature count: split base (35) + reprojected prev demod-
/// log (3) + validity (1) = 39.
pub const HIST_FEATURES_SPLIT: usize = INPUT_FEATURES_SPLIT + 4;

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
    fn layer_sizes_with(&self, input: usize) -> Vec<usize> {
        let mut sizes = vec![input];
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
        Self::new_random_with_input(config, INPUT_FEATURES, seed)
    }

    /// He-init with an EXPLICIT input width — the N2 recurrent net widens the
    /// input layer to [`HIST_FEATURES`] (27) for the reprojected-history channels.
    pub fn new_random_with_input(config: RdirectConfig, input_features: usize, seed: u64) -> Self {
        let sizes = config.layer_sizes_with(input_features);
        Self::new_random_with_sizes(config, &sizes, seed)
    }

    /// v7 TWO-HEAD: explicit input AND output width (bypasses the global
    /// `OUTPUT_CHANNELS` constant — needed for the 6-out split-head net).
    pub fn new_random_with_shape(
        config: RdirectConfig,
        input_features: usize,
        output_features: usize,
        seed: u64,
    ) -> Self {
        let mut sizes = vec![input_features];
        for _ in 0..config.hidden_layers {
            sizes.push(config.hidden_width);
        }
        sizes.push(output_features);
        Self::new_random_with_sizes(config, &sizes, seed)
    }

    fn new_random_with_sizes(config: RdirectConfig, sizes: &[usize], seed: u64) -> Self {
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

    /// v7 TWO-HEAD WARM START: body (every layer but the last) copied
    /// byte-identical from `base` (a converged 3-output net); the last layer
    /// is rebuilt 6-wide — rows 0..3 (the E head) copied from `base`'s
    /// output layer so the net starts exactly where the 3-out checkpoint
    /// left off, rows 3..6 (the new D head) SMALL-random He-init (0.1x
    /// scale) since nothing trained them yet. Panics if `base` isn't 3-out.
    pub fn warm_start_two_head(base: &Mlp, seed: u64) -> Mlp {
        let mut layers = base.layers.clone();
        let n = layers.len();
        let last = layers[n - 1].clone();
        assert_eq!(last.out_dim, 3, "warm_start_two_head expects a 3-output base net");
        let in_dim = last.in_dim;
        let mut new_last = Layer::zeros(in_dim, 6);
        new_last.w[..in_dim * 3].copy_from_slice(&last.w);
        new_last.b[..3].copy_from_slice(&last.b);
        let mut rng = SplitMix64::new(seed);
        let scale = (2.0 / in_dim.max(1) as f32).sqrt() * 0.1;
        for i in 0..in_dim * 3 {
            new_last.w[in_dim * 3 + i] = rng.next_signed_unit() * scale;
        }
        layers[n - 1] = new_last;
        Mlp { config: base.config, layers }
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

    /// Polyak/EMA in-place update: `self = decay*self + (1-decay)*live`.
    /// `self` is the shadow (evaluated/checkpointed) net, `live` is the net
    /// Adam is actually stepping. Panics on shape mismatch (same config).
    pub fn ema_update(&mut self, live: &Mlp, decay: f32) {
        assert_eq!(self.layers.len(), live.layers.len(), "ema_update: layer count mismatch");
        for (shadow, l) in self.layers.iter_mut().zip(&live.layers) {
            assert_eq!(shadow.w.len(), l.w.len(), "ema_update: weight shape mismatch");
            for (sw, lw) in shadow.w.iter_mut().zip(&l.w) {
                *sw = decay * *sw + (1.0 - decay) * *lw;
            }
            for (sb, lb) in shadow.b.iter_mut().zip(&l.b) {
                *sb = decay * *sb + (1.0 - decay) * *lb;
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::needless_range_loop)]
    fn forward_raw(&self, input: &[f32]) -> Vec<f32> {
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
        activation
    }

    /// v7 TWO-HEAD: the RAW last-layer output, untruncated (len == out_dim —
    /// 6 for a two-head net: [E_dl(3), D_dl(3)]). The trainer needs this to
    /// backprop separate E/D losses; ordinary 3-out nets get the same 3
    /// values `forward()` returns.
    pub fn forward_full(&self, input: &[f32]) -> Vec<f32> {
        self.forward_raw(input)
    }

    pub fn forward(&self, input: &[f32]) -> [f32; OUTPUT_CHANNELS] {
        let raw = self.forward_raw(input);
        if raw.len() == 6 && input.len() >= 29 {
            // v7 TWO-HEAD PRESENTATION: raw = [E_dl(3), D_dl(3)] against the
            // split-feature layout's embedded hi-res albedo (input[26..29] —
            // same offset in INPUT_FEATURES_SPLIT and HIST_FEATURES_SPLIT).
            // Undo each head in LINEAR space, sum (E+D = the presented
            // image), re-encode as ONE demod-log value so every existing
            // caller's `expm1(forward())*divisor` reconstructs the right
            // image untouched — ordeal/settle_still need zero changes.
            let hi_albedo = Vec3::new(input[26], input[27], input[28]);
            let divisor = demod_divisor(hi_albedo);
            let e_lin = undo_log_demod(Vec3::new(raw[0], raw[1], raw[2]), divisor);
            let d_lin = undo_log_demod(Vec3::new(raw[3], raw[4], raw[5]), divisor);
            let combined = log_demod(e_lin + d_lin, divisor);
            return [combined.x, combined.y, combined.z];
        }
        [raw[0], raw[1], raw[2]]
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
        // MSE delta at the output: dL/d_out = 2(out-target)/N.
        let output = self.forward(input);
        let delta0: [f32; OUTPUT_CHANNELS] = std::array::from_fn(|c| {
            2.0 * (output[c] - target[c]) / OUTPUT_CHANNELS as f32
        });
        self.backward_from_delta(input, &delta0)
    }

    /// Backprop an ARBITRARY output-space gradient (dL/d_out) — lets a caller
    /// supply a custom loss delta (e.g. MSE + a spatial firefly clamp term, N3).
    #[allow(clippy::needless_range_loop)]
    fn backward_from_delta(
        &self,
        input: &[f32],
        delta0: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (pre_activations, activations) = self.forward_train(input);
        let n_layers = self.layers.len();
        let mut w_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut b_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();

        let mut delta: Vec<f32> = delta0.to_vec();

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
    pub fn lr(&self) -> f32 {
        self.lr
    }

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

/// N5: build one target pixel's SPLIT feature vector from CURRENT-FRAME
/// buffers: E's 2×2 demod-log taps (12), then D's 2×2 demod-log taps (12),
/// then subpixel offset (2), hi-res albedo (3), normal (3), log-depth (1),
/// motion (2) = 35. Both radiance channels share this pixel's albedo demod
/// divisor (so the net sees them in the same output space). E and D resolve
/// from the integrator's split buffer; their sum is the ordinary radiance.
#[allow(clippy::too_many_arguments)]
pub fn pixel_features_split(
    low_e: &[Vec3],
    low_d: &[Vec3],
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
) -> [f32; INPUT_FEATURES_SPLIT] {
    let (taps, dx, dy) = bilinear_taps(tx, ty, low_w, low_h, target_w, target_h);
    let divisor = demod_divisor(hi_albedo);
    let mut f = [0.0f32; INPUT_FEATURES_SPLIT];
    let mut k = 0usize;
    for &tap in &taps {
        let dl = log_demod(low_e[tap], divisor);
        f[k] = dl.x;
        f[k + 1] = dl.y;
        f[k + 2] = dl.z;
        k += 3;
    }
    for &tap in &taps {
        let dl = log_demod(low_d[tap], divisor);
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

// ── v7e EVIDENCE CLAMP (structural act, not a loss penalty) ────────────────
// autopsy abbcb64/v7d-autopsy.log diagnosis: the net systematically
// overshoots genuinely-bright E-structure by 1.15x-4.1x at sparkle-outlier
// pixels (COPIED-dominant: the smoothed TARGET itself is locally bright
// there, the net just amplifies past it). Fix at the ACT, not the loss:
// presented_linear = min(net_linear, gamma * local_max(evidence)), where
// evidence is the SAME E+D composite the net's own input taps are built
// from (`low_e`/`low_d`, bilinearly reconstructed to native res exactly as
// `pixel_features_split` samples them) and local_max is a 3x3 (r=1) max-pool
// of that composite in native-pixel space (so a genuine bright feature that
// only ONE nearby noisy 1-spp tap caught still clears the ceiling). Applied
// in RAW LINEAR radiance space (post albedo-demod undo) — the same space
// net/teacher/E/D are already compared in by the autopsy/ordeal/sparkle
// metric. History (`prev_dl`, log-demod space) is fed forward UNCLAMPED —
// this is a presentation ceiling, not a state edit; it never rewrites what
// the recurrent net believes, only what pixel reaches the screen/loss.
//
// GAMMA DERIVATION (IRON law — derived, not picked): scratch/v7d-autopsy.log's
// overshoot table (net_lum / (E_lum+D_raw_lum), ref_frames-converged, at
// every top-N sparkle outlier, both resolutions) measured OVERSHOOT ratios
// 1.149x-4.119x (matches the task's own "1.15x-4.1x" figure). But ref_frames
// E_full/D_full costs 96 rays/px — unavailable at runtime (defeats the whole
// 1-spp-net point) — so it can only diagnose, not gate. The runtime-honest
// ceiling can only be built from what the net ACTUALLY reads: the low_e/
// low_d 1-spp taps. scratch/v7e-gamma-derive.log measured this directly
// (temporal-MEAN of the composite across the K settle taps — mean, not max:
// max-across-time was tried FIRST and is USELESS, a single noisy 1-spp
// specular sample spikes far above the converged value, so the outlier
// population's own ratio against a max-across-time ceiling fell BELOW 1,
// i.e. the "ceiling" already exceeded the net's overshoot at every gamma≥1;
// mean over K taps approximates what the net's own recurrent averaging
// estimates, variance ~1/K — then spatially 3x3 max-pooled per the task's
// window) split by outlier/non-outlier: non-outlier p99.9 ratio = 1.51
// (480x360) / 1.44 (640x480); outlier ratios cluster at 1.2-2.0 (median 1.6/
// 2.0). GAMMA is set just above the non-outlier p99.9 ceiling (full headroom
// for real bright detail) while sitting below the outlier median (clamps the
// bulk of the overshoot mass) — GAMMA = 1.5.
pub const EVIDENCE_CLAMP_GAMMA_DEFAULT: f32 = 1.5;

/// `GAIA_V7_CLAMP_GAMMA` env override, else the derived default above.
pub fn evidence_clamp_gamma() -> f32 {
    std::env::var("GAIA_V7_CLAMP_GAMMA").ok().and_then(|v| v.parse().ok()).unwrap_or(EVIDENCE_CLAMP_GAMMA_DEFAULT)
}

/// GHOST AUTOPSY (room: sky history smear) — `GAIA_V7_SKY_HISTORY=reject`
/// forces history validity=0 for no-hit/sky pixels instead of the default
/// trivial `prev_miss` accept (both-sky ⇒ valid=1 with NO distance/direction
/// check at all, unlike the geometry branch's depth+normal guard). Default
/// (unset, or any other value) is byte-identical to prior behavior — this is
/// an opt-in diagnostic/fix flag, not a default change. Mirrored exactly in
/// `gather_hist_split` (`rdirect_gather_split.wgsl`) via `HistUniform.params2.y`
/// — see `rdirect_gather.rs::encode`. `rotate` (motion-compensated sky) is
/// NOT implemented; only `reject` and the default are wired.
pub fn sky_history_reject() -> bool {
    matches!(std::env::var("GAIA_V7_SKY_HISTORY").as_deref(), Ok("reject"))
}

/// One step/frame's evidence composite (E+D, bilinearly reconstructed to
/// native res from the exact low-res taps the net reads — same taps
/// `pixel_features_split` samples). The raw building block; NOT yet
/// temporally averaged or spatially pooled.
pub fn evidence_composite_frame(low_e: &[Vec3], low_d: &[Vec3], low_w: u32, low_h: u32, tw: u32, th: u32) -> Vec<Vec3> {
    let e_up = bilinear_upsample(low_e, low_w, low_h, tw, th);
    let d_up = bilinear_upsample(low_d, low_w, low_h, tw, th);
    e_up.iter().zip(d_up.iter()).map(|(&e, &d)| e + d).collect()
}

/// Spatial 3x3 (r=1) max-pool per channel, border-clamped — the task's
/// `local_max` window, applied to an already temporally-averaged composite.
pub fn local_max_3x3(img: &[Vec3], w: u32, h: u32) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; img.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut m = Vec3::ZERO;
            for dy in -1..=1 {
                let ny = (y + dy).clamp(0, h as i32 - 1);
                for dx in -1..=1 {
                    let nx = (x + dx).clamp(0, w as i32 - 1);
                    m = m.max(img[(ny as u32 * w + nx as u32) as usize]);
                }
            }
            out[(y as u32 * w + x as u32) as usize] = m;
        }
    }
    out
}

/// Running TEMPORAL-MEAN accumulator for the evidence-clamp ceiling across a
/// recurrent settle (K unroll steps or a streaming frame sequence). Mean, not
/// max — see the GAMMA DERIVATION note above for why max-across-time is a
/// dead end. `ceiling()` applies the spatial [`local_max_3x3`] on demand (the
/// task's window), so callers can fetch the ceiling after every push.
pub struct EvidenceAccum {
    sum: Vec<Vec3>,
    n: u32,
    w: u32,
    h: u32,
}
impl EvidenceAccum {
    pub fn new(w: u32, h: u32) -> Self {
        Self { sum: vec![Vec3::ZERO; (w * h) as usize], n: 0, w, h }
    }
    /// Fold in one step's evidence composite (from [`evidence_composite_frame`]).
    pub fn push(&mut self, frame_composite: &[Vec3]) {
        for (s, f) in self.sum.iter_mut().zip(frame_composite.iter()) {
            *s += *f;
        }
        self.n += 1;
    }
    /// The current clamp ceiling: temporal mean of everything pushed so far,
    /// then spatially 3x3 max-pooled.
    pub fn ceiling(&self) -> Vec<Vec3> {
        let n = (self.n.max(1)) as f32;
        let mean: Vec<Vec3> = self.sum.iter().map(|&s| s / n).collect();
        local_max_3x3(&mean, self.w, self.h)
    }
}

/// THE CLAMP ITSELF: `presented = min(net_linear, gamma * local_max_evidence)`,
/// per channel, in raw linear radiance space. `local_max_evidence` comes from
/// [`EvidenceAccum::ceiling`] at this pixel.
pub fn clamp_evidence_lin(net_lin: Vec3, local_max_evidence: Vec3, gamma: f32) -> Vec3 {
    Vec3::new(
        net_lin.x.min(gamma * local_max_evidence.x.max(0.0)),
        net_lin.y.min(gamma * local_max_evidence.y.max(0.0)),
        net_lin.z.min(gamma * local_max_evidence.z.max(0.0)),
    )
}

/// N5: the 39-feature recurrent split input — split base (35) + reprojected
/// previous demod-log radiance (3) + validity (1).
pub fn hist_features_split(
    base: &[f32; INPUT_FEATURES_SPLIT],
    prev_dl: [f32; 3],
    valid: f32,
) -> [f32; HIST_FEATURES_SPLIT] {
    let mut f = [0.0f32; HIST_FEATURES_SPLIT];
    f[..INPUT_FEATURES_SPLIT].copy_from_slice(base);
    f[INPUT_FEATURES_SPLIT] = prev_dl[0];
    f[INPUT_FEATURES_SPLIT + 1] = prev_dl[1];
    f[INPUT_FEATURES_SPLIT + 2] = prev_dl[2];
    f[INPUT_FEATURES_SPLIT + 3] = valid;
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

// ══════════════════════════ N2 — MEMORY (reprojected history) ═══════════════
// The dots the Architect sees are single-frame spp=1 VARIANCE the current-frame
// net cannot close from one sample. N2 gives the net a MEMORY: its OWN previous
// output, reprojected into this frame (the light-fix reprojection math, reused
// as FEATURE PLUMBING — not a separate present path), fed as extra per-pixel
// features + a validity flag that GATES it (like the light-fix `still_px` /
// disocclusion reject). One render, one net: the history is INPUT, the image is
// still the only output. Trained with the recurrence UNROLLED (the net's own
// prev output fed back) so it learns to AVERAGE across frames at stillness and
// to DROP stale history under motion (validity=0) — killing dots without ghosts.

/// N2 feature vector: the v2 base (23) + reprojected previous demod-log radiance
/// (3) + history validity (1) = 27. The base half is byte-identical to v2, so a
/// v2 net (in_dim 23) and a v3 net (in_dim 27) coexist; the loader dispatches on
/// the first layer's in_dim.
pub const HIST_FEATURES: usize = INPUT_FEATURES + 4;

/// SNAP_EPS — symmetric pixel-boundary snap (v7 seam closure, room 5). See
/// the mirrored constant + full derivation comment in
/// `rdirect_gather_split.wgsl` (`cam_reproject`) — this value and its use
/// must stay identical on both sides of the CPU/GPU seam. Summary: a static
/// camera's self-reprojected edge pixels land the fractional reproject
/// coordinate EXACTLY on an integer pixel boundary; sub-ULP GPU-vs-CPU noise
/// then flips which side of that boundary the coordinate lands on, which the
/// accept/reject test converts into a full history valid=0/1 disagreement.
/// Snapping the coordinate to the nearest integer whenever within SNAP_EPS of
/// one removes the ambiguity at its root, applied identically on both sides.
/// Magnitude: ~1e-6 observed ULP noise floor, half-pixel tie point is 0.5;
/// 1e-3 sits ~1000x above the former and ~500x below the latter.
const SNAP_EPS: f32 = 1.0e-3;

/// A previous-frame camera pose, enough to reproject a world point into its
/// screen. `half_tan` = tan(fov_y/2); `aspect` = w/h. Kept dependency-free
/// (raw glam vectors) so this module stays decoupled from `scene::Camera`.
#[derive(Debug, Clone, Copy)]
pub struct CamPose {
    pub eye: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
    pub half_tan: f32,
    pub aspect: f32,
}

impl CamPose {
    /// World-space ray direction through target pixel (tx,ty) — the same primary
    /// ray the integrator generates (pixel centre).
    pub fn ray_dir(&self, tx: u32, ty: u32, w: u32, h: u32) -> Vec3 {
        let cx = (2.0 * (tx as f32 + 0.5) / w as f32) - 1.0;
        let cy = 1.0 - (2.0 * (ty as f32 + 0.5) / h as f32);
        (self.forward + self.right * cx * self.half_tan * self.aspect + self.up * cy * self.half_tan)
            .normalize_or_zero()
    }

    /// Reproject a world point into THIS (previous) camera's fractional screen
    /// pixel. `None` when behind the eye or off-screen (a disocclusion). Mirrors
    /// the light-fix `temporal_resolve` reprojection, sign-for-sign.
    pub fn reproject(&self, world: Vec3, w: u32, h: u32) -> Option<(f32, f32)> {
        let rel = world - self.eye;
        let rz = rel.dot(self.forward);
        if rz <= 1e-4 {
            return None;
        }
        let sx = rel.dot(self.right) / (rz * self.half_tan * self.aspect);
        let sy = rel.dot(self.up) / (rz * self.half_tan);
        let mut fpx = (sx + 1.0) * 0.5 * w as f32 - 0.5;
        let mut fpy = (1.0 - sy) * 0.5 * h as f32 - 0.5;
        // Symmetric pixel-boundary snap — see SNAP_EPS doc above. Must mirror
        // `cam_reproject` in rdirect_gather_split.wgsl exactly.
        let snap_x = (fpx + 0.5).floor();
        if (fpx - snap_x).abs() < SNAP_EPS {
            fpx = snap_x;
        }
        let snap_y = (fpy + 0.5).floor();
        if (fpy - snap_y).abs() < SNAP_EPS {
            fpy = snap_y;
        }
        if fpx < 0.0 || fpy < 0.0 || fpx > (w - 1) as f32 || fpy > (h - 1) as f32 {
            return None;
        }
        Some((fpx, fpy))
    }
}

/// Bilinear fetch of a Vec3 image at a fractional position (clamped) — the
/// standard TAA history resample (light-fix `t_hist_bilinear`). `pub` (v8
/// lane, mandate a): the trainer needs this to sample a reprojected-history
/// image outside this module, same as the eval path already does internally
/// — visibility widened only, body byte-identical, no behavior change.
pub fn bilinear_vec3(img: &[Vec3], fx: f32, fy: f32, w: u32, h: u32) -> Vec3 {
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let cl = |v: i32, hi: u32| v.clamp(0, hi as i32 - 1) as usize;
    let x0c = cl(x0, w);
    let x1c = cl(x0 + 1, w);
    let y0c = cl(y0, h);
    let y1c = cl(y0 + 1, h);
    let idx = |x: usize, y: usize| y * w as usize + x;
    let a = img[idx(x0c, y0c)];
    let b = img[idx(x1c, y0c)];
    let c = img[idx(x0c, y1c)];
    let d = img[idx(x1c, y1c)];
    let top = a * (1.0 - tx) + b * tx;
    let bot = c * (1.0 - tx) + d * tx;
    top * (1.0 - ty) + bot * ty
}

/// Assemble one target pixel's 27-feature N2 input: the v2 base (23) followed by
/// the reprojected previous demod-log radiance (3) + validity (1). `prev_dl` is
/// ALREADY in the net's output space (demod-log), so at stillness the net's own
/// previous output feeds straight back (no re-demod round-trip). `valid` is 1.0
/// when the history was reprojected and passed the depth/normal guard, else 0.0
/// (and `prev_dl` must be zeroed by the caller).
pub fn hist_features(base: &[f32; INPUT_FEATURES], prev_dl: [f32; 3], valid: f32) -> [f32; HIST_FEATURES] {
    let mut f = [0.0f32; HIST_FEATURES];
    f[..INPUT_FEATURES].copy_from_slice(base);
    f[INPUT_FEATURES] = prev_dl[0];
    f[INPUT_FEATURES + 1] = prev_dl[1];
    f[INPUT_FEATURES + 2] = prev_dl[2];
    f[INPUT_FEATURES + 3] = valid;
    f
}

/// The absolute target (demod-log radiance) for a reference pixel given its
/// high-res albedo — the net's output space. Public for the v3 trainer.
pub fn target_demod_log(reference: Vec3, hi_albedo: Vec3) -> [f32; OUTPUT_CHANNELS] {
    let divisor = demod_divisor(hi_albedo);
    let dl = log_demod(reference, divisor);
    [dl.x, dl.y, dl.z]
}

/// Backprop one (27-feature, target) pair, accumulating into caller grads.
/// Exposed so the v3 trainer can unroll the recurrence (feed the net's own
/// previous output) without re-featurising through the private dataset path.
pub fn accumulate_backward(
    mlp: &Mlp,
    feat: &[f32; HIST_FEATURES],
    target: &[f32; OUTPUT_CHANNELS],
    w_grads: &mut [Vec<f32>],
    b_grads: &mut [Vec<f32>],
    scale: f32,
) {
    let (wg, bg) = mlp.backward(feat, target);
    for li in 0..w_grads.len() {
        for k in 0..w_grads[li].len() {
            w_grads[li][k] += wg[li][k] * scale;
        }
        for k in 0..b_grads[li].len() {
            b_grads[li][k] += bg[li][k] * scale;
        }
    }
}

/// N3 THE FIREFLY LOSS. Backprop MSE + a SPATIAL FIREFLY CLAMP in one pass.
/// `out` is the net's forward output for `feat` (caller already has it, so no
/// extra forward). MSE delta = 2(out-target)/N. Firefly delta: for each channel
/// an excess `e = out[c] - cap[c]` over the TEACHER's local-neighbourhood cap;
/// if `e > 0` (net brighter than anything the teacher shows nearby) add a
/// heavy quadratic penalty `ff_w * e^2`, delta `2*ff_w*e`. Isolated bright
/// outliers over dark teacher neighbourhoods get crushed; genuine bright edges
/// (high cap) are untouched. Differentiable, cheap, spatial (cap is precomputed
/// from the teacher image). Returns (mse_loss, ff_loss) for monitoring.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_backward_firefly(
    mlp: &Mlp,
    feat: &[f32; HIST_FEATURES],
    out: &[f32; OUTPUT_CHANNELS],
    target: &[f32; OUTPUT_CHANNELS],
    cap: &[f32; OUTPUT_CHANNELS],
    ff_w: f32,
    w_grads: &mut [Vec<f32>],
    b_grads: &mut [Vec<f32>],
    scale: f32,
) -> (f64, f64) {
    let mut delta = [0.0f32; OUTPUT_CHANNELS];
    let mut mse_loss = 0.0f64;
    let mut ff_loss = 0.0f64;
    for c in 0..OUTPUT_CHANNELS {
        let d = out[c] - target[c];
        mse_loss += (d * d) as f64;
        delta[c] = 2.0 * d / OUTPUT_CHANNELS as f32;
        let e = out[c] - cap[c];
        if e > 0.0 {
            ff_loss += (ff_w * e * e) as f64;
            delta[c] += 2.0 * ff_w * e;
        }
    }
    let (wg, bg) = mlp.backward_from_delta(feat, &delta);
    for li in 0..w_grads.len() {
        for k in 0..w_grads[li].len() {
            w_grads[li][k] += wg[li][k] * scale;
        }
        for k in 0..b_grads[li].len() {
            b_grads[li][k] += bg[li][k] * scale;
        }
    }
    (mse_loss, ff_loss)
}

/// N4 THE TEACHER-GATED FIREFLY LOSS (the Pareto escape). Same delta-backward
/// path as `accumulate_backward_firefly`, but the firefly clamp is GATED BY THE
/// TEACHER'S LOCAL TRUTH: the excess-over-cap penalty is multiplied by `gate` —
/// 1.0 ONLY where the teacher's k×k neighbourhood is genuinely DARK, 0.0 where
/// the teacher itself is bright (real neon / lit windows / the cyan waterline).
/// Where gate=0 plain MSE rules — the net is free to render the real emissive
/// exactly, so N3's over-clamp of real cyan neon into a broken smear cannot
/// recur. Where gate=1 (dark neighbourhood, no real light) an invented bright
/// dot over `cap` is crushed. `gate` is precomputed per pixel from the teacher
/// (neighbourhood luminance vs a percentile ceiling). Returns (mse, ff) losses.
///
///   LOSS = MSE(out, teacher) + gate · ff_w · Σ_c relu(out_c − cap_c)²
#[allow(clippy::too_many_arguments)]
pub fn accumulate_backward_firefly_gated(
    mlp: &Mlp,
    feat: &[f32; HIST_FEATURES],
    out: &[f32; OUTPUT_CHANNELS],
    target: &[f32; OUTPUT_CHANNELS],
    cap: &[f32; OUTPUT_CHANNELS],
    gate: f32,
    ff_w: f32,
    w_grads: &mut [Vec<f32>],
    b_grads: &mut [Vec<f32>],
    scale: f32,
) -> (f64, f64) {
    let mut delta = [0.0f32; OUTPUT_CHANNELS];
    let mut mse_loss = 0.0f64;
    let mut ff_loss = 0.0f64;
    let g = gate.clamp(0.0, 1.0);
    for c in 0..OUTPUT_CHANNELS {
        let d = out[c] - target[c];
        mse_loss += (d * d) as f64;
        delta[c] = 2.0 * d / OUTPUT_CHANNELS as f32;
        if g > 0.0 {
            let e = out[c] - cap[c];
            if e > 0.0 {
                ff_loss += (g * ff_w * e * e) as f64;
                delta[c] += 2.0 * g * ff_w * e;
            }
        }
    }
    let (wg, bg) = mlp.backward_from_delta(feat, &delta);
    for li in 0..w_grads.len() {
        for k in 0..w_grads[li].len() {
            w_grads[li][k] += wg[li][k] * scale;
        }
        for k in 0..b_grads[li].len() {
            b_grads[li][k] += bg[li][k] * scale;
        }
    }
    (mse_loss, ff_loss)
}

/// N5: the teacher-gated firefly loss over a SLICE feature (the 39-input split
/// net; `feat.len()` is not a fixed array). Byte-identical math to
/// [`accumulate_backward_firefly_gated`]. With `ff_w == 0.0` this is PLAIN MSE
/// (the N5 default — the split is the structural escape; the gate is only
/// re-armed at low weight if val still shows fireflies).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_backward_firefly_gated_slice(
    mlp: &Mlp,
    feat: &[f32],
    out: &[f32; OUTPUT_CHANNELS],
    target: &[f32; OUTPUT_CHANNELS],
    cap: &[f32; OUTPUT_CHANNELS],
    gate: f32,
    ff_w: f32,
    w_grads: &mut [Vec<f32>],
    b_grads: &mut [Vec<f32>],
    scale: f32,
) -> (f64, f64) {
    let mut delta = [0.0f32; OUTPUT_CHANNELS];
    let mut mse_loss = 0.0f64;
    let mut ff_loss = 0.0f64;
    let g = gate.clamp(0.0, 1.0);
    for c in 0..OUTPUT_CHANNELS {
        let d = out[c] - target[c];
        mse_loss += (d * d) as f64;
        delta[c] = 2.0 * d / OUTPUT_CHANNELS as f32;
        if g > 0.0 && ff_w > 0.0 {
            let e = out[c] - cap[c];
            if e > 0.0 {
                ff_loss += (g * ff_w * e * e) as f64;
                delta[c] += 2.0 * g * ff_w * e;
            }
        }
    }
    let (wg, bg) = mlp.backward_from_delta(feat, &delta);
    for li in 0..w_grads.len() {
        for k in 0..w_grads[li].len() {
            w_grads[li][k] += wg[li][k] * scale;
        }
        for k in 0..b_grads[li].len() {
            b_grads[li][k] += bg[li][k] * scale;
        }
    }
    (mse_loss, ff_loss)
}

/// v7e EVIDENCE CLAMP, TRAINABLE: backprop plain MSE(presented, target)
/// through the ARCHITECTURAL clamp `presented_dl = min(out_dl, ceiling_dl)`.
/// `ceiling_dl` is the clamp ceiling ALREADY converted to demod-log space via
/// [`evidence_ceiling_demod_log`] (log_demod is monotonic per channel, so a
/// log-space clamp against `log_demod(gamma*evidence)` is EXACTLY the linear-
/// space clamp against `gamma*evidence`, no space-conversion of the gradient
/// needed). Where the clamp is ACTIVE (`out_dl[c] > ceiling_dl[c]`) the
/// gradient of `presented_dl[c]` w.r.t. `out_dl[c]` is 0 — the loss can no
/// longer reward pushing the net higher there. This is NOT a loss-shape
/// addition (no penalty term, no cap/gate weight) — same plain MSE, only the
/// ACT the loss is computed against changes, matching what the ordeal/CPU-
/// reference path actually presents.
pub fn evidence_ceiling_demod_log(local_max_evidence: Vec3, gamma: f32, hi_albedo: Vec3) -> [f32; OUTPUT_CHANNELS] {
    let divisor = demod_divisor(hi_albedo);
    let ceiling_lin = Vec3::new(
        (gamma * local_max_evidence.x).max(0.0),
        (gamma * local_max_evidence.y).max(0.0),
        (gamma * local_max_evidence.z).max(0.0),
    );
    let dl = log_demod(ceiling_lin, divisor);
    [dl.x, dl.y, dl.z]
}

#[allow(clippy::too_many_arguments)]
pub fn accumulate_backward_clamped_slice(
    mlp: &Mlp,
    feat: &[f32],
    out_dl: &[f32; OUTPUT_CHANNELS],
    ceiling_dl: &[f32; OUTPUT_CHANNELS],
    target: &[f32; OUTPUT_CHANNELS],
    w_grads: &mut [Vec<f32>],
    b_grads: &mut [Vec<f32>],
    scale: f32,
) -> f64 {
    let mut delta = [0.0f32; OUTPUT_CHANNELS];
    let mut mse_loss = 0.0f64;
    for c in 0..OUTPUT_CHANNELS {
        let presented = out_dl[c].min(ceiling_dl[c]);
        let d = presented - target[c];
        mse_loss += (d * d) as f64;
        let active = out_dl[c] <= ceiling_dl[c];
        delta[c] = if active { 2.0 * d / OUTPUT_CHANNELS as f32 } else { 0.0 };
    }
    let (wg, bg) = mlp.backward_from_delta(feat, &delta);
    for li in 0..w_grads.len() {
        for k in 0..w_grads[li].len() {
            w_grads[li][k] += wg[li][k] * scale;
        }
        for k in 0..b_grads[li].len() {
            b_grads[li][k] += bg[li][k] * scale;
        }
    }
    mse_loss
}

/// v7 STRUCTURAL FIX: two-headed backward. `out` is the net's RAW 6-length
/// [`Mlp::forward_full`] output (E_dl(3), D_dl(3)) for `feat`. `target_e` is
/// the SHARP E teacher (exact, unblurred); `target_d` is the SMOOTHED D
/// teacher (box-blurred at the source). Two independent MSE terms, summed —
/// the single shared 3-out target that let the net trade real-light
/// sharpness for variance-chasing (v7's first structural attempt) cannot
/// recur: each head only ever sees its own teacher. Returns (mse_e, mse_d).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_backward_two_head_slice(
    mlp: &Mlp,
    feat: &[f32],
    out: &[f32],
    target_e: &[f32; 3],
    target_d: &[f32; 3],
    w_grads: &mut [Vec<f32>],
    b_grads: &mut [Vec<f32>],
    scale: f32,
) -> (f64, f64) {
    assert_eq!(out.len(), 6, "two-head backward expects a 6-wide raw output");
    let mut delta = [0.0f32; 6];
    let mut mse_e = 0.0f64;
    let mut mse_d = 0.0f64;
    for c in 0..3 {
        let de = out[c] - target_e[c];
        mse_e += (de * de) as f64;
        delta[c] = 2.0 * de / 6.0;
        let dd = out[c + 3] - target_d[c];
        mse_d += (dd * dd) as f64;
        delta[c + 3] = 2.0 * dd / 6.0;
    }
    let (wg, bg) = mlp.backward_from_delta(feat, &delta);
    for li in 0..w_grads.len() {
        for k in 0..w_grads[li].len() {
            w_grads[li][k] += wg[li][k] * scale;
        }
        for k in 0..b_grads[li].len() {
            b_grads[li][k] += bg[li][k] * scale;
        }
    }
    (mse_e, mse_d)
}

/// Allocate zero grad buffers shaped like the MLP.
pub fn zero_grads(mlp: &Mlp) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    (
        mlp.layers.iter().map(|l| vec![0.0; l.w.len()]).collect(),
        mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect(),
    )
}

/// Apply accumulated grads via Adam (v3 trainer step).
pub fn adam_apply(adam: &mut Adam, mlp: &mut Mlp, w_grads: &[Vec<f32>], b_grads: &[Vec<f32>]) {
    adam.step(mlp, w_grads, b_grads);
}

/// One frame of a reprojection sequence for the recurrent (N2) eval path.
pub struct HistFrame<'a> {
    pub low_radiance: &'a [Vec3],
    pub low_w: u32,
    pub low_h: u32,
    pub hi_albedo: &'a [Vec3],
    pub hi_normal: &'a [Vec3],
    pub hi_depth: &'a [f32],
    pub target_w: u32,
    pub target_h: u32,
    pub cam: CamPose,
}

/// Render a whole sequence through the RECURRENT N2 net: each frame consumes the
/// PREVIOUS frame's net output, reprojected into this frame's screen with a
/// depth+normal validity guard (disocclusions → history dropped, validity 0).
/// Returns every frame's output image (the caller reads the tail for stillness
/// convergence / the motion frames for ghosting). This is the eval embodiment
/// of the live recurrence — the ordeal runs it.
pub fn direct_render_sequence_hist(
    mlp: &Mlp,
    frames: &[HistFrame],
    depth_tol: f32,
    normal_thresh: f32,
) -> Vec<Vec<Vec3>> {
    let mut outputs: Vec<Vec<Vec3>> = Vec::with_capacity(frames.len());
    // Previous frame's net output (demod-log radiance) + gbuffer + camera.
    let mut prev: Option<(Vec<[f32; 3]>, Vec<f32>, Vec<Vec3>, CamPose, u32, u32)> = None;
    for f in frames {
        let tw = f.target_w;
        let th = f.target_h;
        let n = (tw * th) as usize;
        let mut out_rgb = vec![Vec3::ZERO; n];
        let mut out_dl = vec![[0.0f32; 3]; n];
        for ty in 0..th {
            for tx in 0..tw {
                let i = (ty * tw + tx) as usize;
                let albedo = f.hi_albedo[i];
                let divisor = demod_divisor(albedo);
                let base = pixel_features(
                    f.low_radiance, f.low_w, f.low_h, tw, th, tx, ty, albedo, f.hi_normal[i],
                    f.hi_depth[i], Vec2::ZERO,
                );
                // Reproject the previous net output into THIS pixel.
                let (prev_dl, valid) = match &prev {
                    None => ([0.0f32; 3], 0.0f32),
                    Some((p_dl, p_depth, p_norm, p_cam, pw, ph)) => {
                        let depth = f.hi_depth[i];
                        let is_miss = depth <= 0.0;
                        let dir = f.cam.ray_dir(tx, ty, tw, th);
                        let dist = if is_miss { 1.0e5 } else { depth };
                        let world = f.cam.eye + dir * dist;
                        match p_cam.reproject(world, *pw, *ph) {
                            None => ([0.0f32; 3], 0.0f32),
                            Some((fx, fy)) => {
                                let ipx = fx.round().clamp(0.0, (*pw - 1) as f32) as usize;
                                let ipy = fy.round().clamp(0.0, (*ph - 1) as f32) as usize;
                                let pj = ipy * *pw as usize + ipx;
                                let prev_depth = p_depth[pj];
                                let prev_miss = prev_depth <= 0.0;
                                let ok = if is_miss {
                                    prev_miss
                                } else if prev_miss {
                                    false
                                } else {
                                    let dist_prev = (world - p_cam.eye).length();
                                    let depth_ok = (dist_prev - prev_depth).abs()
                                        <= depth_tol * dist_prev.max(1e-4);
                                    let normal_ok = f.hi_normal[i].dot(p_norm[pj]) >= normal_thresh;
                                    depth_ok && normal_ok
                                };
                                if ok {
                                    // Bilinear resample of the prev demod-log output.
                                    let img: Vec<Vec3> =
                                        p_dl.iter().map(|d| Vec3::new(d[0], d[1], d[2])).collect();
                                    let s = bilinear_vec3(&img, fx, fy, *pw, *ph);
                                    ([s.x, s.y, s.z], 1.0)
                                } else {
                                    ([0.0f32; 3], 0.0)
                                }
                            }
                        }
                    }
                };
                let feat = hist_features(&base, prev_dl, valid);
                let dl = mlp.forward(&feat);
                out_dl[i] = dl;
                out_rgb[i] = undo_log_demod(Vec3::new(dl[0], dl[1], dl[2]), divisor);
            }
        }
        prev = Some((
            out_dl,
            f.hi_depth.to_vec(),
            f.hi_normal.to_vec(),
            f.cam,
            tw,
            th,
        ));
        outputs.push(out_rgb);
    }
    outputs
}

// ── N5 SIGNED EVIDENCE: recurrent split-radiance eval ─────────────────────
/// One frame of a reprojection sequence for the N5 split net. Same as
/// [`HistFrame`] but carries the two radiance channels (E, D).
pub struct HistFrameSplit<'a> {
    pub low_e: &'a [Vec3],
    pub low_d: &'a [Vec3],
    pub low_w: u32,
    pub low_h: u32,
    pub hi_albedo: &'a [Vec3],
    pub hi_normal: &'a [Vec3],
    pub hi_depth: &'a [f32],
    pub target_w: u32,
    pub target_h: u32,
    pub cam: CamPose,
}

/// N5: render a sequence through the recurrent split net (39-input). Identical
/// reprojection / validity logic to [`direct_render_sequence_hist`]; only the
/// per-pixel feature is the split base + history.
pub fn direct_render_sequence_hist_split(
    mlp: &Mlp,
    frames: &[HistFrameSplit],
    depth_tol: f32,
    normal_thresh: f32,
) -> Vec<Vec<Vec3>> {
    let gamma = evidence_clamp_gamma();
    let sky_reject = sky_history_reject();
    let mut outputs: Vec<Vec<Vec3>> = Vec::with_capacity(frames.len());
    let mut prev: Option<(Vec<[f32; 3]>, Vec<f32>, Vec<Vec3>, CamPose, u32, u32)> = None;
    // v7e evidence clamp ceiling, TEMPORAL-MEAN accumulated across the frame
    // sequence (a single frame's 1-spp evidence is far too noisy alone —
    // see the GAMMA DERIVATION note above EvidenceAccum). Reset per new
    // (tw,th) in case frames vary size (they don't in current callers).
    let mut evidence_accum: Option<EvidenceAccum> = None;
    for f in frames {
        let tw = f.target_w;
        let th = f.target_h;
        let n = (tw * th) as usize;
        let mut out_rgb = vec![Vec3::ZERO; n];
        let mut out_dl = vec![[0.0f32; 3]; n];
        let frame_composite = evidence_composite_frame(f.low_e, f.low_d, f.low_w, f.low_h, tw, th);
        if evidence_accum.as_ref().map(|a| a.sum.len()) != Some(n) {
            evidence_accum = Some(EvidenceAccum::new(tw, th));
        }
        let accum = evidence_accum.as_mut().unwrap();
        accum.push(&frame_composite);
        let evidence_max = accum.ceiling();
        for ty in 0..th {
            for tx in 0..tw {
                let i = (ty * tw + tx) as usize;
                let albedo = f.hi_albedo[i];
                let divisor = demod_divisor(albedo);
                let base = pixel_features_split(
                    f.low_e, f.low_d, f.low_w, f.low_h, tw, th, tx, ty, albedo, f.hi_normal[i],
                    f.hi_depth[i], Vec2::ZERO,
                );
                let (prev_dl, valid) = match &prev {
                    None => ([0.0f32; 3], 0.0f32),
                    Some((p_dl, p_depth, p_norm, p_cam, pw, ph)) => {
                        let depth = f.hi_depth[i];
                        let is_miss = depth <= 0.0;
                        let dir = f.cam.ray_dir(tx, ty, tw, th);
                        let dist = if is_miss { 1.0e5 } else { depth };
                        let world = f.cam.eye + dir * dist;
                        match p_cam.reproject(world, *pw, *ph) {
                            None => ([0.0f32; 3], 0.0f32),
                            Some((fx, fy)) => {
                                let ipx = fx.round().clamp(0.0, (*pw - 1) as f32) as usize;
                                let ipy = fy.round().clamp(0.0, (*ph - 1) as f32) as usize;
                                let pj = ipy * *pw as usize + ipx;
                                let prev_depth = p_depth[pj];
                                let prev_miss = prev_depth <= 0.0;
                                let ok = if is_miss {
                                    // GHOST AUTOPSY: the plain-`prev_miss` accept has no
                                    // distance/direction check at all (unlike the geometry
                                    // branch below) — under `GAIA_V7_SKY_HISTORY=reject` a
                                    // sky pixel never carries history forward. See
                                    // `sky_history_reject` doc.
                                    prev_miss && !sky_reject
                                } else if prev_miss {
                                    false
                                } else {
                                    let dist_prev = (world - p_cam.eye).length();
                                    let depth_ok = (dist_prev - prev_depth).abs()
                                        <= depth_tol * dist_prev.max(1e-4);
                                    let normal_ok = f.hi_normal[i].dot(p_norm[pj]) >= normal_thresh;
                                    depth_ok && normal_ok
                                };
                                if ok {
                                    let img: Vec<Vec3> =
                                        p_dl.iter().map(|d| Vec3::new(d[0], d[1], d[2])).collect();
                                    let s = bilinear_vec3(&img, fx, fy, *pw, *ph);
                                    ([s.x, s.y, s.z], 1.0)
                                } else {
                                    ([0.0f32; 3], 0.0)
                                }
                            }
                        }
                    }
                };
                let feat = hist_features_split(&base, prev_dl, valid);
                let dl = mlp.forward(&feat);
                out_dl[i] = dl;
                let net_lin = undo_log_demod(Vec3::new(dl[0], dl[1], dl[2]), divisor);
                out_rgb[i] = clamp_evidence_lin(net_lin, evidence_max[i], gamma);
            }
        }
        prev = Some((out_dl, f.hi_depth.to_vec(), f.hi_normal.to_vec(), f.cam, tw, th));
        outputs.push(out_rgb);
    }
    outputs
}

// ──────────────────── THE REAL-IMAGE ORDEAL STAMP + GATE ────────────────────
// THE REAL IMAGE BAR (Architect, 2026-07-18): REAL OR BLACK. The app presents a
// neural frame ONLY when the shipped weights carry a PASS stamp from the
// real-image ordeal (residual-vs-teacher + sparkle bars). The stamp is a sidecar
// file beside the weights whose recorded sha256 must match the weights bytes and
// whose status must be PASS. Unstamped / failing / tampered → the gate denies →
// present_black. There is NO env override: the bar models HIS eye.

/// sha256 of a raw weights blob (== `weights_sha256(mlp)` of the same bytes).
pub fn blob_sha256(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// Canonical stamp path for a weights file: `<weights>.stamp`.
pub fn stamp_path_for(weights_path: &std::path::Path) -> std::path::PathBuf {
    let mut s = weights_path.as_os_str().to_os_string();
    s.push(".stamp");
    std::path::PathBuf::from(s)
}

/// The stamp text an ordeal PASS writes. Deterministic, greppable.
pub fn stamp_pass_text(weights_bytes: &[u8], metrics: &[(&str, f64)]) -> String {
    let mut t = String::from("GAIA-REAL-IMAGE-ORDEAL v1\n");
    t.push_str(&format!("weights_sha256={}\n", blob_sha256(weights_bytes)));
    t.push_str("status=PASS\n");
    for (k, v) in metrics {
        t.push_str(&format!("{k}={v:.6}\n"));
    }
    t
}

/// GATE: do these weights bytes carry a valid PASS stamp at `stamp_path`?
/// True ONLY when the file exists, records this exact sha256, and status=PASS.
/// Any mismatch / missing / FAIL → false (→ present_black). No env override.
pub fn verify_stamp(weights_bytes: &[u8], stamp_path: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(stamp_path) else {
        return false;
    };
    let want_sha = blob_sha256(weights_bytes);
    let mut sha_ok = false;
    let mut pass = false;
    for line in text.lines() {
        let line = line.trim();
        if let Some(sha) = line.strip_prefix("weights_sha256=") {
            sha_ok = sha.trim() == want_sha;
        } else if line == "status=PASS" {
            pass = true;
        }
    }
    sha_ok && pass
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hist_feature_count_is_27() {
        assert_eq!(HIST_FEATURES, 27);
    }

    #[test]
    fn stamp_gate_accepts_only_matching_pass() {
        let dir = std::env::temp_dir().join(format!("gaia-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wpath = dir.join("w.bin");
        let bytes = vec![1u8, 2, 3, 4, 5];
        std::fs::write(&wpath, &bytes).unwrap();
        let spath = stamp_path_for(&wpath);
        // No stamp → denied (v2/unstamped case = BLACK).
        assert!(!verify_stamp(&bytes, &spath));
        // Valid PASS stamp → allowed.
        std::fs::write(&spath, stamp_pass_text(&bytes, &[("resid", 0.01)])).unwrap();
        assert!(verify_stamp(&bytes, &spath));
        // Tampered weights (sha mismatch) → denied.
        let tampered = vec![9u8, 9, 9];
        assert!(!verify_stamp(&tampered, &spath));
        // FAIL stamp → denied.
        std::fs::write(&spath, format!("GAIA-REAL-IMAGE-ORDEAL v1\nweights_sha256={}\nstatus=FAIL\n", blob_sha256(&bytes))).unwrap();
        assert!(!verify_stamp(&bytes, &spath));
        let _ = std::fs::remove_dir_all(&dir);
    }

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
