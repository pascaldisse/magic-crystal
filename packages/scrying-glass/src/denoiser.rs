//! RITE VIII-1 — THE DREAM-DENOISER: a tiny per-pixel MLP that reconstructs
//! toward the traced (converged) result from ONE noisy frame plus this SAME
//! frame's auxiliary buffers (albedo/normal/depth). See
//! docs/proposals/RITE-VIII-THE-DREAM-DENOISER.md §VIII-1.
//!
//! THE BAN, architecturally: every public function in this module takes
//! CURRENT-FRAME buffers only — no frame index, no previous-frame parameter,
//! no cross-frame accumulation state anywhere in this module's
//! public API. This is checked by a grep-gate ordeal
//! (`tests/viii1_ordeals.rs`), not merely promised here.
//!
//! Architecture (Breda/NRC-class per-pixel fused MLP): input features per
//! pixel = albedo-demodulated, log-transformed noisy radiance (3) + albedo
//! (3) + normal (3) + log-transformed depth (1) = 10 scalars, ALL sampled
//! from the current frame's AOV/beauty buffers. Hidden layers: ReLU dense,
//! width/depth are parameters (`MlpConfig`, defaulted below). Output: 3
//! scalars, the log-transformed, albedo-demodulated denoised radiance —
//! inverted (expm1, then re-modulated by albedo) to get denoised radiance.
//!
//! Feature engineering (standard, current-frame-only, documented per the
//! atom spec):
//!   - albedo demodulation: `noisy_radiance / (albedo + eps)` before the log
//!     transform, and the network's output is re-multiplied by the SAME
//!     frame's albedo afterward. This separates "how much light landed
//!     here" from "what color is the surface", which the network does not
//!     need to relearn per-material.
//!   - log transform: `ln(1 + x)` on demodulated radiance (HDR-safe,
//!     compresses the long tail so a fixed-width net trained with plain MSE
//!     does not get dominated by a few bright pixels); inverted with
//!     `exp(x) - 1` (expm1) at the output.
//!   - depth: `ln(1 + depth)` (world-space distance can range from a few
//!     units to the far plane in the thousands; the log keeps it on a
//!     comparable scale to the other 9 features without a per-frame
//!     normalization constant, which would itself be a piece of cross-frame
//!     state this module must NOT carry).
//!
//! Pure Rust, f32 inference, FIXED index-ordered loops (`for i in 0..n`,
//! never a HashMap or any other order-unstable structure) — byte-
//! deterministic by construction. No threading in the reference path.

// BAN-SCOPED

use glam::Vec3;

/// Per-pixel input feature count: demodulated-log radiance (3) + albedo (3)
/// + normal (3) + log depth (1).
pub const INPUT_FEATURES: usize = 10;
/// Per-pixel output channel count: denoised radiance (3).
pub const OUTPUT_CHANNELS: usize = 3;

/// Numerical floor added under albedo before dividing (avoids a divide by
/// exactly zero on pure-black/no-hit pixels; small enough not to perturb any
/// real surface's demodulation).
pub const ALBEDO_DEMOD_EPS: f32 = 1e-3;

/// Hidden-layer shape of the per-pixel MLP. Defaults land in the Breda/NRC-
/// class range the proposal names (~4-6 hidden layers, 32-64 wide).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlpConfig {
    pub hidden_layers: usize,
    pub hidden_width: usize,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 4,
            hidden_width: 32,
        }
    }
}

impl MlpConfig {
    /// The full layer-size chain: input -> hidden* -> output.
    fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![INPUT_FEATURES];
        for _ in 0..self.hidden_layers {
            sizes.push(self.hidden_width);
        }
        sizes.push(OUTPUT_CHANNELS);
        sizes
    }
}

/// One dense layer: `out = w * in + b`, row-major `w` (rows = out_dim,
/// cols = in_dim).
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

/// Deterministic, dependency-free PRNG (SplitMix64) — used ONLY for weight
/// initialization (forge-time), never in the inference path. Same seed =>
/// same initial weights, always.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [-1, 1).
    fn next_signed_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 significant bits
        let unit = (bits as f32) / (1u32 << 24) as f32; // [0, 1)
        unit * 2.0 - 1.0
    }
}

/// The per-pixel denoiser MLP. Pure feed-forward, ReLU hidden activations,
/// linear output (the output is a signed log-radiance residual, not a
/// probability/probability-like quantity, so no output nonlinearity).
#[derive(Debug, Clone)]
pub struct Mlp {
    config: MlpConfig,
    layers: Vec<Layer>,
}

impl Mlp {
    /// He-initialized random network (forge-time only — see [`SplitMix64`]).
    /// Deterministic given `seed`; training itself is NOT promised bit-
    /// reproducible (proposal OPEN 4) but a fixed starting point is cheap
    /// honesty and helps debugging.
    pub fn new_random(config: MlpConfig, seed: u64) -> Self {
        let sizes = config.layer_sizes();
        let mut rng = SplitMix64::new(seed);
        let mut layers = Vec::with_capacity(sizes.len() - 1);
        for pair in sizes.windows(2) {
            let (in_dim, out_dim) = (pair[0], pair[1]);
            let mut layer = Layer::zeros(in_dim, out_dim);
            // He init: scale = sqrt(2 / in_dim), drawn from the signed-unit
            // PRNG above (uniform, not Gaussian — a cheap, deterministic,
            // dependency-free approximation; fine for a network this small).
            let scale = (2.0 / in_dim.max(1) as f32).sqrt();
            for w in layer.w.iter_mut() {
                *w = rng.next_signed_unit() * scale;
            }
            // Biases start at zero (standard).
            layers.push(layer);
        }
        Self { config, layers }
    }

    pub fn config(&self) -> MlpConfig {
        self.config
    }

    /// Forward pass for one pixel's feature vector. Fixed loop order
    /// throughout (index-ordered `for` loops only) — byte-deterministic.
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

    /// Forward pass that ALSO returns every layer's pre-activation and
    /// post-activation vectors (needed by [`Self::backward`]). Index-ordered,
    /// same math as [`Self::forward`] — kept as a separate function so the
    /// hot inference path stays simple and allocation-light.
    fn forward_train(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut activations = vec![input.to_vec()];
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        for (li, layer) in self.layers.iter().enumerate() {
            let is_last = li == self.layers.len() - 1;
            let prev = activations.last().unwrap();
            let mut pre = vec![0.0f32; layer.out_dim];
            let mut act = vec![0.0f32; layer.out_dim];
            for o in 0..layer.out_dim {
                let mut sum = layer.b[o];
                let row = o * layer.in_dim;
                for i in 0..layer.in_dim {
                    sum += layer.w[row + i] * prev[i];
                }
                pre[o] = sum;
                act[o] = if is_last { sum } else { sum.max(0.0) };
            }
            pre_activations.push(pre);
            activations.push(act);
        }
        (pre_activations, activations)
    }

    /// One backprop pass for a single pixel's (input, target) pair under
    /// mean-squared-error loss on the OUTPUT_CHANNELS-dim output. Returns
    /// per-layer weight/bias gradients (same shapes as `self.layers`),
    /// index-ordered throughout.
    fn backward(
        &self,
        input: &[f32],
        target: &[f32; OUTPUT_CHANNELS],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let (pre_activations, activations) = self.forward_train(input);
        let n_layers = self.layers.len();
        let mut w_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut b_grads: Vec<Vec<f32>> = self.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();

        // dL/d(output) for MSE over OUTPUT_CHANNELS: 2*(pred - target)/C.
        let output = activations.last().unwrap();
        let mut delta: Vec<f32> = (0..OUTPUT_CHANNELS)
            .map(|c| 2.0 * (output[c] - target[c]) / OUTPUT_CHANNELS as f32)
            .collect();

        for li in (0..n_layers).rev() {
            let layer = &self.layers[li];
            let is_last = li == n_layers - 1;
            // ReLU derivative on hidden layers (last layer is linear).
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

/// Minimal in-repo Adam optimizer (~60 lines) — no new dependency, matches
/// the atom spec's "write it, it's ~100 lines" allowance. Index-ordered,
/// forge-time only (never runs in the inference path).
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

    /// One Adam update from raw (summed, unaveraged is fine — caller
    /// controls batch scale) gradients.
    pub fn step(&mut self, mlp: &mut Mlp, w_grads: &[Vec<f32>], b_grads: &[Vec<f32>]) {
        self.t += 1;
        let t = self.t as f32;
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);
        for li in 0..mlp.layers.len() {
            for i in 0..mlp.layers[li].w.len() {
                let g = w_grads[li][i];
                self.m_w[li][i] = self.beta1 * self.m_w[li][i] + (1.0 - self.beta1) * g;
                self.v_w[li][i] = self.beta2 * self.v_w[li][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m_w[li][i] / bias_correction1;
                let v_hat = self.v_w[li][i] / bias_correction2;
                mlp.layers[li].w[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            for i in 0..mlp.layers[li].b.len() {
                let g = b_grads[li][i];
                self.m_b[li][i] = self.beta1 * self.m_b[li][i] + (1.0 - self.beta1) * g;
                self.v_b[li][i] = self.beta2 * self.v_b[li][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m_b[li][i] / bias_correction1;
                let v_hat = self.v_b[li][i] / bias_correction2;
                mlp.layers[li].b[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

/// Below this squared-length, `albedo` is treated as a NO-HIT (sky) pixel,
/// not a legitimately near-black surface: the AOV export writes exactly
/// zero albedo for a primary-ray miss (there is no surface to shade), so
/// any nonzero length here is real material data.
const NO_HIT_ALBEDO_THRESHOLD_SQ: f32 = 1e-8;

/// The demodulation divisor for a pixel's albedo. Demodulating by "surface
/// albedo" is only meaningful where a surface was actually hit; for a
/// primary-ray miss (sky), `albedo` is exactly zero and dividing by
/// `ALBEDO_DEMOD_EPS` (a small floor meant only to avoid a divide-by-
/// literal-zero on dark-but-real surfaces) would amplify sky radiance by up
/// to ~1000x — blowing the feature/target scale far past hit pixels' and
/// making the network's generalization depend on each frame's sky-vs-hit
/// pixel MIX rather than the underlying signal. So: no-hit pixels use a
/// divisor of 1.0 (no demodulation, radiance passes through as-is); real
/// hits use `albedo + ALBEDO_DEMOD_EPS` as before. Found and fixed during
/// this atom's honest-iteration pass (validation frame `orbit_+40`, which
/// has a different sky/hit mix than the training poses, initially made the
/// denoiser WORSE than noisy — see the training report).
fn demod_divisor(albedo: Vec3) -> Vec3 {
    if albedo.length_squared() > NO_HIT_ALBEDO_THRESHOLD_SQ {
        albedo + Vec3::splat(ALBEDO_DEMOD_EPS)
    } else {
        Vec3::ONE
    }
}

/// Build the 10-scalar feature vector for one pixel from CURRENT-FRAME
/// buffers only (see module docs — this is the architecture guarantee the
/// ban ordeal checks).
pub fn pixel_features(
    noisy_radiance: Vec3,
    albedo: Vec3,
    normal: Vec3,
    depth: f32,
) -> [f32; INPUT_FEATURES] {
    let demod = noisy_radiance / demod_divisor(albedo);
    let log_r = Vec3::new(
        (demod.x.max(0.0) + 1.0).ln(),
        (demod.y.max(0.0) + 1.0).ln(),
        (demod.z.max(0.0) + 1.0).ln(),
    );
    let log_depth = (depth.max(0.0) + 1.0).ln();
    [
        log_r.x, log_r.y, log_r.z, albedo.x, albedo.y, albedo.z, normal.x, normal.y, normal.z,
        log_depth,
    ]
}

/// Invert the network's log/demodulated output space back to linear
/// radiance, re-modulating by the SAME frame's albedo (current-frame only).
fn undo_output_transform(raw: [f32; OUTPUT_CHANNELS], albedo: Vec3) -> Vec3 {
    let expm1 = Vec3::new(raw[0].exp() - 1.0, raw[1].exp() - 1.0, raw[2].exp() - 1.0);
    let demod = Vec3::new(expm1.x.max(0.0), expm1.y.max(0.0), expm1.z.max(0.0));
    demod * demod_divisor(albedo)
}

/// Forward-transform a target (reference) radiance the same way training
/// targets are prepared, for computing the training loss in the network's
/// output space.
fn target_transform(reference_radiance: Vec3, albedo: Vec3) -> [f32; OUTPUT_CHANNELS] {
    let demod = reference_radiance / demod_divisor(albedo);
    [
        (demod.x.max(0.0) + 1.0).ln(),
        (demod.y.max(0.0) + 1.0).ln(),
        (demod.z.max(0.0) + 1.0).ln(),
    ]
}

/// Denoise a whole image: one MLP forward pass per pixel, FIXED index order
/// (`for i in 0..n`), no threading in the reference path — byte-
/// deterministic by construction. Inputs are ALL current-frame buffers
/// (noisy radiance, albedo, normal, depth) — no cross-frame state, no frame index.
/// This is the architecture guarantee THE BAN ordeal checks against this
/// function's public signature.
pub fn denoise_image(
    mlp: &Mlp,
    noisy_radiance: &[Vec3],
    albedo: &[Vec3],
    normal: &[Vec3],
    depth: &[f32],
) -> Vec<Vec3> {
    let n = noisy_radiance.len();
    assert_eq!(albedo.len(), n);
    assert_eq!(normal.len(), n);
    assert_eq!(depth.len(), n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let features = pixel_features(noisy_radiance[i], albedo[i], normal[i], depth[i]);
        let raw = mlp.forward(&features);
        out.push(undo_output_transform(raw, albedo[i]));
    }
    out
}

/// One training example: current-frame inputs + the reference (converged)
/// radiance for the SAME pixel. Whole-frame grouping is the caller's job
/// (dataset generation keeps frames together so a train/validation split can
/// be done PER FRAME, never per pixel — see `examples/viii1_train.rs`).
pub struct TrainingPixel {
    pub noisy_radiance: Vec3,
    pub albedo: Vec3,
    pub normal: Vec3,
    pub depth: f32,
    pub reference_radiance: Vec3,
}

/// Train `mlp` for one epoch over `pixels` (index order fixed — the pixel
/// list's order, whatever the caller assembled; this function does not
/// shuffle, so callers wanting shuffled minibatches must pre-shuffle the
/// slice deterministically before calling). Returns the epoch's mean MSE
/// (network output space) for progress reporting.
pub fn train_epoch(
    mlp: &mut Mlp,
    adam: &mut Adam,
    pixels: &[TrainingPixel],
    batch_size: usize,
) -> f64 {
    let mut total_loss = 0.0f64;
    let mut i = 0usize;
    while i < pixels.len() {
        let end = (i + batch_size).min(pixels.len());
        let batch = &pixels[i..end];
        let mut w_grads: Vec<Vec<f32>> = mlp.layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
        let mut b_grads: Vec<Vec<f32>> = mlp.layers.iter().map(|l| vec![0.0; l.b.len()]).collect();
        for px in batch {
            let features = pixel_features(px.noisy_radiance, px.albedo, px.normal, px.depth);
            let target = target_transform(px.reference_radiance, px.albedo);
            let (wg, bg) = mlp.backward(&features, &target);
            for li in 0..w_grads.len() {
                for k in 0..w_grads[li].len() {
                    w_grads[li][k] += wg[li][k] / batch.len() as f32;
                }
                for k in 0..b_grads[li].len() {
                    b_grads[li][k] += bg[li][k] / batch.len() as f32;
                }
            }
            let pred = mlp.forward(&features);
            for c in 0..OUTPUT_CHANNELS {
                let d = (pred[c] - target[c]) as f64;
                total_loss += d * d;
            }
        }
        adam.step(mlp, &w_grads, &b_grads);
        i = end;
    }
    total_loss / (pixels.len() * OUTPUT_CHANNELS).max(1) as f64
}

// ─────────────────────────── serialization ────────────────────────────────

const WEIGHTS_MAGIC: &[u8; 8] = b"GAIADEN1";

/// Serialize `mlp`'s weights to a flat, versioned binary format: magic (8B),
/// layer count (u32 LE), then per layer (in_dim u32, out_dim u32, weights
/// f32*in*out LE, biases f32*out LE). Deterministic byte layout — fixed
/// field order, no HashMap.
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

/// Deserialize weights written by [`serialize_weights`]. Returns `None` on
/// any malformed input (wrong magic, truncated buffer) rather than panicking
/// — a corrupt data artifact is a caller-visible failure, not a crash.
pub fn deserialize_weights(bytes: &[u8]) -> Option<Mlp> {
    if bytes.len() < 16 || &bytes[0..8] != WEIGHTS_MAGIC {
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
            let bytes4: [u8; 4] = bytes.get(cursor..cursor + 4)?.try_into().ok()?;
            w.push(f32::from_le_bytes(bytes4));
            cursor += 4;
        }
        let mut b = Vec::with_capacity(out_dim);
        for _ in 0..out_dim {
            let bytes4: [u8; 4] = bytes.get(cursor..cursor + 4)?.try_into().ok()?;
            b.push(f32::from_le_bytes(bytes4));
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
        config: MlpConfig {
            hidden_layers,
            hidden_width,
        },
        layers,
    })
}

/// Pure-Rust SHA-256 (no dependency) — used to hash-pin the weights artifact
/// in its provenance sidecar, per the atom spec ("sha256 of the weights
/// pinned... like the golden-hash precedent"). Correctness proven by a unit
/// test against the standard NIST test vector for "abc" below.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    h.iter().map(|x| format!("{x:08x}")).collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_test_vector_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_matches_nist_test_vector_empty() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn weights_roundtrip_through_serialization() {
        let mlp = Mlp::new_random(MlpConfig::default(), 42);
        let bytes = serialize_weights(&mlp);
        let restored = deserialize_weights(&bytes).expect("deserialize");
        let input = vec![0.1f32; INPUT_FEATURES];
        assert_eq!(mlp.forward(&input), restored.forward(&input));
    }

    #[test]
    fn forward_is_pure_and_repeatable() {
        let mlp = Mlp::new_random(MlpConfig::default(), 7);
        let input = vec![0.2f32; INPUT_FEATURES];
        assert_eq!(mlp.forward(&input), mlp.forward(&input));
    }

    #[test]
    fn training_reduces_loss_on_a_trivial_identity_task() {
        // Sanity check for the backprop implementation itself: a network
        // should be able to drive down MSE loss toward a fixed, learnable
        // target over a handful of epochs on a tiny synthetic set.
        let mut mlp = Mlp::new_random(
            MlpConfig {
                hidden_layers: 2,
                hidden_width: 16,
            },
            123,
        );
        let mut adam = Adam::new(&mlp, 0.01, 0.9, 0.999, 1e-8);
        let pixels: Vec<TrainingPixel> = (0..64)
            .map(|i| {
                let t = i as f32 / 64.0;
                TrainingPixel {
                    noisy_radiance: Vec3::splat(t),
                    albedo: Vec3::splat(0.5),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    depth: 10.0,
                    reference_radiance: Vec3::splat(t),
                }
            })
            .collect();
        let first_loss = train_epoch(&mut mlp, &mut adam, &pixels, 8);
        let mut last_loss = first_loss;
        for _ in 0..50 {
            last_loss = train_epoch(&mut mlp, &mut adam, &pixels, 8);
        }
        assert!(
            last_loss < first_loss,
            "training did not reduce loss on a trivial task: first={first_loss} last={last_loss}"
        );
    }
}
