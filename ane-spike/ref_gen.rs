//! Rust-side golden reference for R-Direct ANE parity.
//! Loads the REAL trained artifact (rdirect-weights-v1.bin, GAIARDR1) and runs
//! the EXACT forward from packages/scrying-glass/src/rdirect.rs::Mlp::forward
//! (feed-forward, ReLU hidden, LINEAR output, fixed index order — f32).
//! Emits N (features[23], output[3]) golden pairs as JSON for the CoreML harness.

use std::fs;
use std::io::Write;

const INPUT_FEATURES: usize = 23;
const OUTPUT_CHANNELS: usize = 3;

struct Layer { in_dim: usize, out_dim: usize, w: Vec<f32>, b: Vec<f32> }
struct Mlp { layers: Vec<Layer> }

fn rd_u32(b: &[u8], c: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*c..*c+4].try_into().unwrap()); *c += 4; v
}
fn rd_f32(b: &[u8], c: &mut usize) -> f32 {
    let v = f32::from_le_bytes(b[*c..*c+4].try_into().unwrap()); *c += 4; v
}

fn load(path: &str) -> Mlp {
    let bytes = fs::read(path).expect("read weights");
    assert_eq!(&bytes[0..8], b"GAIARDR1");
    let mut c = 8usize;
    let layer_count = rd_u32(&bytes, &mut c) as usize;
    let _hidden_layers = rd_u32(&bytes, &mut c);
    let _hidden_width = rd_u32(&bytes, &mut c);
    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let in_dim = rd_u32(&bytes, &mut c) as usize;
        let out_dim = rd_u32(&bytes, &mut c) as usize;
        let mut w = Vec::with_capacity(in_dim*out_dim);
        for _ in 0..(in_dim*out_dim) { w.push(rd_f32(&bytes, &mut c)); }
        let mut b = Vec::with_capacity(out_dim);
        for _ in 0..out_dim { b.push(rd_f32(&bytes, &mut c)); }
        layers.push(Layer { in_dim, out_dim, w, b });
    }
    assert_eq!(c, bytes.len(), "trailing bytes");
    Mlp { layers }
}

impl Mlp {
    fn forward(&self, input: &[f32]) -> [f32; OUTPUT_CHANNELS] {
        let mut act = input.to_vec();
        for (li, layer) in self.layers.iter().enumerate() {
            let is_last = li == self.layers.len()-1;
            let mut next = vec![0.0f32; layer.out_dim];
            for o in 0..layer.out_dim {
                let mut sum = layer.b[o];
                let row = o*layer.in_dim;
                for i in 0..layer.in_dim { sum += layer.w[row+i]*act[i]; }
                next[o] = if is_last { sum } else { sum.max(0.0) };
            }
            act = next;
        }
        [act[0], act[1], act[2]]
    }
}

// SplitMix64 — deterministic feature generation (in-distribution-ish).
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Self { s: seed } }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.s;
        z = (z ^ (z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z>>27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z>>31)
    }
    fn unit(&mut self) -> f32 { ((self.next()>>40) as f32)/((1u32<<24) as f32) } // [0,1)
}

// Build a plausible feature vector matching rdirect.rs pixel_features layout:
// 12 demod-log radiance taps (>=0, small), 2 subpixel [0,1], 3 albedo [0,1],
// 3 normal (unit), 1 log-depth (>=0), 2 motion (0 static dataset).
fn make_features(rng: &mut Rng) -> [f32; INPUT_FEATURES] {
    let mut f = [0.0f32; INPUT_FEATURES];
    // taps: demod-log radiance, ln(x+1) of nonneg ~ [0, ~2]
    for k in 0..12 { f[k] = (rng.unit()*4.0 + 1.0).ln(); }
    f[12] = rng.unit(); f[13] = rng.unit();          // subpixel dx,dy
    f[14] = rng.unit(); f[15] = rng.unit(); f[16] = rng.unit(); // albedo
    // normal: random unit vector
    let nx = rng.unit()*2.0-1.0; let ny = rng.unit()*2.0-1.0; let nz = rng.unit()*2.0-1.0;
    let l = (nx*nx+ny*ny+nz*nz).sqrt().max(1e-6);
    f[17] = nx/l; f[18] = ny/l; f[19] = nz/l;
    f[20] = (rng.unit()*50.0 + 1.0).ln();            // log-depth
    f[21] = 0.0; f[22] = 0.0;                         // motion (static)
    f
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weights = &args[1];
    let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let out = &args[3];
    let mlp = load(weights);
    let mut rng = Rng::new(0xA11CE);
    let mut s = String::from("{\"features\":[");
    let mut outs = String::from("],\"outputs\":[");
    for j in 0..n {
        let f = make_features(&mut rng);
        let o = mlp.forward(&f);
        if j>0 { s.push(','); outs.push(','); }
        s.push('[');
        for (i,v) in f.iter().enumerate() { if i>0 {s.push(',');} s.push_str(&format!("{:.9}", v)); }
        s.push(']');
        outs.push('[');
        for (i,v) in o.iter().enumerate() { if i>0 {outs.push(',');} outs.push_str(&format!("{:.9}", v)); }
        outs.push(']');
    }
    s.push_str(&outs);
    s.push_str("]}");
    let mut fh = fs::File::create(out).unwrap();
    fh.write_all(s.as_bytes()).unwrap();
    eprintln!("wrote {} golden pairs -> {}", n, out);
}
