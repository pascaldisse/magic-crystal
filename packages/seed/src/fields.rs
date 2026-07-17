//! Noise fields built ONLY on the hash — no `rand` crate, no global state.
//!
//! Value noise, gradient (Perlin-style) noise, fBm octave stacks, and domain
//! warp. Every lattice value/gradient is a [`hash_seq`] draw, so a field is a
//! pure function of `(seed, coords)` — the substrate foliage density rides on
//! (§ GEOMETRY.md foliage-as-density).

use crate::hash::{coord_key, coord_key_i64, domain, hash_seq, signed_f32};

/// Quintic fade curve `6t^5 - 15t^4 + 10t^3` (Perlin's smootherstep).
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 8 unit-ish 2D gradient directions (axes + diagonals).
const GRAD2: [(f32, f32); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (
        core::f32::consts::FRAC_1_SQRT_2,
        core::f32::consts::FRAC_1_SQRT_2,
    ),
    (
        -core::f32::consts::FRAC_1_SQRT_2,
        core::f32::consts::FRAC_1_SQRT_2,
    ),
    (
        core::f32::consts::FRAC_1_SQRT_2,
        -core::f32::consts::FRAC_1_SQRT_2,
    ),
    (
        -core::f32::consts::FRAC_1_SQRT_2,
        -core::f32::consts::FRAC_1_SQRT_2,
    ),
];

/// 12 classic Perlin edge-of-cube 3D gradient directions.
const GRAD3: [(f32, f32, f32); 12] = [
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
];

/// Perlin gradient noise is bounded by ~1/sqrt(N); these renormalize to ~[-1,1].
const GRAD2_NORM: f32 = core::f32::consts::SQRT_2;
const GRAD3_NORM: f32 = 1.1547006; // ~ 2/sqrt(3)

/// A seeded noise field. Cheap to copy; carries only the field seed.
#[derive(Clone, Copy, Debug)]
pub struct Noise {
    seed: u64,
}

impl Noise {
    /// A field on `seed`, in the [`domain::FIELD`] key-space.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Noise {
            seed: hash_seq(seed, &[domain::FIELD]),
        }
    }

    #[inline]
    fn lattice2(&self, ix: i32, iy: i32) -> f32 {
        signed_f32(hash_seq(self.seed, &[coord_key(ix), coord_key(iy)]))
    }

    /// `i64` sibling of [`Noise::lattice2`] — same lattice-value draw, but for
    /// lattice cell coordinates that can range beyond `i32` (RITE VII
    /// planet-scale tiles: a coarse octave's lattice cell index grows with
    /// the tile coordinate, so it needs the full `i64` a tile key carries).
    #[inline]
    fn lattice2_i64(&self, ix: i64, iy: i64) -> f32 {
        signed_f32(hash_seq(self.seed, &[coord_key_i64(ix), coord_key_i64(iy)]))
    }

    /// Value noise sampled from an ALREADY-DECOMPOSED lattice location:
    /// `(cell_x, cell_y)` is the integer lattice cell (any `i64` magnitude)
    /// and `(frac_x, frac_y)` is the fractional offset within that cell, each
    /// in `[0, 1)`. Unlike [`Noise::value2`], the caller never hands this a
    /// single large float to `floor()` — the cell/fraction split is exact
    /// integer arithmetic done by the caller (`terrain::height_at_grid_index`
    /// derives it via `i64::div_euclid`/`rem_euclid`), so this stays exact at
    /// any tile-coordinate magnitude instead of losing precision the way a
    /// large `f32`/`f64` world coordinate would.
    #[inline]
    pub fn value2_at_lattice(&self, cell_x: i64, cell_y: i64, frac_x: f32, frac_y: f32) -> f32 {
        let (u, v) = (fade(frac_x), fade(frac_y));
        let v00 = self.lattice2_i64(cell_x, cell_y);
        let v10 = self.lattice2_i64(cell_x + 1, cell_y);
        let v01 = self.lattice2_i64(cell_x, cell_y + 1);
        let v11 = self.lattice2_i64(cell_x + 1, cell_y + 1);
        lerp(lerp(v00, v10, u), lerp(v01, v11, u), v)
    }

    #[inline]
    fn grad2(&self, ix: i32, iy: i32, dx: f32, dy: f32) -> f32 {
        let h = hash_seq(self.seed, &[coord_key(ix), coord_key(iy)]);
        let (gx, gy) = GRAD2[(h % 8) as usize];
        gx * dx + gy * dy
    }

    #[inline]
    fn lattice3(&self, ix: i32, iy: i32, iz: i32) -> f32 {
        signed_f32(hash_seq(
            self.seed,
            &[coord_key(ix), coord_key(iy), coord_key(iz)],
        ))
    }

    #[inline]
    fn grad3(&self, ix: i32, iy: i32, iz: i32, dx: f32, dy: f32, dz: f32) -> f32 {
        let h = hash_seq(self.seed, &[coord_key(ix), coord_key(iy), coord_key(iz)]);
        let (gx, gy, gz) = GRAD3[(h % 12) as usize];
        gx * dx + gy * dy + gz * dz
    }

    /// 2D value noise, output in `[-1, 1]`.
    pub fn value2(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let (ix, iy) = (x0 as i32, y0 as i32);
        let (fx, fy) = (x - x0, y - y0);
        let (u, v) = (fade(fx), fade(fy));
        let v00 = self.lattice2(ix, iy);
        let v10 = self.lattice2(ix + 1, iy);
        let v01 = self.lattice2(ix, iy + 1);
        let v11 = self.lattice2(ix + 1, iy + 1);
        lerp(lerp(v00, v10, u), lerp(v01, v11, u), v)
    }

    /// 2D gradient (Perlin) noise, renormalized to ~`[-1, 1]`.
    pub fn gradient2(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let (ix, iy) = (x0 as i32, y0 as i32);
        let (fx, fy) = (x - x0, y - y0);
        let (u, v) = (fade(fx), fade(fy));
        let n00 = self.grad2(ix, iy, fx, fy);
        let n10 = self.grad2(ix + 1, iy, fx - 1.0, fy);
        let n01 = self.grad2(ix, iy + 1, fx, fy - 1.0);
        let n11 = self.grad2(ix + 1, iy + 1, fx - 1.0, fy - 1.0);
        lerp(lerp(n00, n10, u), lerp(n01, n11, u), v) * GRAD2_NORM
    }

    /// 3D value noise, output in `[-1, 1]`.
    pub fn value3(&self, x: f32, y: f32, z: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let z0 = z.floor();
        let (ix, iy, iz) = (x0 as i32, y0 as i32, z0 as i32);
        let (fx, fy, fz) = (x - x0, y - y0, z - z0);
        let (u, v, w) = (fade(fx), fade(fy), fade(fz));
        let c000 = self.lattice3(ix, iy, iz);
        let c100 = self.lattice3(ix + 1, iy, iz);
        let c010 = self.lattice3(ix, iy + 1, iz);
        let c110 = self.lattice3(ix + 1, iy + 1, iz);
        let c001 = self.lattice3(ix, iy, iz + 1);
        let c101 = self.lattice3(ix + 1, iy, iz + 1);
        let c011 = self.lattice3(ix, iy + 1, iz + 1);
        let c111 = self.lattice3(ix + 1, iy + 1, iz + 1);
        let x00 = lerp(c000, c100, u);
        let x10 = lerp(c010, c110, u);
        let x01 = lerp(c001, c101, u);
        let x11 = lerp(c011, c111, u);
        lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
    }

    /// 3D gradient (Perlin) noise, renormalized to ~`[-1, 1]`.
    pub fn gradient3(&self, x: f32, y: f32, z: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let z0 = z.floor();
        let (ix, iy, iz) = (x0 as i32, y0 as i32, z0 as i32);
        let (fx, fy, fz) = (x - x0, y - y0, z - z0);
        let (u, v, w) = (fade(fx), fade(fy), fade(fz));
        let g = |dx: i32, dy: i32, dz: i32| {
            self.grad3(
                ix + dx,
                iy + dy,
                iz + dz,
                fx - dx as f32,
                fy - dy as f32,
                fz - dz as f32,
            )
        };
        let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
        let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
        let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
        let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
        lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * GRAD3_NORM
    }
}

/// fBm octave-stack parameters. All fields have engine defaults.
#[derive(Clone, Copy, Debug)]
pub struct Fbm {
    /// Number of octaves summed.
    pub octaves: u32,
    /// Frequency multiplier per octave.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub gain: f32,
    /// Base frequency of the first octave.
    pub frequency: f32,
}

impl Default for Fbm {
    fn default() -> Self {
        Fbm {
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            frequency: 1.0,
        }
    }
}

impl Fbm {
    /// Sum a 2D `sampler` (e.g. `|x, y| noise.value2(x, y)`) over the octaves,
    /// amplitude-normalized so the result keeps the sampler's range.
    pub fn sample2<F: Fn(f32, f32) -> f32>(&self, sampler: F, x: f32, y: f32) -> f32 {
        let mut freq = self.frequency;
        let mut amp = 1.0f32;
        let mut sum = 0.0f32;
        let mut norm = 0.0f32;
        for _ in 0..self.octaves {
            sum += amp * sampler(x * freq, y * freq);
            norm += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }

    /// Sum a 3D `sampler` over the octaves, amplitude-normalized.
    pub fn sample3<F: Fn(f32, f32, f32) -> f32>(&self, sampler: F, x: f32, y: f32, z: f32) -> f32 {
        let mut freq = self.frequency;
        let mut amp = 1.0f32;
        let mut sum = 0.0f32;
        let mut norm = 0.0f32;
        for _ in 0..self.octaves {
            sum += amp * sampler(x * freq, y * freq, z * freq);
            norm += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }
}

impl Noise {
    /// Warp a 2D coordinate by the field itself, `strength` units of offset —
    /// the standard "flow" distortion applied before sampling terrain/foliage.
    pub fn domain_warp2(&self, x: f32, y: f32, strength: f32) -> (f32, f32) {
        let wx = self.value2(x + 5.2, y + 1.3);
        let wy = self.value2(x + 1.7, y + 9.2);
        (x + strength * wx, y + strength * wy)
    }
}
