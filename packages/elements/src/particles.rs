//! The Monads that matter is made of. A Structure-of-Arrays store: the
//! solver marches over flat lanes (`pos`, `prev`, `vel`, `inv_mass`) rather
//! than chasing pointers — cache-shaped, the CPU's native grain (Terry's
//! lesson: he runs everything on a CPU).

use crate::hash::StateHasher;
use crate::math::Vec3;

/// A body of particles in Structure-of-Arrays layout. Every lane shares one
/// index space; a particle *is* an index. `inv_mass == 0.0` pins a particle
/// immovable (a fixed anchor — the point the world hangs from).
#[derive(Clone, Debug, Default)]
pub struct Particles {
    /// Current positions — the observable state (the entropy at this tick).
    pub pos: Vec<Vec3>,
    /// Positions at the start of the substep — the memory the velocity is
    /// read back from.
    pub prev: Vec<Vec3>,
    /// Velocities, carried between ticks.
    pub vel: Vec<Vec3>,
    /// Inverse masses (`w = 1/m`). Zero = infinite mass = anchored.
    pub inv_mass: Vec<f64>,
}

impl Particles {
    pub fn new() -> Self {
        Particles::default()
    }

    /// Birth a particle at `pos` with the given inverse mass. Returns its
    /// index — its name in this body's index space.
    pub fn add(&mut self, pos: Vec3, inv_mass: f64) -> usize {
        let id = self.pos.len();
        self.pos.push(pos);
        self.prev.push(pos);
        self.vel.push(Vec3::ZERO);
        self.inv_mass.push(inv_mass);
        id
    }

    /// Birth a particle of finite `mass` (`inv_mass = 1/mass`). A `mass <= 0`
    /// is read as an anchor.
    pub fn add_mass(&mut self, pos: Vec3, mass: f64) -> usize {
        let inv = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.add(pos, inv)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Fold the full observable state into a single fingerprint. The witness
    /// the determinism ordeal holds up: two worldlines that fold identically
    /// ARE the same worldline (byte-for-byte).
    pub fn state_hash(&self) -> u64 {
        let mut h = StateHasher::new();
        h.absorb_u64(self.pos.len() as u64);
        for i in 0..self.pos.len() {
            let p = self.pos[i];
            let v = self.vel[i];
            h.absorb_f64(p.x);
            h.absorb_f64(p.y);
            h.absorb_f64(p.z);
            h.absorb_f64(v.x);
            h.absorb_f64(v.y);
            h.absorb_f64(v.z);
        }
        h.finish()
    }
}
