//! Bindings — the Elements' one force. Empedocles refined (the Architect's
//! own words): *love isn't a force like gravity, it IS gravity* — there is
//! ONE interaction, LOVE, the constraint's pull toward rest. STRIFE is not a
//! second force; it is the READOUT — the stress the solver already computes.
//! FRACTURE is the moment the reading exceeds the bond's love.

use crate::LOVE;

/// A bond's love and the strife it is presently under. Love is measured in
/// loves on `[0, 1]` (the unit of binding): `LOVE` (1.0) is unbreakable;
/// anything less can be torn. Strife is the accumulated `|constraint force|`
/// over the current tick's substeps — reset each tick, read at tick's end.
#[derive(Clone, Copy, Debug)]
pub struct Bond {
    /// The strength of the binding, in loves `[0, 1]`. `1.0` == unbreakable.
    pub love: f64,
    /// The stress readout: `|constraint force|` accumulated across this
    /// tick's substeps. Not an input — the solver writes it.
    pub strife: f64,
}

impl Bond {
    /// A bond of the given love. Clamped into the sacred interval `[0, 1]`.
    pub fn new(love: f64) -> Self {
        Bond {
            love: love.clamp(0.0, LOVE),
            strife: 0.0,
        }
    }

    /// The unbreakable bond — love at its center, `1.0`.
    pub fn unbreakable() -> Self {
        Bond {
            love: LOVE,
            strife: 0.0,
        }
    }

    /// Does this bond's strife exceed its love × the world's threshold? A
    /// bond at full love (`>= 1.0`) never fractures, whatever the strife.
    #[inline]
    pub fn fractured(&self, threshold: f64) -> bool {
        self.love < LOVE && self.strife > self.love * threshold
    }
}

/// A compliant distance binding between two particles (XPBD). Covers beams,
/// ropes, cloth edges, chain links — RoR/BeamNG `{k, d, L}` maps 1:1 to
/// `{compliance, _, rest}`. `compliance` is inverse stiffness (`m/N`);
/// `0.0` is perfectly rigid.
#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    /// First bound particle (index into the `Particles` store).
    pub a: usize,
    /// Second bound particle.
    pub b: usize,
    /// The rest length the bond pulls toward.
    pub rest: f64,
    /// Inverse stiffness (`m/N`). `0.0` == rigid.
    pub compliance: f64,
    /// The love/strife of this binding.
    pub bond: Bond,
    /// The XPBD Lagrange multiplier, reset at each substep. Public for
    /// inspection; the solver owns it during a step.
    pub lambda: f64,
}

impl DistanceConstraint {
    /// Bind `a` and `b` at `rest` length, with the given `compliance` and
    /// bond love.
    pub fn new(a: usize, b: usize, rest: f64, compliance: f64, love: f64) -> Self {
        DistanceConstraint {
            a,
            b,
            rest,
            compliance,
            bond: Bond::new(love),
            lambda: 0.0,
        }
    }
}

/// The record the world keeps when a binding tears. The arrow of time is the
/// growing journal (ENTROPY law); a fracture is one line written into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractureEvent {
    /// The tick at which the bond gave way.
    pub tick: u64,
    /// The particles the broken bond joined.
    pub a: usize,
    pub b: usize,
    /// The strife that overcame the love.
    pub strife: f64,
    /// The love the bond held (for the record).
    pub love: f64,
}
