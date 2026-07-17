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
    ///
    /// RECONCILIATION (VI-2, per-bond love vs `SolverConfig::fracture_threshold`):
    /// the break point is `love × threshold` — the PER-BOND `love` is what
    /// actually drives whether and when this bond gives way (two bonds under
    /// identical strife break at different moments if their loves differ, as
    /// `ordeal_fracture_weak_link_breaks` proves for a love-0.1 vs love-1.0
    /// chain). `threshold` is a GLOBAL solver dial — a strife-to-love ratio
    /// (a tolerance/unit-conversion knob, e.g. "how many newtons of strife
    /// does one unit of love withstand"), the same for every bond in the
    /// world. It is never itself a hardcoded break value: no bond tears at a
    /// literal strife number written into this function or `fracture_pass`;
    /// every tear is `strife > love * threshold`, so the threshold path
    /// always flows through the bond's own `love`. grep for `fracture_pass`
    /// and `Bond::fractured` — there is no other break condition anywhere.
    #[inline]
    pub fn fractured(&self, threshold: f64) -> bool {
        self.love < LOVE && self.strife > self.love * threshold
    }
}

/// Reference density (kg/m³) for quarried stone / masonry — GRIMOIRE's
/// essence ordering names stone the densest common building matter ("stone >
/// wood > glass"), and NARUKO's Guardian rulings name a hard stone surface as
/// the VI-2 drop target. A bonded body authored at this density gets bonds at
/// full `LOVE` (unbreakable in ordinary handling — the drop's target surface
/// itself must never fracture under its own definition).
pub const STONE_DENSITY: f64 = 2700.0;

/// Reference density (kg/m³) for balsa — the lightest common structural
/// timber — floors the essence-derived default love so an extremely light
/// bonded body still holds together under nothing but its own weight
/// (`default_bond_love` never returns below this ratio).
pub const BALSA_DENSITY: f64 = 150.0;

/// The essence-derived default love floor: the ratio of the lightest common
/// structural essence (balsa) to the reference unbreakable essence (stone).
/// Documented derivation, not an arbitrary tuning literal (never-hardcode law):
/// `BALSA_DENSITY / STONE_DENSITY`.
pub const BOND_LOVE_FLOOR: f64 = BALSA_DENSITY / STONE_DENSITY;

/// The default per-bond love for a bonded lattice built from a material of
/// the given `density` (kg/m³) — the essence-derived default the VI-2 spec
/// asks for. Density stands in for essence (GRIMOIRE: "stone > wood > glass"
/// is a density ordering): a bond in matter at [`STONE_DENSITY`] or denser is
/// authored unbreakable (`LOVE`); lighter matter gets proportionally less
/// love, floored at [`BOND_LOVE_FLOOR`] so even a very light lattice still
/// coheres under its own weight (a `love` of exactly `0.0` could never hold
/// anything together, which no real bonded matter exhibits).
///
/// This is a DEFAULT, not a forced value — any caller (the `body` sigil, an
/// example, an ordeal) may author an explicit `love` instead; this function
/// only supplies the number when the caller wants "derive it from what the
/// crate is made of" rather than pick one.
///
/// STATUS (adversary A3, night-of-07-17): this is a documented PROXY
/// DEFAULT — a density-ratio heuristic standing in until the Architect
/// rules on materials/essences (`docs/proposals/RITE-VI-STRIFE.md`'s "OPEN
/// W/ ARCHITECT" item 4, which REMAINS OPEN as of this note). It is not a
/// physically-grounded bond-strength model (real material toughness does
/// not track density linearly, or even monotonically — glass is denser
/// than balsa yet far more brittle), and a real essence/materials ruling
/// may replace this function's body entirely, not just retune its
/// constants. Treat every number this function returns as provisional.
pub fn default_bond_love(density: f64) -> f64 {
    (density / STONE_DENSITY).clamp(BOND_LOVE_FLOOR, LOVE)
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
