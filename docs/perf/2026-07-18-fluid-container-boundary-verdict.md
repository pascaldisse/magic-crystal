# Fluid buoyancy — CLOSING verdict (round 10)

2026-07-18 · branch `fluid-truth` · closes the fluid-cure lane.
Outcome **B**: gate 4 (buoyancy) does NOT pass. 3/4 gates green + this map is
the deliverable. NO merge — adversary gates next.

## Gate scoreboard

| gate | witness | status |
|------|---------|--------|
| 1 density-RMS at rest | `ordeal_incompressibility_at_rest_derived` | GREEN |
| 2 min-separation floor | `ordeal_min_separation_holds` | GREEN |
| 3 hydrostatic endurance | `ordeal_hydrostatic_endurance` (6 s, no detonation) | GREEN |
| 4 buoyancy / Archimedes | `ordeal_buoyancy_rises` (#[ignore], EXPECTED RED) | **OPEN** |

Solver locked: `compression_only = true` (unilateral liquid density
constraint), s_corr RETIRED (`tensile_k` inert), min-separation contact floor
(`solve_fluid_contacts`) curing collapse. `container_boundary = false`.

## The map — every lever tried and what it MEASURED

- **s_corr (artificial pressure, Macklin §4)** — round 7. DETONATED with
  `tensile_k>0`; with it OFF the SPH density estimate read a coincident-particle
  collapse as near-ρ₀ (SPH-blind). Killed.
- **tensile_k = 0** — the round-7/8 default. Stable but pressureless bulk.
- **min-separation contact floor** — round 8. GEOMETRIC (non-SPH) collision
  floor `r_min = min_sep_factor × spacing` through the rigid-contact solver.
  Cured collapse forever (gate 2), stabilised the column (gates 1,3). Does NOT
  manufacture pressure.
- **ρ₀-below-packing (`rest_density_factor`, 1.0→0.85)** — round 9. A
  free-surface compression-only pool simply EXPANDS to relieve over-density, so
  the settled column reads UNDER-dense in every depth bin at every factor
  (λ≈0 at rest). Discrimination sweep converged EVERY density (200–2000 kg/m³)
  and every release height to the SAME depth (~0.165 m): ZERO mass
  discrimination — a displacement artifact, not Archimedes. Also breaks gate 1's
  C≤0-at-spawn. Reverted to 1.0.
- **Akinci two-way fluid↔solid coupling (`solid_coupling`)** — round 9. The
  submerged body's particles contribute ψ_b·W to the fluid density and receive
  the mirrored inverse-mass-split correction (live, committed 740955f). With no
  fluid pressure to couple, it produces only a mass-BLIND upward shove (a
  light body's transient PEAK, no net rest discrimination).
- **container-boundary Akinci static samples (`container_boundary`)** —
  round 10, THIS shift. Static floor+wall samples tile the pool at fluid
  spacing, feed ψ_b·W into the SPH density and push the fluid inward with the
  pressure `li`, intended to end the base's boundary-deficient reading and grow
  a depth-increasing λ. **MEASURED: it DETONATES.** With it ON the pool surface
  climbs 0.58 m → 3.2 m in 40 ticks; `ordeal_hydrostatic_endurance` fails at
  tick 0 (max speed 6.90 > CFL bound 3.60 m/s). Off case is stable (surf ~0.38
  over 50 ticks). Root cause: the one-sided, immovable boundary pressure push
  injects energy — there is no stable fixed point under the unilateral
  compression-only λ + Jacobi under-relaxation. Left `false` and inert; gates
  1–3 recovered green. Reproduce with `container_discrimination_probe`
  (`container=true`) or by flipping the default.

## Why compression-only + open surface has no hydrostatic gradient

The unilateral density constraint resists compression only (`max(C_i,0)`), and
ρ₀ is calibrated to the MAX (fullest) packing so `C≤0` at spawn by construction.
A free surface therefore relaxes to `C≈0` everywhere — the column expands until
no bin is over-dense — leaving λ≈0 through the bulk. With no λ gradient there is
no pressure gradient, so neither the pairwise contact coupling nor the Akinci
solid coupling has anything to transmit: a submerged body feels a mass-blind
displacement shove at most, never depth-proportional lift. Gate 1 passes
*precisely because* the bulk is nearly pressureless. Confining the fluid with a
boundary that adds density does create over-density at the base, but as a
one-sided position push it injects energy and detonates rather than settling
into a static gradient.

## Recommended next machinery (escalated — do NOT fake)

A real hydrostatic λ field needs a BILATERAL density constraint or an explicit
pressure term, not the unilateral clamp:

1. **Bilateral (two-sided) density constraint** — resist both compression AND
   stretch relative to a ρ₀ that is NOT max-packing, so the column cannot
   expand its way out of over-density and a depth-increasing λ can stand. Must
   re-derive gate 1's floor (C is no longer ≤0 at spawn).
2. **Explicit EOS pressure term** (e.g. Tait/WCSPH `p = k((ρ/ρ₀)^γ − 1)`) with
   a symmetric pressure force + artificial viscosity, giving a genuine
   depth-proportional pressure independent of the position-constraint clamp.
3. If keeping PBF, add a **proper pressure-mirrored, relaxation-tuned Akinci
   boundary** (ψ from the boundary's own packing — already derived here — but
   applied so the boundary correction is bounded/energy-neutral, not the raw
   one-sided push that detonated). This is the disciplined form of the round-10
   attempt.

Any of these is bigger than a one-shift patch and belongs to a fresh lane. This
lane is CLOSED at 3/4 gates + this map.
