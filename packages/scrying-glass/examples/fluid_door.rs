//! FLUID DOOR — PLAY.b proof harness (magic-crystal-play, playable-physics
//! lane). A param'd volume of water bursts in through an opening above one
//! rim of `bldg_basin` (the naruko realm's stone basin south of
//! `bldg_tower`, `worlds/naruko/scenes/main.json`) and fills it: splash on
//! impact with the dry floor, spread, settle to a flat pool. Built directly
//! on the shared `elements::fluid` pool machinery (the same `fill`/`settle`
//! the render diorama and the ordeals use) — no new physics, no window.
//!
//! NO BUOYANCY CLAIM: this harness drops no object into the water and makes
//! no float/sink assertion. It proves fill -> splash -> settle only.
//!
//! `bldg_basin`'s authored mesh (main.json): a 3.6x3.6m floor slab (0.2m
//! thick) ringed by four 0.2m-thick, 1.0m-tall walls set 1.7m off-centre —
//! inner cavity `[-1.6, 1.6]` in x and z (3.2 x 3.2m), floor top at y=0.2,
//! wall top at y≈1.2. The basin carries no `body` (mesh-only prop in the
//! world today); this harness builds an equivalent physics pool at those
//! same dimensions via `elements::fluid`'s own container (open-top box,
//! floor + 4 walls) rather than loading the world (fluid has no world.json
//! `body` wiring yet — a gap noted in the return report, not silently
//! papered over).
//!
//! Run: cargo run -p scrying-glass --release --example fluid_door

use std::path::Path;

use elements::fluid::{fill, surface_height, FluidPoolSpec};
use elements::fluid_kernel::poly6;
use elements::pointgrid::PointGrid;
use elements::Vec3;

/// PLAY.c — offline scene-state snapshot (window-ban proof): every current
/// fluid particle's position + velocity, written as JSON to
/// `proof/fluid-door-<label>.json`. The headless substitute for a
/// screenshot when what's proven is PARTICLE STATE, not pixels.
fn snapshot(fluid: &[usize], solver: &elements::Solver, label: &str, note: &str) {
    let particles: Vec<serde_json::Value> = fluid
        .iter()
        .map(|&i| {
            let p = solver.particles.pos[i];
            let v = solver.particles.vel[i];
            serde_json::json!({"pos": [p.x, p.y, p.z], "vel": [v.x, v.y, v.z]})
        })
        .collect();
    let doc = serde_json::json!({
        "label": label,
        "engine_tick": solver.tick,
        "note": note,
        "fluid_particle_count": particles.len(),
        "particles": particles,
    });
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proof");
    std::fs::create_dir_all(&dir).expect("proof dir");
    let path = dir.join(format!("fluid-door-{label}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).expect("write snapshot");
    println!("[snapshot] {label} -> {}", path.display());
}

/// bldg_basin's inner cavity, derived from its authored mesh (see module
/// doc): walls at |x|,|z| = 1.7 +/- 0.1 half-thickness -> inner span
/// [-1.6, 1.6] each axis.
const BASIN_INNER: (f64, f64) = (3.2, 3.2);
/// Wall top above the floor slab (mesh wall size.y = 1.0).
const BASIN_WALL_HEIGHT: f64 = 1.0;
/// Particle spacing — coarse enough for a fast headless bench at basin
/// scale (~3.2m span); the render diorama uses 0.06 at a 1.2m pool, this
/// keeps a comparable particles-per-metre density (~0.15 vs 0.06 x 2.67).
const SPACING: f64 = 0.14;
/// The residual "wet floor" film left before the door bursts — the fill()
/// helper always spawns >= one particle layer (see its `fill_dims.y =
/// (fill_height - s).max(s)` floor), so this is the minimum, not a design
/// choice: the basin starts effectively dry.
const RESIDUAL_FILM: f64 = SPACING;

/// The door-burst dial set (IRON — the volume that "fills the basin"
/// through the door). Positioned above the basin's south rim (z = +1.0,
/// inside the [-1.6,1.6] inner span), moving down and inward (-z) so it
/// crosses the open top near that wall rather than clipping through solid
/// geometry — the physical stand-in for "a door bursts open and water
/// floods in": the basin has no roof, so a volume entering from above one
/// wall IS the flood-through-the-doorway event.
const DOOR_DIMS: Vec3 = Vec3 { x: 2.4, y: 0.8, z: 0.8 };
const DOOR_CENTER: Vec3 = Vec3 { x: 0.0, y: BASIN_WALL_HEIGHT + 0.5, z: 1.0 };
const DOOR_VELOCITY: Vec3 = Vec3 { x: 0.0, y: -2.5, z: -3.0 };
const DOOR_RADIUS_FACTOR: f64 = 0.5;

const SETTLE_BEFORE: u64 = 30;
const SPLASH_TICKS: u64 = 150;
const POOL_SETTLE_TICKS: u64 = 400;

fn max_speed(fluid: &[usize], solver: &elements::Solver) -> f64 {
    fluid.iter().map(|&i| solver.particles.vel[i].length()).fold(0.0, f64::max)
}

fn total_kinetic_energy(fluid: &[usize], solver: &elements::Solver) -> f64 {
    fluid
        .iter()
        .map(|&i| {
            let m = 1.0 / solver.particles.inv_mass[i];
            0.5 * m * solver.particles.vel[i].dot(solver.particles.vel[i])
        })
        .sum()
}

fn surface_height_all(fluid: &[usize], solver: &elements::Solver) -> f64 {
    fluid.iter().map(|&i| solver.particles.pos[i].y).fold(f64::NEG_INFINITY, f64::max)
}

/// Fraction of fluid particles whose xz position sits inside the basin's
/// inner footprint — the honest "did it actually land in the basin, not
/// fly over the wall" readout.
fn fraction_in_basin(fluid: &[usize], solver: &elements::Solver) -> f64 {
    let hx = BASIN_INNER.0 * 0.5;
    let hz = BASIN_INNER.1 * 0.5;
    let inside = fluid
        .iter()
        .filter(|&&i| {
            let p = solver.particles.pos[i];
            p.x.abs() <= hx && p.z.abs() <= hz
        })
        .count();
    inside as f64 / fluid.len() as f64
}

/// Max relative over-density (compression readout) — same measure
/// `fluid_measure.rs` uses, imported inline so this harness stays a single
/// self-contained proof file.
fn max_overdensity(fluid: &[usize], solver: &elements::Solver) -> f64 {
    let cfg = solver.fluid.expect("fluid config installed");
    let h = cfg.h;
    let rho0 = cfg.rest_density;
    let grid = PointGrid::build(&solver.particles.pos, fluid, PointGrid::cell_size(h));
    let mut cand = Vec::new();
    let mut worst = 0.0_f64;
    for &i in fluid {
        grid.query_ball(solver.particles.pos[i], h, &mut cand);
        let mut density = 0.0;
        for &jc in &cand {
            let j = jc as usize;
            let mj = 1.0 / solver.particles.inv_mass[j];
            density += mj * poly6((solver.particles.pos[i] - solver.particles.pos[j]).length(), h);
        }
        worst = worst.max(density / rho0 - 1.0);
    }
    worst
}

/// Flatness of the settled surface: max - min column-top height sampled
/// over a coarse grid inside the basin footprint (a true rest pool reads
/// near-zero; a jagged/unsettled one does not).
fn surface_flatness(fluid: &[usize], solver: &elements::Solver) -> f64 {
    let hx = BASIN_INNER.0 * 0.5 - SPACING;
    let hz = BASIN_INNER.1 * 0.5 - SPACING;
    let cell = 0.4_f64;
    let mut tops: Vec<f64> = Vec::new();
    let mut x = -hx;
    while x <= hx {
        let mut z = -hz;
        while z <= hz {
            let top = fluid
                .iter()
                .filter(|&&i| {
                    let p = solver.particles.pos[i];
                    (p.x - x).abs() <= cell * 0.5 && (p.z - z).abs() <= cell * 0.5
                })
                .map(|&i| solver.particles.pos[i].y)
                .fold(f64::NEG_INFINITY, f64::max);
            if top.is_finite() {
                tops.push(top);
            }
            z += cell;
        }
        x += cell;
    }
    if tops.len() < 4 {
        return f64::NAN; // too few sampled columns to call it a surface
    }
    let max = tops.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = tops.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

fn main() {
    let spec = FluidPoolSpec {
        inner: BASIN_INNER,
        wall_height: BASIN_WALL_HEIGHT,
        fill_height: RESIDUAL_FILM,
        spacing: SPACING,
        rest_density: 1000.0,
        h_factor: 3.0,
        fluid_radius_factor: 0.5,
        substeps: 4,
    };
    println!(
        "[door] basin pool: inner {:?} m, wall_height {} m, spacing {} m, residual film {} m",
        spec.inner, spec.wall_height, spec.fill_height, spec.fill_height
    );
    let mut pool = fill(spec);
    println!("[door] residual film: {} fluid particles", pool.fluid.len());

    // Settle the residual film before the burst.
    for _ in 0..SETTLE_BEFORE {
        pool.solver.step();
    }
    println!("[door] residual film settled, surface {:.4} m", surface_height(&pool));

    snapshot(&pool.fluid, &pool.solver, "pre-burst", "residual film settled, door about to open");

    // ── THE DOOR BURSTS — spawn the flood volume above the south rim with
    // an inward+downward impulse, register it into the SAME fluid particle
    // set the pool machinery tracks. ─────────────────────────────────────
    let radius = DOOR_RADIUS_FACTOR * SPACING;
    let burst = pool.solver.spawn_fluid_box(
        DOOR_CENTER,
        DOOR_DIMS,
        SPACING,
        spec.rest_density,
        spec.h_factor,
        radius,
    );
    pool.solver.apply_impulse_to_particles(&burst, DOOR_VELOCITY);
    // Re-derive rho0/cfm from the now-larger packed set (fill() calibrated
    // once for the residual film alone; recalibrating after the burst spawn
    // keeps the incompressibility target honest for the combined volume).
    pool.solver.calibrate_fluid_rest_density();
    let mut all_fluid = pool.fluid.clone();
    all_fluid.extend_from_slice(&burst);
    all_fluid.sort_unstable();
    all_fluid.dedup();
    println!(
        "[door] burst: {} particles, dims {:?} m, center {:?}, velocity {:?} m/s -> total fluid {}",
        burst.len(), DOOR_DIMS, DOOR_CENTER, DOOR_VELOCITY, all_fluid.len()
    );

    let pre_burst_surface = surface_height(&pool);
    let mut peak_splash = pre_burst_surface;
    let mut peak_tick: u64 = 0;
    let mut mid_splash_snapped = false;
    println!("\n-- SPLASH ({SPLASH_TICKS} ticks) --");
    for t in 0..SPLASH_TICKS {
        pool.solver.step();
        let surf = surface_height_all(&all_fluid, &pool.solver);
        if surf > peak_splash {
            peak_splash = surf;
            peak_tick = pool.solver.tick;
        }
        // Mid-splash snapshot: the tick the surface first crosses halfway
        // between the pre-burst rest height and the eventual peak splash
        // (i.e. the burst is clearly airborne/impacting, not yet settled).
        if !mid_splash_snapped && surf > pre_burst_surface + 0.5 {
            snapshot(&all_fluid, &pool.solver, "mid-splash", "surface crossing pre-burst+0.5m, actively splashing");
            mid_splash_snapped = true;
        }
        if t % 25 == 0 {
            println!(
                "tick {:3}: surface {:.4} m, in-basin {:.3}, max speed {:.3} m/s, KE {:.4} J, max overdensity {:.4}",
                pool.solver.tick,
                surf,
                fraction_in_basin(&all_fluid, &pool.solver),
                max_speed(&all_fluid, &pool.solver),
                total_kinetic_energy(&all_fluid, &pool.solver),
                max_overdensity(&all_fluid, &pool.solver),
            );
        }
    }
    println!(
        "[door] peak splash surface {peak_splash:.4} m at tick {peak_tick} (pre-burst {pre_burst_surface:.4} m, wall top {:.2} m)",
        BASIN_WALL_HEIGHT
    );
    if !mid_splash_snapped {
        snapshot(&all_fluid, &pool.solver, "mid-splash", "splash window ended without crossing +0.5m; last splash-phase state");
    }

    println!("\n-- POOL SETTLE ({POOL_SETTLE_TICKS} ticks) --");
    for t in 0..POOL_SETTLE_TICKS {
        pool.solver.step();
        if t % 80 == 0 {
            println!(
                "tick {:3}: surface {:.4} m, in-basin {:.3}, max speed {:.3} m/s, KE {:.4} J",
                pool.solver.tick,
                surface_height_all(&all_fluid, &pool.solver),
                fraction_in_basin(&all_fluid, &pool.solver),
                max_speed(&all_fluid, &pool.solver),
                total_kinetic_energy(&all_fluid, &pool.solver),
            );
        }
    }

    let final_surface = surface_height_all(&all_fluid, &pool.solver);
    let final_in_basin = fraction_in_basin(&all_fluid, &pool.solver);
    let final_speed = max_speed(&all_fluid, &pool.solver);
    let final_ke = total_kinetic_energy(&all_fluid, &pool.solver);
    let flatness = surface_flatness(&all_fluid, &pool.solver);
    println!("\n[door] FINAL: surface {final_surface:.4} m, in-basin {final_in_basin:.3}, max speed {final_speed:.4} m/s, KE {final_ke:.4} J, flatness (max-min column top) {flatness:.4} m");
    snapshot(&all_fluid, &pool.solver, "settled", "final tick, pool settled");

    assert!(peak_splash > pre_burst_surface + 0.05, "the burst must raise the surface above the dry-film rest height (a real splash), peak={peak_splash:.4} pre={pre_burst_surface:.4}");
    assert!(final_in_basin > 0.9, "the flood must land and stay mostly INSIDE the basin footprint, got {final_in_basin:.3}");
    assert!(final_speed < 0.5, "the pool must settle (low residual speed), got {final_speed:.4} m/s");
    assert!(flatness.is_finite() && flatness < 0.3, "the settled surface must read roughly FLAT (a pool, not a jagged pile), got {flatness:.4} m");

    println!("\nFLUID DOOR PROOF PASSED — a param'd volume bursts through the basin's open top, splashes, and settles to a contained pool. (No buoyancy claim: no object was dropped.)");
}
