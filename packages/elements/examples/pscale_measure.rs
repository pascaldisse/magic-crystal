//! P-SCALE — THE BUILDING FALLS · the measurement lane's real product.
//!
//! Builds a bonded multi-storey structure at several particle-count scales,
//! knocks it down with an authored lateral impulse, and records the per-tick
//! solver CPU cost over the FULL collapse, broken down by phase
//! (`Solver::step_profiled` → `PhaseProfile`): constraint solve, fragment
//! flood-fill, body-vs-body O(k²), static collision. Median + worst tick per
//! phase, at 2-3 scales, so the cost curve vs N is visible — measured wall-
//! clock, single core, against the 16.667 ms/tick 60 FPS budget.
//!
//! NOT gated (a perf gate at unknown scale would be a guess) — the numbers
//! ARE the evidence. Run:
//!   cargo run -p elements --release --example pscale_measure

use elements::building::{erect, settle, topple, BuildingSpec};
use elements::PhaseProfile;

const RUN_TICKS: u64 = 400; // ~6.7 s at dt=1/60: standing, collapse, rubble at rest
const TOPPLE_TICK: u64 = 1; // strike right after the 40-tick settle (the standing plateau)
const TOPPLE_SPEED: f64 = 30.0; // authored lateral shove (m/s) on the upper storeys
const TOPPLE_FRACTION: f64 = 0.5; // upper half gets the shove

/// Median of a slice (sorts a copy) — the house convention for the typical
/// tick, robust to the one-off warm-up/GC spike a mean would smear.
fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn worst(xs: &[f64]) -> f64 {
    xs.iter().cloned().fold(0.0, f64::max)
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

struct ScaleResult {
    n: usize,
    fragments_final: usize,
    max_fragments: usize,
    // per-phase per-tick series (ms)
    total: Vec<f64>,
    solve_distance: Vec<f64>,
    cluster_floodfill: Vec<f64>,
    collision_body: Vec<f64>,
    collision_static: Vec<f64>,
    velocity_passes: Vec<f64>,
    integrate: Vec<f64>,
    fracture: Vec<f64>,
    peak_pair_checks: u64,
    peak_clustered: usize,
}

fn run_scale(lattice: (usize, usize, usize)) -> ScaleResult {
    // The building's dials are the locked [`BuildingSpec`] defaults (derived,
    // documented there); only the lattice (hence N) varies across scales.
    let spec = BuildingSpec {
        lattice,
        ..BuildingSpec::default()
    };
    let mut b = erect(spec);
    let n = spec.particle_count();
    // Settle to the standing plateau (see `settle`'s METASTABILITY doc).
    let rest_frags = settle(&mut b, 40);
    // PSCALE_PEAK: a diagnostic mode that prints the standing height, the
    // rest-window strife ceiling, and the collapse's dynamic strife peak —
    // the measurements the fracture-threshold gap in `BuildingSpec` is
    // DERIVED from (see its doc). PSCALE_SPEED overrides the topple speed.
    if std::env::var("PSCALE_PEAK").is_ok() {
        let top = |bb: &elements::building::Building| bb.whole.iter().map(|&p| bb.solver.particles.pos[p].y).fold(0.0_f64, f64::max);
        let speed: f64 = std::env::var("PSCALE_SPEED").ok().and_then(|s| s.parse().ok()).unwrap_or(TOPPLE_SPEED);
        let stand_h = top(&b);
        let steady = b.solver.constraints.iter().map(|c| c.bond.strife).fold(0.0_f64, f64::max);
        b.solver.config.fracture_threshold = f64::INFINITY;
        // Rest ceiling over the STABLE bounded window (60 ticks = 1s) — no topple.
        let mut rest_ceiling = 0.0_f64;
        for _ in 0..60 { b.solver.step(); rest_ceiling = rest_ceiling.max(b.solver.constraints.iter().map(|c| c.bond.strife).fold(0.0, f64::max)); }
        let rest_h = top(&b);
        topple(&mut b, speed, TOPPLE_FRACTION);
        let mut peak = 0.0_f64;
        for _ in 0..200 { b.solver.step(); peak = peak.max(b.solver.constraints.iter().map(|c| c.bond.strife).fold(0.0, f64::max)); }
        eprintln!("   [gap] N={} stand_h={:.1}m rest_h={:.1}m (auth {:.0}m) | steady={:.3e} rest_ceiling={:.3e} peak={:.3e}", n, stand_h, rest_h, b.spec.height, steady, rest_ceiling, peak);
        return ScaleResult { n, fragments_final: 1, max_fragments: 1, total: vec![], solve_distance: vec![], cluster_floodfill: vec![], collision_body: vec![], collision_static: vec![], velocity_passes: vec![], integrate: vec![], fracture: vec![], peak_pair_checks: 0, peak_clustered: 0 };
    }
    let max_strife = b
        .solver
        .constraints
        .iter()
        .map(|c| c.bond.strife)
        .fold(0.0_f64, f64::max);
    eprintln!(
        "   [rest-check] N={} fragments after settle: {} | max steady bond strife: {:.4e} \
         (love*threshold bar = {:.4e})",
        n, rest_frags, max_strife, b.spec.love * b.spec.fracture_threshold
    );

    let mut r = ScaleResult {
        n,
        fragments_final: 1,
        max_fragments: 1,
        total: Vec::new(),
        solve_distance: Vec::new(),
        cluster_floodfill: Vec::new(),
        collision_body: Vec::new(),
        collision_static: Vec::new(),
        velocity_passes: Vec::new(),
        integrate: Vec::new(),
        fracture: Vec::new(),
        peak_pair_checks: 0,
        peak_clustered: 0,
    };

    for t in 0..RUN_TICKS {
        if t == TOPPLE_TICK {
            let speed = std::env::var("PSCALE_SPEED")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(TOPPLE_SPEED);
            topple(&mut b, speed, TOPPLE_FRACTION);
        }
        let p: PhaseProfile = b.solver.step_profiled();
        r.total.push(ms(p.total));
        r.solve_distance.push(ms(p.solve_distance));
        r.cluster_floodfill.push(ms(p.cluster_floodfill));
        r.collision_body.push(ms(p.collision_body));
        r.collision_static.push(ms(p.collision_static));
        r.velocity_passes.push(ms(p.velocity_passes));
        r.integrate.push(ms(p.integrate));
        r.fracture.push(ms(p.fracture_pass));
        r.peak_pair_checks = r.peak_pair_checks.max(p.body_pair_checks);
        r.peak_clustered = r.peak_clustered.max(p.clustered_particles);

        let frags = b.solver.fragment_components(&b.whole).len();
        r.max_fragments = r.max_fragments.max(frags);
    }
    r.fragments_final = b.solver.fragment_components(&b.whole).len();
    r
}

fn main() {
    // Three scales spanning ~250 → ~2000 particles: enough to see the O(k²)
    // body-collision curve bend away from the O(N) phases WITHOUT any single
    // tick running into wall-clock minutes at this atom's default 8 substeps.
    let scales = [(8, 16, 8), (10, 20, 10), (12, 24, 12)];

    let results: Vec<ScaleResult> = scales.iter().map(|&l| run_scale(l)).collect();

    println!("\n=== P-SCALE — THE BUILDING FALLS · per-tick solver cost over the collapse ===");
    println!("config: dt=1/60, substeps=8, {} ticks, topple @tick {} (+{} m/s upper {:.0}%)",
        RUN_TICKS, TOPPLE_TICK, TOPPLE_SPEED, TOPPLE_FRACTION * 100.0);
    println!("budget: 16.667 ms/tick (60 FPS), single core, wall-clock\n");

    for r in &results {
        println!("── N = {} particles ── fragments: 1 → {} (peak {}), \
                  peak clustered={}, peak body-pair-checks/tick={}",
            r.n, r.fragments_final, r.max_fragments, r.peak_clustered, r.peak_pair_checks);
        let row = |name: &str, series: &[f64]| {
            println!("   {:<20} median {:>10.4} ms   worst {:>10.4} ms",
                name, median(series.to_vec()), worst(series));
        };
        row("constraint solve", &r.solve_distance);
        row("fragment floodfill", &r.cluster_floodfill);
        row("body-vs-body O(k^2)", &r.collision_body);
        row("static collision", &r.collision_static);
        row("integrate", &r.integrate);
        row("velocity passes", &r.velocity_passes);
        row("fracture pass", &r.fracture);
        row("WHOLE TICK", &r.total);
        println!();
    }

    // Cost-curve summary: median WHOLE-TICK and the dominant phase vs N.
    println!("=== cost curve vs N (median whole-tick ms, dominant phase) ===");
    for r in &results {
        let phases = [
            ("constraint", median(r.solve_distance.clone())),
            ("floodfill", median(r.cluster_floodfill.clone())),
            ("body-O(k^2)", median(r.collision_body.clone())),
            ("static-col", median(r.collision_static.clone())),
            ("velocity", median(r.velocity_passes.clone())),
        ];
        let dom = phases.iter().cloned().fold(("", 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        let tot = median(r.total.clone());
        println!("   N={:>5}: {:>10.4} ms/tick  ({}x over budget)  dominant: {} @ {:.4} ms",
            r.n, tot, format!("{:.2}", tot / 16.667), dom.0, dom.1);
    }
    println!();
}
