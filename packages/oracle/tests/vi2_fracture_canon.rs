//! RITE VI-2 — SOMETHING BREAKS. The oracle-facing half of the mandatory
//! ordeal list: (e) no orphan matter, and the same-wave canon-learning
//! ordeal (a gaze immediately after the break tick sees each fragment, with
//! bounds derived from ITS OWN live fragment mesh+transform, and traced to
//! its parent vessel — no lag).
//!
//! DEPENDENCY NOTE (per the VI-2 spec's own instruction to check crate-graph
//! direction first): `scrying-glass` and `oracle` do not depend on each
//! other (checked: neither `Cargo.toml` names the other), so this ordeal
//! cannot live wherever `scrying-glass`'s Dynamics birth fragments and ALSO
//! be read by `oracle::model` in the same binary without a cycle. It lives
//! HERE, in oracle's own dev-only test, using `fracture` (which depends on
//! `elements` + `transmutation` + `crystal`, never on `oracle`) to build the
//! identical break -> flood-fill -> fragment-vessel pipeline `scrying-glass`
//! uses at runtime, then reads it back through `oracle::model::World` —
//! exactly the live-ECS path a real gaze would take. New file (not
//! `canon.rs`) per the instruction that another lane may be touching that
//! file concurrently.

use crystal::Core;
use elements::{default_bond_love, Collider, ContactMaterial, Solver, SolverConfig, Vec3, LOVE};
use fracture::{birth_fragment_entities, compute_fragments, ensure_component};
use oracle::model::World;

/// Drop a soft (breakable) crate onto a ground plane and run it until it
/// fractures. Mirrors `elements/tests/vi2_break_ordeals.rs::drop_scenario`
/// (kept independent per file — this crate does not depend on `elements`'
/// test harness, only its library).
fn broken_crate() -> (Solver, Vec<usize>) {
    let cfg = SolverConfig {
        dt: 1.0 / 120.0,
        substeps: 12,
        fracture_threshold: 4.0e3,
        ..SolverConfig::default()
    };
    let mut s = Solver::new(cfg);
    s.collider = Some(Collider::ground_plane(0.0, 5.0, ContactMaterial::default()));
    let density = 200.0; // soft essence — under STONE_DENSITY, so breakable
    let love = default_bond_love(density);
    assert!(love < LOVE, "test setup must derive a breakable bond");
    let whole = s.spawn_bonded_box(
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        (3, 3, 3),
        density,
        love,
        1.0e-7,
        0.03,
    );
    for _ in 0..400 {
        s.step();
        if !s.fractures.is_empty() {
            break;
        }
    }
    (s, whole)
}

/// Build a live oracle `World` around a freshly-birthed set of fragment
/// vessels, traced to a hand-authored PARENT vessel entity (standing in for
/// the realm-authored crate). No disk round-trip, no reload — the same ECS
/// object the fragments were just written into is the one the gaze reads,
/// which is the same-wave guarantee itself (there is no OTHER wave for a
/// lag to hide in).
fn build_world_with_fragments() -> (World, String, Vec<String>, Solver, Vec<usize>) {
    let (solver, whole) = broken_crate();
    let fragments = compute_fragments(&solver, &whole);
    assert!(fragments.len() >= 2, "setup must actually break, or this ordeal is vacuous");

    let mut core = Core::default();
    // The authored parent vessel: a `transform` + `mesh` (box) entity, the
    // SAME schema real realm-authored vessels carry (see
    // `RITE-VI-STRIFE.md`'s `body` sigil route — the parent is the crate
    // that broke).
    let transform_id = ensure_component(&mut core.world, "transform");
    let mesh_id = ensure_component(&mut core.world, "mesh");
    let parent_entity = core
        .world
        .create_entity(vec![
            (transform_id, serde_json::json!({ "v": { "position": [0.0, 3.0, 0.0] } })),
            (
                mesh_id,
                serde_json::json!({ "v": { "parts": [{ "shape": "box", "size": [1.0, 1.0, 1.0] }] } }),
            ),
        ])
        .expect("create parent vessel");
    let parent_id = "vi2.crate".to_string();
    core.world.bind_gaia_id(parent_id.clone(), parent_entity).unwrap();

    // Birth the fragments THE SAME WAVE — no separate tick, same Core.
    let fragment_ids = birth_fragment_entities(&mut core.world, &parent_id, &fragments);

    let mut world = World::from_core(core, "vi2-ordeal-no-disk");
    world.register(parent_id.clone(), parent_entity, vec!["transform".into(), "mesh".into()]);
    for id in &fragment_ids {
        let entity = world.core.world.entity_for_gaia(id).unwrap();
        world.register(
            id.clone(),
            entity,
            vec!["transform".into(), "mesh".into(), "fragment_of".into()],
        );
    }
    (world, parent_id, fragment_ids, solver, whole)
}

/// Same-wave canon ordeal: immediately after birth (no lag, no reload), a
/// gaze at each fragment returns bounds derived from ITS OWN live particle
/// data — not the parent's, not stale, not zero.
#[test]
fn ordeal_oracle_learns_fragments_same_wave() {
    let (world, parent_id, fragment_ids, solver, whole) = build_world_with_fragments();
    let fragments = compute_fragments(&solver, &whole);
    assert_eq!(fragment_ids.len(), fragments.len());

    let parent_geom = world.geometry(&parent_id).expect("parent vessel must gaze");
    assert!(parent_geom.bounds.is_some(), "the parent crate must still have bounds pre-break geometry");

    for (id, fragment) in fragment_ids.iter().zip(fragments.iter()) {
        let geom = world.geometry(id).unwrap_or_else(|| panic!("no gaze reached fragment {id}"));
        let bounds = geom.bounds.unwrap_or_else(|| panic!("fragment {id} has no bounds — the gaze \
            must derive bounds from the fragment's OWN live mesh, not fall through to nothing"));
        // The fragment's own centroid must be the transform origin (its OWN
        // pose, not the parent's) — proves same-wave (no stale parent pose)
        // and no-lag (available immediately, this same Core object).
        let c = fragment.centroid;
        let origin_matches = (geom.origin[0] - c.x as f32).abs() < 1e-4
            && (geom.origin[1] - c.y as f32).abs() < 1e-4
            && (geom.origin[2] - c.z as f32).abs() < 1e-4;
        assert!(
            origin_matches,
            "fragment {id}'s gazed origin {:?} did not match its OWN live centroid {:?} — the \
             fragment's transform must be its own, not the parent's or stale",
            geom.origin, c
        );
        // Bounds must be non-degenerate (this fragment's OWN extent, not a
        // zero-size point that would indicate the gaze fell through to an
        // empty mesh).
        let size = [
            bounds.max[0] - bounds.min[0],
            bounds.max[1] - bounds.min[1],
            bounds.max[2] - bounds.min[2],
        ];
        assert!(
            size.iter().any(|s| *s > 0.0) || fragment.particles.len() == 1,
            "fragment {id} gazed with degenerate (zero-size) bounds {size:?} — expected its own \
             live extent (particle count {})",
            fragment.particles.len()
        );
    }
    println!(
        "ORDEAL oracle-same-wave: {} fragments born and gazed in ONE Core, no reload, each with \
         its own live centroid+extent (parent {parent_id} still gazable too)",
        fragment_ids.len()
    );
}

/// (e) NO ORPHAN MATTER — walk ALL fragment vessels after a break and assert
/// each one's `fragment_of.parent` resolves to the authored crate vessel
/// that broke (not a typo'd id, not missing, not pointing at another
/// fragment).
#[test]
fn ordeal_no_orphan_matter_every_fragment_traces_to_its_parent() {
    let (world, parent_id, fragment_ids, _solver, _whole) = build_world_with_fragments();
    assert!(!fragment_ids.is_empty());

    for id in &fragment_ids {
        let parent_value = world
            .component_value(id, "fragment_of")
            .unwrap_or_else(|| panic!("fragment {id} carries no fragment_of component — orphan matter"));
        let traced_parent = parent_value
            .get("parent")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("fragment {id}'s fragment_of.parent is missing or not a string"));
        assert_eq!(
            traced_parent, parent_id,
            "fragment {id} traces to {traced_parent:?}, not the authored crate vessel {parent_id:?} \
             that actually broke"
        );
        // The traced parent must itself resolve to a real, gazable vessel —
        // not a dangling reference to an id nothing binds.
        assert!(
            world.geometry(&traced_parent).is_some(),
            "fragment {id}'s parent reference {traced_parent:?} does not resolve to any vessel \
             in the live ECS"
        );
    }
    println!(
        "ORDEAL no-orphan-matter: all {} fragments trace to parent {parent_id:?}, which resolves",
        fragment_ids.len()
    );
}
