use crystal::{load_world_dir, EcsWorld, QuerySpec};
use std::path::{Path, PathBuf};

fn naruko_world() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko")
}

fn twin_world() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/twin-world")
}

#[test]
fn world_json_composes_two_scenes_into_one_ecs() {
    let path = twin_world();
    assert!(
        path.join("world.json").exists(),
        "twin-world must exercise world.json-driven loading"
    );

    let mut ecs = EcsWorld::default();
    let loaded = load_world_dir(&path, &mut ecs).expect("load twin-world through world.json");

    // world.json scene map ordering (BTreeMap keys) is deterministic.
    assert_eq!(loaded.scenes, ["a", "b"]);
    assert!(loaded.meta.is_some(), "world.json parsed into WorldMeta");
    // Both scenes merged: a_env, a_spawn, a_box (scene a) + b_pillar (scene b).
    assert_eq!(loaded.entity_count, 4);
    assert_eq!(ecs.query(&QuerySpec::default()).len(), 4);
    assert!(ecs.entity_for_gaia("a_box").is_some());
    assert!(ecs.entity_for_gaia("b_pillar").is_some());

    let transform = ecs.component_id("transform").unwrap();
    let mesh = ecs.component_id("mesh").unwrap();
    assert_eq!(
        ecs.query(&QuerySpec {
            all: vec![transform, mesh],
            ..Default::default()
        })
        .len(),
        2,
        "one mesh entity per scene survives the merge"
    );
}

/// The realm document is the source of truth: expectations are READ from the
/// scene file, never frozen into the ordeal — the canonical realm grows rite
/// by rite, and this gate must discriminate loader faults, not realm growth.
#[test]
fn naruko_blank_page_parses_and_populates_expected_entities() {
    let path = naruko_world();
    assert!(
        !path.join("world.json").exists(),
        "Naruko must exercise blank-page loading"
    );

    let doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path.join("scenes/main.json"))
            .expect("read the canonical Naruko scene document"),
    )
    .expect("scene document parses as JSON");
    let entities = doc
        .as_object()
        .expect("a scene document is an entity map keyed by id");
    let expected_total = entities.len();
    let expected_meshed = entities
        .values()
        .filter(|entity| entity.get("transform").is_some() && entity.get("mesh").is_some())
        .count();
    assert!(expected_total > 0, "the canonical realm must not be empty");

    let mut ecs = EcsWorld::default();
    let loaded = load_world_dir(&path, &mut ecs).expect("load Naruko through GAIA protocol");

    assert_eq!(loaded.scenes, ["main"]);
    assert_eq!(loaded.entity_count, expected_total);
    assert_eq!(ecs.query(&QuerySpec::default()).len(), expected_total);
    for id in ["env", "world_spawn", "lighthouse_tower"] {
        assert!(
            ecs.entity_for_gaia(id).is_some(),
            "the canonical realm must carry `{id}`"
        );
    }

    let transform = ecs.component_id("transform").unwrap();
    let mesh = ecs.component_id("mesh").unwrap();
    assert_eq!(
        ecs.query(&QuerySpec {
            all: vec![transform, mesh],
            ..Default::default()
        })
        .len(),
        expected_meshed
    );
}
