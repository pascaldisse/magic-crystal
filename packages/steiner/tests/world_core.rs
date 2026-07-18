use crystal::{EcsWorld, Op, OpBatch, SetOp};
use serde_json::{Map, Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use steiner::{RealmScenes, Recorder, WorldCore, WorldCoreParams, parse_op_batch, read_journal};

static NEXT_REALM: AtomicU64 = AtomicU64::new(1);

struct TestRealm {
    root: PathBuf,
}

impl TestRealm {
    fn two_scene() -> Self {
        let serial = NEXT_REALM.fetch_add(1, Ordering::Relaxed);
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/world-core-ordeals")
            .join(format!("{}-{serial}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("scenes")).unwrap();
        fs::write(
            root.join("world.json"),
            serde_json::to_vec_pretty(&json!({
                "voidY": -40,
                "scenes": {
                    "a": { "neighbors": ["b"] },
                    "b": { "always": true }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("scenes/a.json"),
            serde_json::to_vec_pretty(&json!({
                "box": {
                    "transform": { "position": [0, 1, 0] },
                    "mesh": { "parts": [{ "shape": "box", "size": [2, 2, 2], "color": "#aa0000" }] }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("scenes/b.json"),
            serde_json::to_vec_pretty(&json!({
                "pillar": {
                    "transform": { "position": [8, 2, 0] },
                    "mesh": { "parts": [{ "shape": "cylinder", "radius": 1, "height": 4, "color": "#2244aa" }] }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        Self { root }
    }
}

impl TestRealm {
    fn scene_text(&self, name: &str) -> String {
        fs::read_to_string(self.root.join("scenes").join(format!("{name}.json"))).unwrap()
    }
}

impl Drop for TestRealm {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn params() -> WorldCoreParams {
    WorldCoreParams {
        seed: 0x5eed,
        event_capacity: 32,
        ..WorldCoreParams::default()
    }
}

fn set_mesh(color: &str, dev: bool) -> OpBatch {
    OpBatch {
        dev,
        from: Some("ordeal".to_owned()),
        ops: vec![Op::Set(SetOp {
            id: "box".to_owned(),
            component: "mesh".to_owned(),
            value: json!({
                "parts": [{ "shape": "box", "size": [2, 2, 2], "color": color }]
            }),
            extra: Map::new(),
        })],
        extra: Map::new(),
    }
}

#[test]
fn boot_from_superscene_materializes_every_scene_into_crystal() {
    let fixture = TestRealm::two_scene();
    let realm = RealmScenes::open(&fixture.root).expect("open authored realm");

    assert_eq!(realm.scene_names().collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(realm.authored_entity_count(), 2);
    assert_eq!(realm.world().extra["voidY"], json!(-40));

    let snapshot = realm.snapshot().expect("expand authored truth");
    assert_eq!(snapshot.entities["box"]["scene"], json!({ "name": "a" }));
    assert_eq!(snapshot.entities["pillar"]["scene"], json!({ "name": "b" }));

    let mut ecs = EcsWorld::default();
    assert_eq!(realm.materialize_into(&mut ecs).unwrap(), 2);
    assert!(ecs.entity_for_gaia("box").is_some());
    assert!(ecs.entity_for_gaia("pillar").is_some());
}

#[test]
fn http_door_batch_shape_preserves_dev_set_and_reset() {
    let batch = parse_op_batch(json!({
        "dev": true,
        "from": "curl",
        "ops": [
            {
                "op": "set",
                "id": "box",
                "component": "mesh",
                "value": { "parts": [{ "shape": "box", "color": "#00aa00" }] }
            },
            { "op": "reset", "scene": "a" }
        ]
    }))
    .expect("parse POST /op contract");

    assert!(batch.dev);
    assert_eq!(batch.from.as_deref(), Some("curl"));
    assert!(matches!(&batch.ops[0], Op::Set(set) if set.id == "box"));
    assert!(matches!(&batch.ops[1], Op::Reset(reset) if reset.scene.as_deref() == Some("a")));
}

#[test]
fn runtime_set_mutates_live_crystal_without_touching_disk() {
    let fixture = TestRealm::two_scene();
    let before = fixture.scene_text("a");
    let mut core = WorldCore::open(&fixture.root, params()).unwrap();

    let report = core.apply(set_mesh("#00aa00", false)).unwrap();

    assert_eq!(report.applied.len(), 1);
    assert_eq!(report.entropy, 1);
    assert_eq!(
        core.component("box", "mesh").unwrap()["parts"][0]["color"],
        json!("#00aa00")
    );
    assert_eq!(
        fixture.scene_text("a"),
        before,
        "runtime batch stays live-only"
    );
}

#[test]
fn set_lands_in_steiner_and_replays_to_identical_state() {
    let fixture = TestRealm::two_scene();
    let mut core = WorldCore::open(&fixture.root, params()).unwrap();
    core.apply(set_mesh("#00aa00", false)).unwrap();

    let decoded = read_journal(core.journal_bytes()).expect("decode authority ledger");
    assert_eq!(decoded.entries.len(), 1);
    assert_eq!(decoded.entries[0].tick, 1);
    assert!(matches!(&decoded.entries[0].ops[0], Op::Set(set) if set.id == "box"));
    let (replayed, _) = Recorder::replay(core.journal_bytes(), None).unwrap();
    assert_eq!(replayed.state_map(), core.state());

    let events = core.events_json(0, 32);
    assert_eq!(events["latest"], json!(1));
    assert_eq!(events["events"][0]["op"], json!("set"));
    assert_eq!(events["events"][0]["entropy"], json!(1));
}

#[test]
fn reset_rereads_scene_and_restores_disk_truth() {
    let fixture = TestRealm::two_scene();
    let mut core = WorldCore::open(&fixture.root, params()).unwrap();
    core.apply(set_mesh("#00aa00", false)).unwrap();
    assert_eq!(
        core.component("box", "mesh").unwrap()["parts"][0]["color"],
        json!("#00aa00")
    );

    let report = core
        .apply(OpBatch {
            from: Some("ordeal".to_owned()),
            ops: vec![serde_json::from_value(json!({ "op": "reset", "scene": "a" })).unwrap()],
            ..OpBatch::default()
        })
        .unwrap();

    assert!(
        report
            .applied
            .iter()
            .any(|op| matches!(op, Op::Other { op, .. } if op == "event"))
    );
    assert!(
        report
            .applied
            .iter()
            .any(|op| matches!(op, Op::Other { op, .. } if op == "despawn"))
    );
    assert!(
        report
            .applied
            .iter()
            .any(|op| matches!(op, Op::Other { op, .. } if op == "spawn"))
    );
    assert_eq!(
        core.component("box", "mesh").unwrap()["parts"][0]["color"],
        json!("#aa0000")
    );
    let (replayed, _) = Recorder::replay(core.journal_bytes(), None).unwrap();
    assert_eq!(
        replayed.state_map(),
        core.state(),
        "reset expansion is replay truth"
    );
}

#[test]
fn dev_noop_keeps_authored_bytes_stable() {
    let fixture = TestRealm::two_scene();
    let before = fixture.scene_text("a");
    let mut core = WorldCore::open(&fixture.root, params()).unwrap();

    core.apply(set_mesh("#aa0000", true)).unwrap();

    assert_eq!(fixture.scene_text("a"), before);
}

#[test]
fn dev_set_persists_diff_to_owning_scene_only() {
    let fixture = TestRealm::two_scene();
    let scene_b_before = fixture.scene_text("b");
    let mut core = WorldCore::open(&fixture.root, params()).unwrap();

    core.apply(set_mesh("#0066ff", true)).unwrap();

    let authored: Value = serde_json::from_str(&fixture.scene_text("a")).unwrap();
    assert_eq!(
        authored["box"]["mesh"]["parts"][0]["color"],
        json!("#0066ff")
    );
    assert!(
        authored["box"].get("scene").is_none(),
        "runtime claim stays out of authored data"
    );
    assert_eq!(
        fixture.scene_text("b"),
        scene_b_before,
        "unrelated scene is byte-stable"
    );
}
