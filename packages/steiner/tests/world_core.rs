use crystal::EcsWorld;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use steiner::RealmScenes;

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

impl Drop for TestRealm {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
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
