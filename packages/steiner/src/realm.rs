//! Authored realm spine: superscene + scene documents → deterministic ECS state.

use crate::{SnapshotFrame, StateMap};
use crystal::{
    prefab, ComponentDescriptor, EcsWorld, EntityDoc, FieldSpec, JsonMap, PrefabDoc, SceneSpec,
    WorldMeta,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

/// One scene file + its authored, unexpanded vessel documents.
#[derive(Clone, Debug)]
pub struct SceneDocument {
    pub path: PathBuf,
    pub entities: BTreeMap<String, JsonMap>,
}

/// Disk-authored realm. Runtime state derives from this source.
#[derive(Clone, Debug)]
pub struct RealmScenes {
    root: PathBuf,
    world: WorldMeta,
    scenes: BTreeMap<String, SceneDocument>,
    prefabs: BTreeMap<String, JsonMap>,
}

impl RealmScenes {
    /// Read `world.json` + every declared/present `scenes/<name>.json`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(format!(
                "realm directory does not exist: {}",
                root.display()
            ));
        }

        let world_path = root.join("world.json");
        let mut world = if world_path.is_file() {
            read_json::<WorldMeta>(&world_path)?
        } else {
            WorldMeta::default()
        };
        let scene_dir = root.join("scenes");
        let mut files = json_files(&scene_dir)?;
        let mut names: BTreeSet<String> = world.scenes.keys().cloned().collect();
        for path in &files {
            let name = path
                .file_stem()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("scene filename is not UTF-8: {}", path.display()))?;
            validate_scene_name(name)?;
            names.insert(name.to_owned());
        }
        if names.is_empty() {
            names.insert("main".to_owned());
            files.push(scene_dir.join("main.json"));
        }

        let multiple = names.len() > 1;
        let mut scenes = BTreeMap::new();
        for name in names {
            validate_scene_name(&name)?;
            let path = scene_dir.join(format!("{name}.json"));
            let entities = if path.is_file() {
                read_scene(&path)?
            } else if name == "main" && !world_path.is_file() {
                BTreeMap::new()
            } else {
                return Err(format!(
                    "superscene declares {name:?}, but {} is missing",
                    path.display()
                ));
            };
            world
                .scenes
                .entry(name.clone())
                .or_insert_with(|| SceneSpec {
                    always: multiple.then_some(true),
                    ..SceneSpec::default()
                });
            scenes.insert(name, SceneDocument { path, entities });
        }

        let prefabs = load_prefabs(&root)?;
        let realm = Self {
            root,
            world,
            scenes,
            prefabs,
        };
        realm.snapshot()?;
        Ok(realm)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn world(&self) -> &WorldMeta {
        &self.world
    }

    pub fn scene_names(&self) -> impl Iterator<Item = &str> {
        self.scenes.keys().map(String::as_str)
    }

    pub fn authored_entity_count(&self) -> usize {
        self.scenes.values().map(|scene| scene.entities.len()).sum()
    }

    pub fn scene(&self, name: &str) -> Option<&SceneDocument> {
        self.scenes.get(name)
    }

    /// Expanded runtime base: prefab inheritance + owning-scene stamp.
    pub fn snapshot(&self) -> Result<SnapshotFrame, String> {
        let mut entities = StateMap::new();
        for (scene_name, scene) in &self.scenes {
            for (id, authored) in &scene.entities {
                if entities.contains_key(id) {
                    return Err(format!("duplicate vessel id {id:?}"));
                }
                let expanded = self.expand_doc(scene_name, id, authored)?;
                entities.insert(id.clone(), expanded.into_iter().collect());
            }
        }
        Ok(SnapshotFrame { entities })
    }

    /// Build a crystal ECS projection from the authored snapshot.
    pub fn materialize_into(&self, world: &mut EcsWorld) -> Result<usize, String> {
        materialize_state(&self.snapshot()?.entities, world)
    }

    pub(crate) fn scenes(&self) -> &BTreeMap<String, SceneDocument> {
        &self.scenes
    }

    pub(crate) fn scenes_mut(&mut self) -> &mut BTreeMap<String, SceneDocument> {
        &mut self.scenes
    }

    pub(crate) fn prefabs(&self) -> &BTreeMap<String, JsonMap> {
        &self.prefabs
    }

    pub(crate) fn reload_world(&mut self) -> Result<(), String> {
        let path = self.root.join("world.json");
        if !path.is_file() {
            return Ok(());
        }
        let mut next = read_json::<WorldMeta>(&path)?;
        for name in self.scenes.keys() {
            next.scenes.entry(name.clone()).or_default();
        }
        self.world = next;
        Ok(())
    }

    pub(crate) fn reload_scene(&mut self, name: &str) -> Result<(), String> {
        let scene = self
            .scenes
            .get_mut(name)
            .ok_or_else(|| format!("unknown scene {name:?}"))?;
        if !scene.path.is_file() {
            return Err(format!("scene source is missing: {}", scene.path.display()));
        }
        scene.entities = read_scene(&scene.path)?;
        Ok(())
    }

    pub(crate) fn expand_doc(
        &self,
        scene_name: &str,
        id: &str,
        authored: &JsonMap,
    ) -> Result<BTreeMap<String, Value>, String> {
        let mut own = authored.clone();
        let expanded = match own.remove("prefab") {
            Some(link) => {
                let name = prefab::prefab_name(&link)
                    .ok_or_else(|| format!("vessel {id:?} has an invalid prefab link"))?;
                let base = self
                    .prefabs
                    .get(&name)
                    .ok_or_else(|| format!("vessel {id:?} references unknown prefab {name:?}"))?;
                prefab::expand_instance(&name, base, &own)
            }
            None => own,
        };
        let mut expanded: BTreeMap<String, Value> = expanded.into_iter().collect();
        expanded
            .entry("scene".to_owned())
            .or_insert_with(|| json!({ "name": scene_name }));
        Ok(expanded)
    }
}

/// Materialize canonical state into an empty/compatible crystal ECS.
pub fn materialize_state(state: &StateMap, world: &mut EcsWorld) -> Result<usize, String> {
    let names: BTreeSet<String> = state
        .values()
        .flat_map(|components| components.keys().cloned())
        .collect();
    for name in names {
        if world.component_id(&name).is_none() {
            world.register_component(ComponentDescriptor {
                name,
                fields: BTreeMap::<String, FieldSpec>::new(),
                enableable: false,
                buffer: true,
                default: None,
            })?;
        }
    }
    for (id, components) in state {
        let values = components
            .iter()
            .map(|(name, value)| {
                world
                    .component_id(name)
                    .map(|component| (component, value.clone()))
                    .ok_or_else(|| format!("component {name:?} was not registered"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let entity = world.create_entity(values)?;
        world.bind_gaia_id(id, entity)?;
    }
    Ok(state.len())
}

fn read_scene(path: &Path) -> Result<BTreeMap<String, JsonMap>, String> {
    let raw = read_json::<BTreeMap<String, Value>>(path)?;
    raw.into_iter()
        .map(|(id, value)| {
            serde_json::from_value::<EntityDoc>(value.clone())
                .map_err(|error| format!("scene {} vessel {id:?}: {error}", path.display()))?;
            let object = value.as_object().cloned().ok_or_else(|| {
                format!("scene {} vessel {id:?} is not an object", path.display())
            })?;
            Ok((id, object))
        })
        .collect()
}

fn load_prefabs(root: &Path) -> Result<BTreeMap<String, JsonMap>, String> {
    let mut prefabs = BTreeMap::new();
    let legacy = root.join("prefabs.json");
    if legacy.is_file() {
        for prefab in read_json::<Vec<PrefabDoc>>(&legacy)? {
            prefabs.insert(prefab.name.clone(), entity_map(prefab.components)?);
        }
    }
    for path in json_files(&root.join("prefabs"))? {
        let prefab = read_json::<PrefabDoc>(&path)?;
        prefabs.insert(prefab.name.clone(), entity_map(prefab.components)?);
    }
    Ok(prefabs)
}

fn entity_map(doc: EntityDoc) -> Result<JsonMap, String> {
    serde_json::to_value(doc)
        .map_err(|error| error.to_string())?
        .as_object()
        .cloned()
        .ok_or_else(|| "prefab components are not an object".to_owned())
}

fn json_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = fs::read_dir(dir)
        .map_err(|error| format!("read {}: {error}", dir.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    files.sort();
    Ok(files)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str(&source).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn validate_scene_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || Path::new(name)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("invalid scene name {name:?}"));
    }
    Ok(())
}
