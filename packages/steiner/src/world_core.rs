//! Native authority: incantation door + authored persistence + Steiner truth.

use crate::{materialize_state, read_journal, RealmScenes, Recorder};
use crystal::{prefab, EcsWorld, JsonMap, Op, OpBatch, SetOp};
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

/// Authority dials. No policy value hides in the apply path.
#[derive(Clone, Debug)]
pub struct WorldCoreParams {
    pub seed: u64,
    pub event_capacity: usize,
    pub runtime_fields: BTreeMap<String, BTreeSet<String>>,
}

impl Default for WorldCoreParams {
    fn default() -> Self {
        Self {
            seed: 0,
            event_capacity: 2_000,
            runtime_fields: BTreeMap::from([(
                "weather".to_owned(),
                BTreeSet::from(["rain".to_owned()]),
            )]),
        }
    }
}

/// One authority result. `applied` = normalized replay operations.
#[derive(Clone, Debug)]
pub struct AppliedBatch {
    pub applied: Vec<Op>,
    pub entropy: u64,
    pub latest: u64,
}

/// Realm authority. Recorder ECS = live truth; scene documents = authored truth.
pub struct WorldCore {
    params: WorldCoreParams,
    realm: RealmScenes,
    recorder: Recorder,
    entropy: u64,
    sequence: u64,
    events: VecDeque<Value>,
}

impl WorldCore {
    pub fn open(root: impl AsRef<Path>, params: WorldCoreParams) -> Result<Self, String> {
        if params.event_capacity == 0 {
            return Err("event_capacity must be positive".to_owned());
        }
        let realm = RealmScenes::open(root)?;
        let snapshot = realm.snapshot()?;
        let recorder = Recorder::from_snapshot(params.seed, snapshot)
            .map_err(|error| format!("open Steiner worldline: {error}"))?;
        Ok(Self {
            params,
            realm,
            recorder,
            entropy: 0,
            sequence: 0,
            events: VecDeque::new(),
        })
    }

    pub fn realm(&self) -> &RealmScenes {
        &self.realm
    }

    pub fn entropy(&self) -> u64 {
        self.entropy
    }

    pub fn latest_sequence(&self) -> u64 {
        self.sequence
    }

    pub fn journal_bytes(&self) -> &[u8] {
        self.recorder.journal_bytes()
    }

    pub fn state(&self) -> crate::StateMap {
        self.recorder.state_map()
    }

    pub fn component(&self, id: &str, component: &str) -> Option<Value> {
        self.recorder
            .state_map()
            .get(id)
            .and_then(|components| components.get(component))
            .cloned()
    }

    pub fn materialize_into(&self, world: &mut EcsWorld) -> Result<usize, String> {
        materialize_state(&self.recorder.state_map(), world)
    }

    /// Apply one ordered batch. Reset expands to concrete replay operations.
    pub fn apply(&mut self, batch: OpBatch) -> Result<AppliedBatch, String> {
        let mut candidate = self.recorder.state_map();
        let mut applied = Vec::new();
        for op in &batch.ops {
            match op {
                Op::Set(set) => {
                    let Some(components) = candidate.get_mut(&set.id) else {
                        continue;
                    };
                    if set.value.is_null() {
                        components.remove(&set.component);
                    } else {
                        components.insert(set.component.clone(), set.value.clone());
                    }
                    applied.push(Op::Set(set.clone()));
                }
                Op::Reset(reset) => {
                    self.expand_reset(reset.scene.as_deref(), &mut candidate, &mut applied)?;
                }
                _ => {}
            }
        }

        if applied.is_empty() {
            return Ok(AppliedBatch {
                applied,
                entropy: self.entropy,
                latest: self.sequence,
            });
        }

        self.entropy = self
            .entropy
            .checked_add(1)
            .ok_or_else(|| "entropy overflow".to_owned())?;
        let normalized = OpBatch {
            dev: false,
            ops: applied.clone(),
            from: batch.from.clone(),
            extra: Map::new(),
        };
        self.recorder
            .record(&normalized, self.entropy)
            .map_err(|error| format!("record Steiner frame: {error}"))?;
        if self.recorder.state_map() != candidate {
            return Err("authority/replay state divergence".to_owned());
        }

        self.record_events(&applied, batch.from.as_deref().unwrap_or("http"));
        if batch.dev {
            self.write_back_sets(&batch.ops)?;
        }
        Ok(AppliedBatch {
            applied,
            entropy: self.entropy,
            latest: self.sequence,
        })
    }

    /// Runtime snapshot for HTTP clients.
    pub fn snapshot_json(&self) -> Result<Value, String> {
        let entities = self
            .recorder
            .state_map()
            .into_iter()
            .map(|(id, components)| {
                let object: Map<String, Value> = components.into_iter().collect();
                (id, Value::Object(object))
            })
            .collect::<Map<String, Value>>();
        Ok(json!({
            "counter": 1,
            "entities": entities,
            "world": serde_json::to_value(self.realm.world()).map_err(|error| error.to_string())?,
            "entropy": self.entropy,
        }))
    }

    /// Bounded event view; Steiner remains the replay ledger underneath.
    pub fn events_json(&self, since: u64, limit: usize) -> Value {
        let events: Vec<Value> = self
            .events
            .iter()
            .filter(|event| event.get("seq").and_then(Value::as_u64).unwrap_or(0) > since)
            .rev()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        json!({ "latest": self.sequence, "entropy": self.entropy, "events": events })
    }

    /// Ledger self-check used by ordeals/live diagnostics.
    pub fn journal_frame_count(&self) -> Result<usize, String> {
        read_journal(self.journal_bytes())
            .map(|journal| journal.entries.len())
            .map_err(|error| error.to_string())
    }

    fn expand_reset(
        &mut self,
        selected: Option<&str>,
        candidate: &mut crate::StateMap,
        applied: &mut Vec<Op>,
    ) -> Result<(), String> {
        self.realm.reload_world()?;
        let names: Vec<String> = match selected {
            Some(name) if self.realm.scene(name).is_some() => vec![name.to_owned()],
            Some(name) => return Err(format!("unknown scene {name:?}")),
            None => self.realm.scene_names().map(str::to_owned).collect(),
        };
        applied.push(event_op("reset", json!({ "scene": selected })));

        for name in names {
            self.realm.reload_scene(&name)?;
            let residents: Vec<String> = candidate
                .iter()
                .filter(|(_, components)| {
                    !components.contains_key("persist")
                        && !components.contains_key("presence")
                        && components
                            .get("scene")
                            .and_then(|scene| scene.get("name"))
                            .and_then(Value::as_str)
                            == Some(name.as_str())
                })
                .map(|(id, _)| id.clone())
                .collect();
            for id in residents {
                candidate.remove(&id);
                applied.push(other_op("despawn", json!({ "id": id }))?);
            }

            let authored = self
                .realm
                .scene(&name)
                .expect("selected scene exists")
                .entities
                .clone();
            for (id, doc) in authored {
                if candidate
                    .get(&id)
                    .is_some_and(|components| components.contains_key("persist"))
                {
                    continue;
                }
                let components = self.realm.expand_doc(&name, &id, &doc)?;
                candidate.insert(id.clone(), components.clone());
                applied.push(other_op(
                    "spawn",
                    json!({ "id": id, "components": components }),
                )?);
            }
        }
        Ok(())
    }

    fn record_events(&mut self, applied: &[Op], from: &str) {
        let wall_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        for op in applied {
            self.sequence += 1;
            let mut event = serde_json::to_value(op)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            event.insert("seq".to_owned(), json!(self.sequence));
            event.insert("t".to_owned(), json!(wall_millis));
            event.insert("entropy".to_owned(), json!(self.entropy));
            event.insert("from".to_owned(), json!(from));
            self.events.push_back(Value::Object(event));
        }
        while self.events.len() > self.params.event_capacity {
            self.events.pop_front();
        }
    }

    fn write_back_sets(&mut self, input: &[Op]) -> Result<(), String> {
        let state = self.recorder.state_map();
        let mut dirty = BTreeSet::new();
        for op in input {
            let Op::Set(SetOp { id, .. }) = op else {
                continue;
            };
            let Some(components) = state.get(id) else {
                continue;
            };
            if components.contains_key("presence") || components.contains_key("persist") {
                continue;
            }
            let Some(scene_name) = components
                .get("scene")
                .and_then(|scene| scene.get("name"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            if self.realm.scene(scene_name).is_none() {
                continue;
            }
            let doc = self.scene_doc(components);
            let scene = self
                .realm
                .scenes_mut()
                .get_mut(scene_name)
                .expect("scene checked above");
            if scene.entities.get(id) != Some(&doc) {
                scene.entities.insert(id.clone(), doc);
                dirty.insert(scene_name.to_owned());
            }
        }
        for scene_name in dirty {
            let scene = self
                .realm
                .scene(&scene_name)
                .expect("dirty scene remains present");
            let mut bytes = serde_json::to_vec_pretty(&scene.entities)
                .map_err(|error| format!("serialize scene {scene_name:?}: {error}"))?;
            bytes.push(b'\n');
            if let Some(parent) = scene.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("create {}: {error}", parent.display()))?;
            }
            fs::write(&scene.path, bytes)
                .map_err(|error| format!("write {}: {error}", scene.path.display()))?;
        }
        Ok(())
    }

    fn scene_doc(&self, components: &BTreeMap<String, Value>) -> JsonMap {
        let mut doc: JsonMap = components.clone().into_iter().collect();
        doc.remove("scene");
        for (component, fields) in &self.params.runtime_fields {
            let Some(Value::Object(value)) = doc.get_mut(component) else {
                continue;
            };
            for field in fields {
                value.remove(field);
            }
        }
        let Some(name) = doc.get("prefab").and_then(prefab::prefab_name) else {
            return doc;
        };
        let Some(base) = self.realm.prefabs().get(&name) else {
            return doc;
        };
        prefab::diff_instance(&name, base, &doc)
    }
}

/// Accept parity forms: one op, op array, or `{ops,from,dev}`.
pub fn parse_op_batch(value: Value) -> Result<OpBatch, String> {
    match value {
        Value::Array(ops) => Ok(OpBatch {
            ops: ops
                .into_iter()
                .map(serde_json::from_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.to_string())?,
            ..OpBatch::default()
        }),
        Value::Object(ref object) if object.contains_key("ops") => {
            serde_json::from_value(value).map_err(|error| error.to_string())
        }
        other => Ok(OpBatch {
            ops: vec![serde_json::from_value(other).map_err(|error| error.to_string())?],
            ..OpBatch::default()
        }),
    }
}

fn event_op(name: &str, data: Value) -> Op {
    Op::Other {
        op: "event".to_owned(),
        fields: Map::from_iter([
            ("name".to_owned(), Value::String(name.to_owned())),
            ("data".to_owned(), data),
        ]),
    }
}

fn other_op(name: &str, fields: Value) -> Result<Op, String> {
    let fields = fields
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{name} fields are not an object"))?;
    Ok(Op::Other {
        op: name.to_owned(),
        fields,
    })
}
