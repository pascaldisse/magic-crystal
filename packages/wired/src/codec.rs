//! Pure protocol layer — no I/O. Builds outbound wire messages and folds
//! inbound ones into a crystal-compatible entity map. Unit-tested standalone.
//!
//! # Wire protocol (recon: server/index.js + client/kernel/net.js)
//! - Transport: one WebSocket, JSON text frames, `{ "type": <kind>, ... }`.
//! - HANDSHAKE — server pushes `snapshot` the instant the socket opens
//!   (`wss.on('connection')`); client answers `{type:'hello', presence}` to
//!   BIND its presence id to the socket. Presence id is CLIENT-chosen
//!   (`player-<clientId>`), never server-assigned; hello only tells the
//!   server which entity to reap on `close`.
//! - `snapshot` (S→C, once): `{time, world, game, materials, counter,
//!   entities}` — `entities` is the full `EntityMap` (id → components).
//! - `ops` (both ways): `{type:'ops', ops:[...], from, dev?}`. Client sends
//!   spawn/merge/set/despawn/event; server NORMALIZES and re-broadcasts —
//!   merge arrives back as `set` with the FULL merged value, so a client
//!   only ever folds spawn/set/despawn/clear/event.
//! - `dev:true` ops write through to the scene files (authoring); gameplay
//!   traffic (presence moves, use, weather) stays plain.
//! - `screenshot-request` / `screenshot`: renderer relay — ignored here.
//! - PRESENCE LIFECYCLE — client spawns its own presence entity via a spawn
//!   op (`{presence:{kind,yaw}, transform:{position}}`); moves are two merge
//!   ops (transform.position + presence.yaw). Server re-stamps the presence's
//!   `scene` as it crosses bounds. On socket close the server despawns the
//!   hello'd presence — clean disconnect reaps it.
//! - Event journal: `event` ops are transient (broadcast + `GET /events`,
//!   never persisted); folded here into the returned event list.

use crystal::{
    EntityDoc, EntityMap, HelloMessage, MaterialLibrary, Op, SetOp, WorldMeta, WsMessage,
    WsOpsMessage,
};
use serde_json::{json, Map, Number, Value};

/// The client's live mirror of world state, rebuilt from snapshot + ops.
#[derive(Clone, Debug, Default)]
pub struct WorldView {
    /// World clock seconds at last snapshot.
    pub time: f64,
    /// id → typed entity document (crystal `EntityMap`).
    pub entities: EntityMap,
    /// Superscene composition from the snapshot, if any.
    pub world: Option<WorldMeta>,
    /// Named material library from the snapshot, if any.
    pub materials: Option<MaterialLibrary>,
    /// Presence id this client bound via `hello`, once known.
    pub presence: Option<String>,
}

impl WorldView {
    /// Fold one inbound message into the view. Returns any `event` ops it
    /// carried (transient — never stored on entities).
    pub fn apply(&mut self, message: &WsMessage) -> Vec<Value> {
        match message {
            WsMessage::Snapshot(snap) => {
                self.time = snap.time.as_f64().unwrap_or(self.time);
                self.entities = snap.entities.clone();
                if snap.world.is_some() {
                    self.world = snap.world.clone();
                }
                if snap.materials.is_some() {
                    self.materials = snap.materials.clone();
                }
                Vec::new()
            }
            WsMessage::Ops(ops) => {
                let mut events = Vec::new();
                for op in &ops.ops {
                    if let Some(event) = apply_op(&mut self.entities, op) {
                        events.push(event);
                    }
                }
                events
            }
            WsMessage::Hello(HelloMessage { presence, .. }) => {
                self.presence = Some(presence.clone());
                Vec::new()
            }
            _ => Vec::new(),
        }
    }
}

/// Fold one op into an entity map. Returns the event payload for `event` ops.
pub fn apply_op(entities: &mut EntityMap, op: &Op) -> Option<Value> {
    match op {
        Op::Set(SetOp {
            id,
            component,
            value,
            ..
        }) => {
            let doc = entities.entry(id.clone()).or_default();
            set_component(doc, component, value);
            None
        }
        Op::Other { op, fields } => match op.as_str() {
            "spawn" => {
                let id = fields.get("id").and_then(Value::as_str)?.to_string();
                let comps = fields
                    .get("components")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new()));
                if let Ok(doc) = serde_json::from_value::<EntityDoc>(comps) {
                    entities.insert(id, doc);
                }
                None
            }
            "despawn" => {
                if let Some(id) = fields.get("id").and_then(Value::as_str) {
                    entities.remove(id);
                }
                None
            }
            "clear" => {
                entities.clear();
                None
            }
            "event" => Some(Value::Object(fields.clone())),
            _ => None,
        },
        _ => None,
    }
}

/// Write (or, on null, clear) one component on a typed doc, routing through
/// JSON so unknown components survive in `extra`.
fn set_component(doc: &mut EntityDoc, component: &str, value: &Value) {
    let mut object = match serde_json::to_value(&*doc) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    };
    if value.is_null() {
        object.remove(component);
    } else {
        object.insert(component.to_string(), value.clone());
    }
    if let Ok(next) = serde_json::from_value::<EntityDoc>(Value::Object(object)) {
        *doc = next;
    }
}

// ---- outbound message builders ----

/// `{type:'hello', presence}` — bind this socket's presence for reaping.
pub fn hello(presence: impl Into<String>) -> WsMessage {
    WsMessage::Hello(HelloMessage {
        presence: presence.into(),
        extra: Map::new(),
    })
}

/// `{type:'ops', ops, from, dev}` batch.
pub fn ops_batch(ops: Vec<Op>, from: impl Into<String>, dev: bool) -> WsMessage {
    WsMessage::Ops(WsOpsMessage {
        ops,
        from: Some(from.into()),
        dev,
        extra: Map::new(),
    })
}

fn num(value: f64) -> Number {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}

fn vec3(position: [f64; 3]) -> Vec<Number> {
    position.iter().map(|v| num(*v)).collect()
}

/// Build a `spawn` op for a player presence entity.
pub fn spawn_presence_op(id: &str, position: [f64; 3], yaw: f64) -> Op {
    let mut fields = Map::new();
    fields.insert("id".into(), Value::String(id.to_string()));
    fields.insert(
        "components".into(),
        json!({
            "presence": { "kind": "player", "yaw": yaw },
            "transform": { "position": vec3(position) },
        }),
    );
    Op::Other {
        op: "spawn".into(),
        fields,
    }
}

/// Build the two merge ops a moving presence sends (position + facing).
pub fn move_presence_ops(id: &str, position: [f64; 3], yaw: f64) -> Vec<Op> {
    vec![
        merge_op(id, "transform", json!({ "position": vec3(position) })),
        merge_op(id, "presence", json!({ "yaw": yaw })),
    ]
}

/// Build a `merge` op (shallow component merge).
pub fn merge_op(id: &str, component: &str, value: Value) -> Op {
    let mut fields = Map::new();
    fields.insert("id".into(), Value::String(id.to_string()));
    fields.insert("component".into(), Value::String(component.to_string()));
    fields.insert("value".into(), value);
    Op::Other {
        op: "merge".into(),
        fields,
    }
}

/// Build a `set` op (replace a whole component; null clears it).
pub fn set_op(id: &str, component: &str, value: Value) -> Op {
    Op::Set(SetOp {
        id: id.to_string(),
        component: component.to_string(),
        value,
        extra: Map::new(),
    })
}

/// Build a `despawn` op.
pub fn despawn_op(id: &str) -> Op {
    let mut fields = Map::new();
    fields.insert("id".into(), Value::String(id.to_string()));
    Op::Other {
        op: "despawn".into(),
        fields,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_snapshot_then_ops() {
        let mut view = WorldView::default();
        let snapshot: WsMessage = serde_json::from_value(json!({
            "type": "snapshot",
            "time": 4.5,
            "entities": { "rock": { "transform": { "position": [1, 0, 2] } } },
            "world": { "scenes": { "main": {} } },
            "materials": { "obsidian": { "color": "#000" } },
        }))
        .unwrap();
        assert!(view.apply(&snapshot).is_empty());
        assert_eq!(view.time, 4.5);
        assert!(view.entities.contains_key("rock"));
        assert!(view.world.is_some());
        assert!(view.materials.is_some());

        // spawn arrives as an ops batch
        let spawn = ops_batch(
            vec![spawn_presence_op("p1", [3.0, 2.0, 1.0], 0.5)],
            "tester",
            false,
        );
        view.apply(&spawn);
        let p1 = view.entities.get("p1").unwrap();
        assert!(p1.presence.is_some());
        assert_eq!(
            p1.transform.as_ref().unwrap().position.as_ref().unwrap()[0]
                .as_f64()
                .unwrap(),
            3.0
        );

        // server normalizes merge → set with the full merged value
        let moved: WsMessage = serde_json::from_value(json!({
            "type": "ops",
            "ops": [{ "op": "set", "id": "p1", "component": "transform", "value": { "position": [9, 2, 9] } }],
        }))
        .unwrap();
        view.apply(&moved);
        assert_eq!(
            view.entities["p1"]
                .transform
                .as_ref()
                .unwrap()
                .position
                .as_ref()
                .unwrap()[0]
                .as_f64()
                .unwrap(),
            9.0
        );
    }

    #[test]
    fn set_null_clears_component_and_unknown_components_survive() {
        let mut entities = EntityMap::new();
        apply_op(
            &mut entities,
            &Op::Other {
                op: "spawn".into(),
                fields: {
                    let mut f = Map::new();
                    f.insert("id".into(), json!("e1"));
                    f.insert(
                        "components".into(),
                        json!({ "health": { "hp": 10 }, "quirk": { "x": 1 } }),
                    );
                    f
                },
            },
        );
        assert!(entities["e1"].health.is_some());
        assert!(entities["e1"].extra.contains_key("quirk"));

        apply_op(&mut entities, &set_op("e1", "health", Value::Null));
        assert!(entities["e1"].health.is_none());
        // unknown component still there
        assert!(entities["e1"].extra.contains_key("quirk"));
    }

    #[test]
    fn despawn_and_event_and_clear() {
        let mut entities = EntityMap::new();
        apply_op(&mut entities, &spawn_op("e1"));
        assert!(entities.contains_key("e1"));

        let event = apply_op(
            &mut entities,
            &Op::Other {
                op: "event".into(),
                fields: {
                    let mut f = Map::new();
                    f.insert("name".into(), json!("lightning"));
                    f
                },
            },
        );
        assert_eq!(event.unwrap()["name"], json!("lightning"));

        apply_op(&mut entities, &despawn_op("e1"));
        assert!(!entities.contains_key("e1"));

        apply_op(&mut entities, &spawn_op("e2"));
        apply_op(
            &mut entities,
            &Op::Other {
                op: "clear".into(),
                fields: Map::new(),
            },
        );
        assert!(entities.is_empty());
    }

    #[test]
    fn hello_and_ops_batch_roundtrip_wire_shape() {
        let hello = hello("player-c1");
        assert_eq!(
            serde_json::to_value(&hello).unwrap(),
            json!({ "type": "hello", "presence": "player-c1" })
        );
        let batch = ops_batch(vec![despawn_op("x")], "cid", true);
        assert_eq!(
            serde_json::to_value(&batch).unwrap(),
            json!({ "type": "ops", "ops": [{ "op": "despawn", "id": "x" }], "from": "cid", "dev": true })
        );
    }

    #[test]
    fn move_presence_emits_position_and_yaw() {
        let ops = move_presence_ops("p1", [1.0, 2.0, 3.0], 1.5);
        assert_eq!(ops.len(), 2);
        assert_eq!(
            serde_json::to_value(&ops[0]).unwrap(),
            json!({ "op": "merge", "id": "p1", "component": "transform", "value": { "position": [1.0, 2.0, 3.0] } })
        );
        assert_eq!(
            serde_json::to_value(&ops[1]).unwrap(),
            json!({ "op": "merge", "id": "p1", "component": "presence", "value": { "yaw": 1.5 } })
        );
    }

    fn spawn_op(id: &str) -> Op {
        let mut fields = Map::new();
        fields.insert("id".into(), Value::String(id.to_string()));
        fields.insert("components".into(), json!({ "state": {} }));
        Op::Other {
            op: "spawn".into(),
            fields,
        }
    }
}
