//! Prefab deep-merge + diff-on-write — the reference GAIA engine's instance
//! semantics (`server/index.js` `deepMerge` / `expandDoc` / `sceneDoc`).
//!
//! A scene entity with a `prefab` key is an INSTANCE: the prefab's components
//! (`world/prefabs/<name>.json`) deep-merged UNDER the entry's own components
//! (the entry stores only its deltas). Write-back diffs the live components
//! against the prefab so an instance stays tiny — a moved torch is three lines.
//!
//! IRON LAW (never special-case): these are pure JSON operations; the core owns
//! no game vocabulary, so any component merges the same way.

use crate::JsonMap;
use serde_json::{json, Value};

/// Deep-merge `over` onto `base`. Two objects merge recursively, key by key;
/// arrays and scalars in `over` REPLACE the base wholesale (reference
/// `deepMerge`: only plain objects recurse). `base`/`over` are unchanged.
pub fn deep_merge(base: &Value, over: &Value) -> Value {
    match (base, over) {
        (Value::Object(base), Value::Object(over)) => {
            let mut out = base.clone();
            for (key, value) in over {
                let merged = match out.get(key) {
                    Some(existing) => deep_merge(existing, value),
                    None => value.clone(),
                };
                out.insert(key.clone(), merged);
            }
            Value::Object(out)
        }
        _ => over.clone(),
    }
}

/// Extract the prefab NAME from a `prefab` field value — either a bare string
/// (`"torch"`) or a detailed link (`{ "name": "torch", ... }`). Reference
/// `expandDoc`: `typeof doc.prefab === 'string' ? doc.prefab : doc.prefab?.name`.
pub fn prefab_name(value: &Value) -> Option<String> {
    match value {
        Value::String(name) => Some(name.clone()),
        Value::Object(map) => map.get("name").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

/// Expand a prefab INSTANCE into its full component set (reference `expandDoc`):
/// deep-merge the instance `deltas` over the prefab `base` components, then
/// stamp the prefab link `{ "name": <name> }` so write-back can diff. `deltas`
/// must NOT contain the `prefab` key (the caller peels it off first).
pub fn expand_instance(name: &str, base: &JsonMap, deltas: &JsonMap) -> JsonMap {
    let merged = deep_merge(&Value::Object(base.clone()), &Value::Object(deltas.clone()));
    let mut out = match merged {
        Value::Object(map) => map,
        _ => JsonMap::new(),
    };
    out.insert("prefab".into(), json!({ "name": name }));
    out
}

/// Diff a live/expanded instance back to its stored deltas (reference
/// `sceneDoc`): `{ "prefab": <name> }` plus only the components that DIFFER
/// from the prefab's (whole-component comparison), plus any component the
/// prefab lacks. The `prefab` component collapses to the bare name string.
pub fn diff_instance(name: &str, base: &JsonMap, expanded: &JsonMap) -> JsonMap {
    let mut out = JsonMap::new();
    out.insert("prefab".into(), Value::String(name.to_owned()));
    for (key, value) in expanded {
        if key == "prefab" {
            continue;
        }
        if base.get(key) == Some(value) {
            continue;
        }
        out.insert(key.clone(), value.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn objects_merge_recursively_arrays_replace() {
        let base = json!({
            "mesh": { "parts": [{ "shape": "cylinder" }], "cast": true },
            "light": { "intensity": 2.0 }
        });
        let over = json!({
            "mesh": { "cast": false, "parts": [{ "shape": "box" }] },
            "transform": { "position": [1, 2, 3] }
        });
        let merged = deep_merge(&base, &over);
        // nested object merges key by key; the array replaces wholesale.
        assert_eq!(
            merged,
            json!({
                "mesh": { "parts": [{ "shape": "box" }], "cast": false },
                "light": { "intensity": 2.0 },
                "transform": { "position": [1, 2, 3] }
            })
        );
    }

    #[test]
    fn prefab_name_reads_string_or_link() {
        assert_eq!(prefab_name(&json!("torch")), Some("torch".into()));
        assert_eq!(
            prefab_name(&json!({ "name": "torch" })),
            Some("torch".into())
        );
        assert_eq!(prefab_name(&json!(42)), None);
    }

    #[test]
    fn expand_then_diff_is_identity_on_deltas() {
        let base = json!({
            "mesh": { "parts": [{ "shape": "cylinder", "color": "#3a2d20" }] },
            "light": { "color": "#ffb347", "intensity": 2.0 }
        })
        .as_object()
        .unwrap()
        .clone();
        let deltas = json!({ "transform": { "position": [0, 19, -120] } })
            .as_object()
            .unwrap()
            .clone();

        let expanded = expand_instance("torch", &base, &deltas);
        // the merge pulled the prefab's mesh + light in, kept the instance delta,
        // and stamped the link.
        assert!(expanded.contains_key("mesh"));
        assert!(expanded.contains_key("light"));
        assert_eq!(expanded["transform"], json!({ "position": [0, 19, -120] }));
        assert_eq!(expanded["prefab"], json!({ "name": "torch" }));

        // diffing back drops every component identical to the prefab, leaving
        // exactly the stored deltas (+ the bare-name link).
        let diffed = diff_instance("torch", &base, &expanded);
        let mut expected = deltas;
        expected.insert("prefab".into(), Value::String("torch".into()));
        assert_eq!(diffed, expected);
    }

    #[test]
    fn diff_keeps_overridden_and_novel_components() {
        let base = json!({ "light": { "intensity": 2.0 } })
            .as_object()
            .unwrap()
            .clone();
        let expanded = json!({
            "prefab": { "name": "torch" },
            "light": { "intensity": 5.0 },
            "transform": { "position": [1, 0, 0] }
        })
        .as_object()
        .unwrap()
        .clone();
        let diffed = diff_instance("torch", &base, &expanded);
        assert_eq!(
            diffed,
            json!({
                "prefab": "torch",
                "light": { "intensity": 5.0 },
                "transform": { "position": [1, 0, 0] }
            })
            .as_object()
            .unwrap()
            .clone()
        );
    }
}
