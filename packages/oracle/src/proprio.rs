//! `proprio()` — an entity's own pose + component summary, read from the ECS.

use crate::geom::Vec3;
use crate::model::World;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug, Serialize)]
pub struct ComponentSummary {
    pub name: String,
    /// One-line preview of the authored value.
    pub preview: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Proprio {
    pub id: String,
    /// Entity origin (transform.position), world space.
    pub position: Vec3,
    /// Bounds center + max extent, when the entity has geometry.
    pub bounds_center: Option<Vec3>,
    pub size: Option<Vec3>,
    /// Emissive color string of the entity's first emissive part, if any.
    pub emissive: Option<String>,
    /// Facing yaw when the entity carries a spawn pose.
    pub yaw: Option<f32>,
    pub components: Vec<ComponentSummary>,
}

/// Read one entity's proprioception straight off the LIVE ECS. Pull-only —
/// geometry is derived fresh, so mutations are reflected immediately.
pub fn proprio(world: &World, id: &str) -> Option<Proprio> {
    let ent = world.get(id)?;
    let geom = world.geometry(id)?;
    let components = ent
        .components
        .iter()
        .map(|name| ComponentSummary {
            name: name.clone(),
            preview: preview(world.component_value(id, name)),
        })
        .collect();
    Some(Proprio {
        id: ent.id.clone(),
        position: geom.origin,
        bounds_center: geom.bounds.map(|b| b.center()),
        size: geom.bounds.map(|b| b.size()),
        emissive: geom.emissive,
        yaw: geom.yaw,
        components,
    })
}

/// Compact one-line JSON preview, truncated so a summary stays a summary.
fn preview(value: Option<Value>) -> String {
    const MAX: usize = 120;
    let text = value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".into());
    if text.len() > MAX {
        let mut end = MAX;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &text[..end])
    } else {
        text
    }
}
