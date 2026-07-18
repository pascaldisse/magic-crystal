//! Matrix-retina primary-ray feature tap. Geometry only: no radiance, pixels,
//! secondary rays, or network state.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{bvh::Bvh, scene::{Camera, RetinaTag}};

pub const EMPTY_ID: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default)]
pub struct Layers {
    pub depth: bool,
    pub normal: bool,
    pub entity_id: bool,
    pub material_id: bool,
    pub world_pos: bool,
}

#[derive(Serialize)]
pub struct RetinaImage {
    pub resolution: [u32; 2],
    /// `[center_x, center_y, radius]` in base-image normalized coordinates.
    pub sample_window: [f32; 3],
    pub eye: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_deg: f32,
    pub ray_model: &'static str,
    pub miss_depth: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal: Option<Vec<[f32; 3]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material_id: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub world_pos: Option<Vec<[f32; 3]>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub entity_table: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub material_table: Vec<String>,
}

/// One deterministic feature pass through the packed native BVH. `tags` is in
/// Bvh leaf order (the `src` permutation from `Bvh::build_indexed` applied).
pub fn trace(bvh: &Bvh, tags: &[RetinaTag], camera: &Camera, width: u32, height: u32, layers: Layers) -> RetinaImage {
    trace_window(bvh, tags, camera, width, height, layers, [0.5, 0.5], 1.0)
}

/// Trace a normalized square image-plane window. A small window rendered at a
/// higher resolution is a real foveal level, not an upsampled base grid.
pub fn trace_window(bvh: &Bvh, tags: &[RetinaTag], camera: &Camera, width: u32, height: u32, layers: Layers, center: [f32; 2], radius: f32) -> RetinaImage {
    let cells = (width as usize) * (height as usize);
    let mut depth = layers.depth.then(|| vec![-1.0; cells]);
    let mut normal = layers.normal.then(|| vec![[0.0; 3]; cells]);
    let mut world_pos = layers.world_pos.then(|| vec![[0.0; 3]; cells]);
    let mut entity_id = layers.entity_id.then(|| vec![EMPTY_ID; cells]);
    let mut material_id = layers.material_id.then(|| vec![EMPTY_ID; cells]);
    let mut entities = BTreeMap::<String, u32>::new();
    let mut materials = BTreeMap::<String, u32>::new();
    let (right, up, forward) = camera.basis();
    let half = (camera.fov_y_radians * 0.5).tan();
    let aspect = width as f32 / height.max(1) as f32;
    for y in 0..height {
        for x in 0..width {
            let base_x = center[0] + (((x as f32 + 0.5) / width as f32) - 0.5) * radius;
            let base_y = center[1] + (((y as f32 + 0.5) / height as f32) - 0.5) * radius;
            let sx = (base_x * 2.0 - 1.0) * half * aspect;
            let sy = (1.0 - base_y * 2.0) * half;
            let direction = (forward + right * sx + up * sy).normalize_or_zero().to_array();
            let index = y as usize * width as usize + x as usize;
            let Some(hit) = bvh.primary_hit(camera.eye.to_array(), direction, camera.near, camera.far) else { continue };
            if let Some(out) = &mut depth { out[index] = hit.distance; }
            if let Some(out) = &mut normal { out[index] = hit.normal; }
            if let Some(out) = &mut world_pos { out[index] = hit.position; }
            let tag = tags.get(hit.tri_index);
            if let (Some(out), Some(tag)) = (&mut entity_id, tag) {
                let next = entities.len() as u32;
                out[index] = *entities.entry(tag.entity_id.clone()).or_insert(next);
            }
            if let (Some(out), Some(tag)) = (&mut material_id, tag) {
                let next = materials.len() as u32;
                out[index] = *materials.entry(tag.material_id.clone()).or_insert(next);
            }
        }
    }
    let table = |map: BTreeMap<String, u32>| {
        let mut table = vec![String::new(); map.len()];
        for (name, index) in map { table[index as usize] = name; }
        table
    };
    RetinaImage {
        resolution: [width, height], sample_window: [center[0], center[1], radius], eye: camera.eye.to_array(), yaw: camera.yaw,
        pitch: camera.pitch, fov_deg: camera.fov_y_radians.to_degrees(),
        ray_model: "native-bvh-primary", miss_depth: -1.0,
        depth, normal, entity_id, material_id, world_pos,
        entity_table: table(entities), material_table: table(materials),
    }
}
