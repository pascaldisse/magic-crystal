//! Matrix-retina primary-ray feature tap. Geometry only: no radiance, pixels,
//! secondary rays, or network state.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{bvh::{Bvh, BvhParams}, scene::{Camera, LeafTriangle, RetinaTag}};

pub const EMPTY_ID: u32 = u32::MAX;

/// CPU BVH cache for Matrix-retina data. Scene epoch + own-body culling are
/// the full geometry key; pose, resolution, and foveal windows reuse it.
#[derive(Default)]
pub struct GeometryCache {
    entries: Vec<CachedGeometry>,
    builds: u64,
}

struct CachedGeometry {
    epoch: u64,
    culls_own_body: bool,
    bvh: Bvh,
    tags: Vec<RetinaTag>,
}

impl GeometryCache {
    pub fn clear(&mut self) { self.entries.clear(); }

    /// Lazy by construction: a hit does not materialize triangles or build a BVH.
    pub fn get_or_build<F>(&mut self, epoch: u64, culls_own_body: bool, params: &BvhParams, build: F) -> (&Bvh, &[RetinaTag])
    where F: FnOnce() -> (Vec<LeafTriangle>, Vec<RetinaTag>), {
        if let Some(index) = self.entries.iter().position(|entry| entry.epoch == epoch && entry.culls_own_body == culls_own_body) {
            let entry = &self.entries[index];
            return (&entry.bvh, &entry.tags);
        }
        self.entries.retain(|entry| entry.epoch == epoch);
        let (triangles, tags) = build();
        debug_assert_eq!(triangles.len(), tags.len());
        let (bvh, source) = Bvh::build_indexed(&triangles, params);
        let tags = source.into_iter().map(|index| tags[index as usize].clone()).collect();
        self.entries.push(CachedGeometry { epoch, culls_own_body, bvh, tags });
        self.builds += 1;
        let entry = self.entries.last().expect("just pushed retina geometry");
        (&entry.bvh, &entry.tags)
    }

    #[cfg(test)]
    fn builds(&self) -> u64 { self.builds }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle(z: f32) -> LeafTriangle {
        LeafTriangle::lambertian([[-1.0, -1.0, z], [1.0, -1.0, z], [0.0, 1.0, z]], [0.5; 3], [0.0; 3])
    }

    #[test]
    fn geometry_cache_reuses_epoch_and_invalidates_on_change() {
        let mut cache = GeometryCache::default();
        let params = BvhParams::default();
        let (bvh, tags) = cache.get_or_build(7, false, &params, || (vec![triangle(-3.0)], vec![RetinaTag { entity_id: "wall".into(), material_id: "stone".into() }]));
        assert_eq!(bvh.tris.len(), 1);
        assert_eq!(tags[0].entity_id, "wall");
        let (_, tags) = cache.get_or_build(7, false, &params, || panic!("cache hit must not rebuild"));
        assert_eq!(tags[0].material_id, "stone");
        assert_eq!(cache.builds(), 1);
        let (bvh, _) = cache.get_or_build(8, false, &params, || (vec![triangle(-5.0)], vec![RetinaTag { entity_id: "far-wall".into(), material_id: "stone".into() }]));
        assert_eq!(bvh.tris.len(), 1);
        assert_eq!(cache.builds(), 2);
    }

    /// Headless ordeal for the HTTP handler's CPU core: actual Naruko geometry,
    /// same primary-ray layers and JSON serialization, with the old per-pull
    /// build contrasted against a warmed scene-epoch cache.
    #[test]
    fn naruko_retina_cache_latency_and_truth() {
        use std::{path::Path, time::Instant};
        use glam::Vec3;
        use crate::{scene::{Camera, RenderScene}, upscaler_dataset::naruko_params};

        let mut core = crystal::Core::default();
        let world = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko");
        crystal::load_world_dir(&world, &mut core.world).expect("load Naruko");
        let scene = RenderScene::from_ecs(std::mem::take(&mut core.world), &naruko_params()).expect("materialize Naruko");
        let camera = Camera { eye: Vec3::new(0.0, 1.7, 44.0), yaw: 0.0, pitch: 0.0, fov_y_radians: 60.0_f32.to_radians(), near: 0.05, far: 1000.0 };
        let layers = Layers { depth: true, normal: true, entity_id: true, material_id: true, world_pos: true };
        let params = BvhParams::default();
        let pulls = 4;
        let mut before = Vec::with_capacity(pulls);
        let mut old_depth = None;
        for _ in 0..pulls {
            let start = Instant::now();
            let (triangles, tags) = scene.retina_triangles_for_eye(camera.eye, crate::scene::OWN_EYE_EPSILON_M, false);
            let (bvh, source) = Bvh::build_indexed(&triangles, &params);
            let tags = source.into_iter().map(|index| tags[index as usize].clone()).collect::<Vec<_>>();
            let image = trace(&bvh, &tags, &camera, 64, 64, layers);
            serde_json::to_vec(&image).expect("retina JSON");
            old_depth = Some(image.depth);
            before.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let mut cache = GeometryCache::default();
        let mut after = Vec::with_capacity(pulls);
        let mut cached_depth = None;
        for _ in 0..pulls {
            let start = Instant::now();
            let (bvh, tags) = cache.get_or_build(1, false, &params, || scene.retina_triangles_for_eye(camera.eye, crate::scene::OWN_EYE_EPSILON_M, false));
            let image = trace(bvh, tags, &camera, 64, 64, layers);
            serde_json::to_vec(&image).expect("retina JSON");
            cached_depth = Some(image.depth);
            after.push(start.elapsed().as_secs_f64() * 1e3);
        }
        assert_eq!(old_depth, cached_depth, "cache must preserve Naruko primary-ray depth exactly");
        assert_eq!(cache.builds(), 1, "four same-epoch pulls must build once");
        let mean = |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!("[retina latency] Naruko 64x64 core old-per-pull={:.3}ms cached-warm={:.3}ms", mean(&before), mean(&after));
    }
}
