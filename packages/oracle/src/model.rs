//! Load a GAIA world directory -> Crystal ECS, and derive the pure-geometry
//! summary each sense reads. PULL-ONLY: no timers, no streaming, no GPU.
//!
//! LIVE ECS: geometry is NOT cached at load. Every [`World::geometry`] reads the
//! entity's `transform`/`mesh` components straight off the live ECS and derives
//! its world-space bounds fresh, so a mutation (`set_component` on transform or
//! mesh) is reflected on the very next gaze. [`World::load`] is the convenience
//! world-dir constructor (build the Core, then use the live path);
//! [`World::from_core`] wraps a Core someone else is already mutating.
//!
//! SCOPE (documented, per the AABB-derivation charter): primitive shapes
//! box/plane/sphere/cylinder/cone/tube are derived from their authored fields,
//! honoring entity+part rotation (Euler XYZ) and scale exactly as the renderer
//! does. Prefab expansion, `material`/`preset` look resolution, and `model`
//! (glTF) mesh bounds are OUT OF SCOPE — a `model` part contributes its
//! authored `placeholderSize` box when present, otherwise a param'd fallback
//! (never an invented per-shape constant). `visible: false` parts are excluded.

use crate::geom::{read_f32, read_vec3, Aabb, Affine, Vec3};
use crystal::{Core, Entity, EntityMap};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Fallback half-extent for a primitive whose shape carries no dimensions
/// (param-driven default, never a hidden constant). Matches the renderer's
/// authored default of a unit primitive (`size [1,1,1]`, `radius 0.5`).
pub const DEFAULT_PART_HALF: f32 = 0.5;

/// Tube's authored default radius. The reference engine's `tubeRadii`
/// (`client/kernel/geometry.js`) resolves `radii ?? radius ?? 3` — the `3`
/// is correctness-bearing (the gizmo rings, the edit lens and the tube builder
/// all index it), so bounds MUST use 3, not the generic primitive half.
pub const TUBE_DEFAULT_RADIUS: f32 = 3.0;

/// Max lump amplitude of the reference tube `wobble` (the three sine terms in
/// `buildTubeGeometry` sum to `0.5 + 0.3 + 0.2 = 1.0`), so the rendered radius
/// peaks at `r · (1 + wobble)`. Bounds inflate by exactly this margin.
const TUBE_WOBBLE_MAX_LUMP: f32 = 1.0;

/// The renderer's default tube `path` when a part omits it
/// (`part.path ?? [[0,0,0],[10,0,0]]` in `buildTubeGeometry`). A bare
/// `{shape:"tube"}` renders 375 real vertices along this segment, so bounds
/// must match it, never return `None`.
const TUBE_DEFAULT_PATH: [Vec3; 2] = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];

/// three.js `Curve.arcLengthDivisions` — the fixed tessellation count over which
/// `CatmullRomCurve3` builds its cumulative chord-length table for arc-length
/// (`u`) → curve (`t`) remapping. The renderer places tube rings by `getPointAt`
/// (arc-length `u`), so bounds MUST reparameterize through the same table.
const TUBE_ARC_DIVISIONS: usize = 200;

/// The entity registry entry: a stable id ↔ ECS handle binding plus the sorted
/// component-name list. Geometry lives in the ECS, derived on demand.
#[derive(Clone, Debug)]
pub struct SenseEntity {
    pub id: String,
    pub entity: Entity,
    pub components: Vec<String>,
}

/// One entity's world-space geometry, derived live from the ECS each gaze.
#[derive(Clone, Debug, Default)]
pub struct EntityGeom {
    /// Entity origin (`transform.position`), world space.
    pub origin: Vec3,
    /// World-space bounds of the visible mesh; `None` = no renderable geometry.
    pub bounds: Option<Aabb>,
    /// Emissive COLOR STRING of the first emissive part, if any (data-side the
    /// engine authors `emissive` as a color; the protocol types it `Option<String>`).
    pub emissive: Option<String>,
    /// Spawn yaw when the entity carries a `spawn` component.
    pub yaw: Option<f32>,
}
impl EntityGeom {
    pub fn is_renderable(&self) -> bool {
        self.bounds.is_some()
    }
    pub fn max_extent(&self) -> f32 {
        self.bounds.map(|b| b.max_extent()).unwrap_or(0.0)
    }
}

/// A loaded world: the live ECS plus the id registry the senses read over.
pub struct World {
    pub core: Core,
    pub entities: Vec<SenseEntity>,
    by_id: HashMap<String, usize>,
    pub world_dir: PathBuf,
    /// Non-fatal protocol/schema mismatches found while validating the world.
    pub schema_warnings: Vec<String>,
    /// Files whose entity docs were loaded (relative to world dir).
    pub scene_files: Vec<String>,
}

impl World {
    /// Resolve the world dir from `GAIA_WORLD` or the provided default.
    pub fn resolve_dir(default: impl AsRef<Path>) -> PathBuf {
        std::env::var_os("GAIA_WORLD")
            .map(PathBuf::from)
            .unwrap_or_else(|| default.as_ref().to_path_buf())
    }

    /// Convenience world-dir constructor: build a Core from every scene document
    /// under `<dir>/scenes/*.json`, then wrap it via the live path. Blank-page
    /// rule: a lone `scenes/main.json` is the implicit `main` scene; world.json
    /// composition is not needed for pull-only senses.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref().to_path_buf();
        let scenes_dir = dir.join("scenes");
        if !scenes_dir.is_dir() {
            return Err(format!("no scenes/ directory under {}", dir.display()));
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&scenes_dir)
            .map_err(|e| format!("read {}: {e}", scenes_dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no scene json under {}", scenes_dir.display()));
        }

        let mut core = Core::default();
        let mut component_ids: HashMap<String, u32> = HashMap::new();
        let mut schema_warnings: Vec<String> = Vec::new();
        let mut scene_files: Vec<String> = Vec::new();
        // Deferred entity registry: filled after the Core exists.
        let mut pending: Vec<(String, Entity, Vec<String>)> = Vec::new();

        for path in &files {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("read {}: {e}", path.display()))?;
            let rel = path
                .strip_prefix(&dir)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            scene_files.push(rel.clone());

            // Raw parse is the source of truth for load (never fails on a
            // valid JSON doc); the typed protocol parse is a validation pass
            // whose failures become reported schema corrections, not blockers.
            let raw: Map<String, Value> = serde_json::from_str(&text)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            if let Err(e) = serde_json::from_value::<EntityMap>(Value::Object(raw.clone())) {
                schema_warnings.push(format!(
                    "{rel}: Crystal protocol typing rejected the doc ({e})"
                ));
            }

            for (id, doc) in &raw {
                let Some(obj) = doc.as_object() else { continue };
                let mut comps: Vec<(u32, Value)> = Vec::new();
                let mut names: Vec<String> = Vec::new();
                for (comp, value) in obj {
                    let cid = *component_ids.entry(comp.clone()).or_insert_with(|| {
                        core.world
                            .register_component_json(&format!(
                                r#"{{"name":{},"fields":{{"v":"object"}}}}"#,
                                Value::String(comp.clone())
                            ))
                            .expect("register sense component")
                    });
                    comps.push((cid, serde_json::json!({ "v": value })));
                    names.push(comp.clone());
                }
                let entity = core
                    .world
                    .create_entity(comps)
                    .map_err(|e| format!("create entity {id}: {e}"))?;
                core.world
                    .bind_gaia_id(id.clone(), entity)
                    .map_err(|e| format!("bind {id}: {e}"))?;
                names.sort();
                pending.push((id.clone(), entity, names));
            }
        }

        let mut world = Self::from_core(core, dir);
        for (id, entity, names) in pending {
            world.register(id, entity, names);
        }
        world.schema_warnings = schema_warnings;
        world.scene_files = scene_files;
        Ok(world)
    }

    /// Wrap a live Core. The registry starts empty; call [`World::register`]
    /// for each id the senses should see. This is the LIVE path — mutate
    /// `world.core` and the next gaze reflects it.
    pub fn from_core(core: Core, world_dir: impl AsRef<Path>) -> Self {
        Self {
            core,
            entities: Vec::new(),
            by_id: HashMap::new(),
            world_dir: world_dir.as_ref().to_path_buf(),
            schema_warnings: Vec::new(),
            scene_files: Vec::new(),
        }
    }

    /// Add an entity to the sense registry (id ↔ ECS handle + component names).
    pub fn register(&mut self, id: String, entity: Entity, mut components: Vec<String>) {
        components.sort();
        components.dedup();
        self.by_id.insert(id.clone(), self.entities.len());
        self.entities.push(SenseEntity {
            id,
            entity,
            components,
        });
    }

    pub fn get(&self, id: &str) -> Option<&SenseEntity> {
        self.by_id.get(id).map(|i| &self.entities[*i])
    }

    /// Derive an entity's world-space geometry LIVE from the ECS. This is the
    /// single source of geometric truth for every sense; it re-reads
    /// `transform`/`mesh`/`spawn` each call so mutations are always visible.
    pub fn geometry(&self, id: &str) -> Option<EntityGeom> {
        let ent = self.get(id)?;
        let transform = self.component_value(id, "transform");
        let mesh = self.component_value(id, "mesh");
        let spawn = self.component_value(id, "spawn");
        Some(derive_geometry(
            transform.as_ref(),
            mesh.as_ref(),
            spawn.as_ref(),
            ent.components.iter().any(|c| c == "spawn"),
        ))
    }

    /// The world spawn eye pose (position + yaw, pitch 0), if a `spawn`
    /// component exists. The eye is `spawn.position` (NOT transform).
    pub fn spawn_pose(&self) -> Option<crate::look::EyePose> {
        let ent = self
            .entities
            .iter()
            .find(|e| e.components.iter().any(|c| c == "spawn"))?;
        let spawn = self.component_value(&ent.id, "spawn")?;
        let g = self.geometry(&ent.id)?;
        let position = read_vec3(spawn.get("position"), g.origin);
        Some(crate::look::EyePose {
            position,
            yaw: g.yaw.unwrap_or(0.0),
            pitch: 0.0,
        })
    }

    /// The environment component's raw value, if any (sky/fog summary source).
    pub fn environment(&self) -> Option<Value> {
        let cid = self.core.world.component_id("environment")?;
        self.entities
            .iter()
            .find(|e| e.components.iter().any(|c| c == "environment"))
            .and_then(|e| self.core.world.get_component(e.entity, cid).ok())
            .and_then(|v| v.get("v").cloned())
    }

    /// Read a component's raw authored value for an entity id (proprio source).
    pub fn component_value(&self, id: &str, component: &str) -> Option<Value> {
        let ent = self.get(id)?;
        let cid = self.core.world.component_id(component)?;
        self.core
            .world
            .get_component(ent.entity, cid)
            .ok()
            .and_then(|v| v.get("v").cloned())
    }
}

/// Derive world-space geometry from authored `transform`/`mesh`/`spawn` values.
fn derive_geometry(
    transform: Option<&Value>,
    mesh: Option<&Value>,
    spawn: Option<&Value>,
    has_spawn: bool,
) -> EntityGeom {
    let origin = read_vec3(transform.and_then(|t| t.get("position")), [0.0; 3]);
    let entity_affine = affine_of(transform, origin);

    let mut bounds: Option<Aabb> = None;
    let mut emissive: Option<String> = None;
    for part in parts_of(mesh) {
        if part_hidden(&part) {
            continue;
        }
        if emissive.is_none() {
            if let Some(color) = part_emissive(&part) {
                emissive = Some(color);
            }
        }
        if let Some(local) = part_local_aabb(&part) {
            let part_affine = affine_of_part(&part);
            let world = entity_affine.then(&part_affine).transform_aabb(&local);
            bounds = Some(match bounds {
                Some(b) => b.union(&world),
                None => world,
            });
        }
    }

    let yaw = spawn
        .filter(|_| has_spawn)
        .and_then(|s| s.get("yaw"))
        .and_then(Value::as_f64)
        .map(|v| v as f32);

    EntityGeom {
        origin,
        bounds,
        emissive,
        yaw,
    }
}

/// Single-part convention: `mesh.parts ?? [mesh]` — a mesh that carries a
/// `shape` directly (no `parts`) is one implicit part.
fn parts_of(mesh: Option<&Value>) -> Vec<Value> {
    let Some(mesh) = mesh else { return Vec::new() };
    if let Some(parts) = mesh.get("parts").and_then(Value::as_array) {
        return parts.clone();
    }
    if mesh.get("shape").is_some() {
        return vec![mesh.clone()];
    }
    Vec::new()
}

/// The entity-level transform (rotation Euler XYZ + scale) about `origin`.
fn affine_of(transform: Option<&Value>, origin: Vec3) -> Affine {
    let rot = read_vec3(transform.and_then(|t| t.get("rotation")), [0.0; 3]);
    let scale = read_scale(transform.and_then(|t| t.get("scale")));
    Affine::from_trs(origin, rot, scale)
}

/// A part's local transform (its `position`/`rotation`/`scale`).
fn affine_of_part(part: &Value) -> Affine {
    let pos = read_vec3(part.get("position"), [0.0; 3]);
    let rot = read_vec3(part.get("rotation"), [0.0; 3]);
    let scale = read_scale(part.get("scale"));
    Affine::from_trs(pos, rot, scale)
}

/// `scale` may be a scalar (uniform) or a `[x,y,z]` array; default 1.
fn read_scale(value: Option<&Value>) -> Vec3 {
    match value {
        Some(Value::Number(n)) => {
            let s = n.as_f64().unwrap_or(1.0) as f32;
            [s, s, s]
        }
        Some(Value::Array(_)) => read_vec3(value, [1.0, 1.0, 1.0]),
        _ => [1.0, 1.0, 1.0],
    }
}

fn part_hidden(part: &Value) -> bool {
    part.get("visible") == Some(&Value::Bool(false))
}

/// The emissive COLOR STRING of a part, if it carries one (empty string = off).
fn part_emissive(part: &Value) -> Option<String> {
    match part.get("emissive") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Axis-aligned LOCAL bounds of a primitive part before its transform, or
/// `None` for shapes with no derivable geometry.
fn part_local_aabb(part: &Value) -> Option<Aabb> {
    let shape = part.get("shape").and_then(Value::as_str).unwrap_or("box");
    let d = DEFAULT_PART_HALF;
    let half = match shape {
        // Renderer default box/plane is a unit primitive (size [1,1,1]).
        "box" | "plane" => {
            let s = read_vec3(part.get("size"), [d * 2.0, d * 2.0, d * 2.0]);
            [s[0] * 0.5, s[1] * 0.5, s[2] * 0.5]
        }
        // Cylinder honors radiusTop/radiusBottom (widest ring bounds x/z).
        "cylinder" => {
            let r = read_f32(Some(part), "radius", d);
            let rt = read_f32(Some(part), "radiusTop", r);
            let rb = read_f32(Some(part), "radiusBottom", r);
            let rmax = rt.max(rb);
            let h = read_f32(Some(part), "height", 1.0);
            [rmax, h * 0.5, rmax]
        }
        "cone" => {
            let r = read_f32(Some(part), "radius", d);
            let h = read_f32(Some(part), "height", 1.0);
            [r, h * 0.5, r]
        }
        "sphere" => {
            let r = read_f32(Some(part), "radius", d);
            [r, r, r]
        }
        // Tube: union of the RENDERED spline (sampled) ± its eased+wobbled
        // radius — not just the control points (a centripetal Catmull-Rom sags
        // past its control polygon).
        "tube" => return tube_local_aabb(part),
        // model / unknown: honor an explicit placeholder box; else a documented
        // param'd fallback (no invented per-shape constant).
        _ => {
            if let Some(v) = part.get("placeholderSize") {
                let s = read_vec3(Some(v), [d * 2.0, d * 2.0, d * 2.0]);
                [s[0] * 0.5, s[1] * 0.5, s[2] * 0.5]
            } else if let Some(v) = part.get("size") {
                let s = read_vec3(Some(v), [d * 2.0, d * 2.0, d * 2.0]);
                [s[0] * 0.5, s[1] * 0.5, s[2] * 0.5]
            } else {
                [d, d, d]
            }
        }
    };
    Some(Aabb::from_center_half([0.0; 3], half))
}

/// Tube bounds, matching the reference renderer's `buildTubeGeometry`
/// (`client/kernel/geometry.js`): the surface is a centripetal Catmull-Rom
/// through `path` with a radius eased (uniform Catmull-Rom over `radii`) per
/// point, optionally inflated by `wobble`. Rings are placed by ARC LENGTH
/// (`curve.getPointAt(u)` = `getPoint(getUtoTmapping(u))`), and the radius eases
/// at the SAME remapped `t` (`radiusAt(curve.getUtoTmapping(u))`) — so bounds
/// SAMPLE the rendered curve through the identical arc-length reparameterization
/// (uniform-`t` sampling underbounds the geometry) and union each ring's
/// conservative `point ± radius` box.
///
/// Radius resolution mirrors `tubeRadii`: an array is per-point (indexed
/// CLAMP-LAST, never a 0.5 fallback), a scalar/`radius` is uniform, and the
/// authored default is [`TUBE_DEFAULT_RADIUS`] (3), not the primitive half.
/// An absent `path` defaults to [`TUBE_DEFAULT_PATH`] (the renderer's default),
/// never `None`.
fn tube_local_aabb(part: &Value) -> Option<Aabb> {
    let pts: Vec<Vec3> = match part.get("path").and_then(Value::as_array) {
        Some(path) if !path.is_empty() => {
            path.iter().map(|p| read_vec3(Some(p), [0.0; 3])).collect()
        }
        _ => TUBE_DEFAULT_PATH.to_vec(),
    };
    let n = pts.len();
    let closed = part.get("closed").and_then(Value::as_bool).unwrap_or(false);

    // `tubeRadii(part)`: array stays per-point; scalar `radii`/legacy `radius`
    // becomes a one-element list; absent => the authored default 3.
    let radii: Vec<f32> = match part.get("radii") {
        Some(Value::Array(a)) if !a.is_empty() => a
            .iter()
            .map(|v| v.as_f64().map(|x| x as f32).unwrap_or(TUBE_DEFAULT_RADIUS))
            .collect(),
        Some(Value::Number(x)) => vec![x.as_f64().unwrap_or(TUBE_DEFAULT_RADIUS as f64) as f32],
        _ => vec![part
            .get("radius")
            .and_then(Value::as_f64)
            .map(|x| x as f32)
            .unwrap_or(TUBE_DEFAULT_RADIUS)],
    };

    // wobble peaks the radius at `r · (1 + wobble)` (see TUBE_WOBBLE_MAX_LUMP).
    let wobble = read_f32(Some(part), "wobble", 0.0).max(0.0);
    let inflate = 1.0 + wobble * TUBE_WOBBLE_MAX_LUMP;

    // Sample count matches the reference default (`pts.length * 12`, min 2);
    // an explicit `tubularSegments` overrides it, same as the builder.
    let segs = part
        .get("tubularSegments")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(n * 12)
        .max(2);

    // Arc-length reparameterization table (three.js `Curve.getLengths`): the
    // renderer evenly spaces rings by ARC LENGTH, so an even `u` grid must be
    // remapped to curve `t` before sampling position AND radius.
    let arc = tube_arc_lengths(&pts, closed);

    let mut out: Option<Aabb> = None;
    for i in 0..=segs {
        let u = i as f32 / segs as f32;
        let t = tube_u_to_t(&arc, u);
        let c = catmull_rom_point(&pts, closed, t);
        let r = tube_radius_at(&radii, n, closed, t) * inflate;
        let a = Aabb::from_center_half(c, [r, r, r]);
        out = Some(match out {
            Some(b) => b.union(&a),
            None => a,
        });
    }
    out
}

/// Cumulative chord-length table over the tube spline, faithful to three.js
/// `Curve.getLengths`: sample `getPoint(p/divisions)` for `p ∈ 0..=divisions`
/// and accumulate successive chord distances. Index 0 is 0; the last entry is
/// the full curve length. [`TUBE_ARC_DIVISIONS`] divisions match the engine.
fn tube_arc_lengths(pts: &[Vec3], closed: bool) -> Vec<f32> {
    let div = TUBE_ARC_DIVISIONS;
    let mut cache = Vec::with_capacity(div + 1);
    let mut last = catmull_rom_point(pts, closed, 0.0);
    cache.push(0.0f32);
    let mut sum = 0.0f32;
    for p in 1..=div {
        let cur = catmull_rom_point(pts, closed, p as f32 / div as f32);
        let d = [cur[0] - last[0], cur[1] - last[1], cur[2] - last[2]];
        sum += (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        cache.push(sum);
        last = cur;
    }
    cache
}

/// Arc-length parameter `u ∈ [0,1]` → curve parameter `t`, faithful to three.js
/// `Curve.getUtoTmapping`: binary-search the target arc length in the table,
/// then linearly interpolate within the found chord segment. This is what makes
/// ring spacing (and thus the sampled bounds) even by distance, not by `t`.
fn tube_u_to_t(arc: &[f32], u: f32) -> f32 {
    let il = arc.len();
    if il < 2 {
        return 0.0;
    }
    let target = u * arc[il - 1];
    // Binary search for the highest index whose arc length ≤ target.
    let (mut low, mut high) = (0isize, il as isize - 1);
    while low <= high {
        let mid = low + (high - low) / 2;
        let comparison = arc[mid as usize] - target;
        if comparison < 0.0 {
            low = mid + 1;
        } else if comparison > 0.0 {
            high = mid - 1;
        } else {
            high = mid;
            break;
        }
    }
    let i = high.max(0) as usize;
    if arc[i] == target {
        return i as f32 / (il - 1) as f32;
    }
    let length_before = arc[i];
    let segment_length = arc[i + 1] - length_before;
    let segment_fraction = if segment_length > 0.0 {
        (target - length_before) / segment_length
    } else {
        0.0
    };
    (i as f32 + segment_fraction) / (il - 1) as f32
}

/// Radius eased along the tube, mirroring the reference `radiusAt`: a uniform
/// Catmull-Rom over the control radii (indexed CLAMP-LAST past the array end),
/// floored at 0.15 exactly as the builder does.
fn tube_radius_at(radii: &[f32], n: usize, closed: bool, t: f32) -> f32 {
    let r_of = |i: isize| -> f32 {
        let ni = n as isize;
        let m = ((i % ni) + ni) % ni; // wrap into 0..n (the control-point count)
        radii[(radii.len() - 1).min(m as usize)] // CLAMP-LAST into the radii list
    };
    let span = if closed {
        n as f32
    } else {
        (n - 1).max(1) as f32
    };
    let f = t.clamp(0.0, 0.99999) * span;
    let k = f.floor();
    let s = f - k;
    let ki = k as isize;
    let (r0, r1, r2, r3) = (r_of(ki - 1), r_of(ki), r_of(ki + 1), r_of(ki + 2));
    let v0 = (r2 - r0) * 0.5;
    let v1 = (r3 - r1) * 0.5;
    let val = (2.0 * r1 - 2.0 * r2 + v0 + v1) * s * s * s
        + (-3.0 * r1 + 3.0 * r2 - 2.0 * v0 - v1) * s * s
        + v0 * s
        + r1;
    val.max(0.15)
}

/// Centripetal Catmull-Rom position at parameter `t ∈ [0,1]`, faithful to
/// three.js `CatmullRomCurve3` (the reference builds the tube on it): the same
/// nonuniform coefficients and endpoint extrapolation, so bounds cover the
/// curve's overshoot past its control polygon.
fn catmull_rom_point(pts: &[Vec3], closed: bool, t: f32) -> Vec3 {
    let l = pts.len();
    if l == 0 {
        return [0.0; 3];
    }
    if l == 1 {
        return pts[0];
    }
    let li = l as isize;
    let wrap = |i: isize| -> Vec3 { pts[(((i % li) + li) % li) as usize] };

    let p = (l as f32 - if closed { 0.0 } else { 1.0 }) * t;
    let mut int_point = p.floor() as isize;
    let mut weight = p - int_point as f32;
    if closed {
        if int_point <= 0 {
            int_point += (int_point.unsigned_abs() / l + 1) as isize * li;
        }
    } else if weight == 0.0 && int_point == li - 1 {
        int_point = li - 2;
        weight = 1.0;
    }

    let p1 = wrap(int_point);
    let p2 = wrap(int_point + 1);
    // Endpoint tangents extrapolate (`2·pts[end] - pts[neighbor]`) exactly as
    // three.js does when the segment sits at an open curve's boundary.
    let p0 = if closed || int_point > 0 {
        wrap(int_point - 1)
    } else {
        [
            2.0 * pts[0][0] - pts[1][0],
            2.0 * pts[0][1] - pts[1][1],
            2.0 * pts[0][2] - pts[1][2],
        ]
    };
    let p3 = if closed || int_point + 2 < li {
        wrap(int_point + 2)
    } else {
        [
            2.0 * pts[l - 1][0] - pts[l - 2][0],
            2.0 * pts[l - 1][1] - pts[l - 2][1],
            2.0 * pts[l - 1][2] - pts[l - 2][2],
        ]
    };

    // Centripetal knot spacing: distance^(2·0.25) between successive points.
    let dist_sq = |a: Vec3, b: Vec3| -> f32 {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    let mut dt0 = dist_sq(p0, p1).powf(0.25);
    let mut dt1 = dist_sq(p1, p2).powf(0.25);
    let mut dt2 = dist_sq(p2, p3).powf(0.25);
    if dt1 < 1e-4 {
        dt1 = 1.0;
    }
    if dt0 < 1e-4 {
        dt0 = dt1;
    }
    if dt2 < 1e-4 {
        dt2 = dt1;
    }

    let axis = |a: usize| -> f32 {
        let (x0, x1, x2, x3) = (p0[a], p1[a], p2[a], p3[a]);
        let mut m1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
        let mut m2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;
        m1 *= dt1;
        m2 *= dt1;
        let c0 = x1;
        let c1 = m1;
        let c2 = -3.0 * x1 + 3.0 * x2 - 2.0 * m1 - m2;
        let c3 = 2.0 * x1 - 2.0 * x2 + m1 + m2;
        let w2 = weight * weight;
        c0 + c1 * weight + c2 * w2 + c3 * w2 * weight
    };
    [axis(0), axis(1), axis(2)]
}

#[cfg(test)]
mod aabb_tests {
    use super::*;
    use serde_json::json;

    /// Approx-equal on Vec3 (float geometry tolerance).
    fn close(a: Vec3, b: Vec3, eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    /// World bounds of an authored `{transform, mesh}` doc (the live path a gaze
    /// takes), so rotation/scale composition is under test end-to-end.
    fn world_bounds(transform: Value, mesh: Value) -> Option<Aabb> {
        derive_geometry(Some(&transform), Some(&mesh), None, false).bounds
    }

    /// Local bounds of a single part (primitive derivation under test).
    fn part_bounds(part: Value) -> Option<Aabb> {
        part_local_aabb(&part)
    }

    #[test]
    fn rotation_rotates_the_world_aabb() {
        // A 2×2×10 bar along Z, entity-rotated 90° about Y → now along X.
        let mesh = json!({ "parts": [{ "shape": "box", "size": [2, 2, 10] }] });
        let rot = std::f32::consts::FRAC_PI_2;
        let b = world_bounds(
            json!({ "position": [0, 0, 0], "rotation": [0, rot, 0] }),
            mesh,
        )
        .expect("bounds");
        // The long axis is now X (±5), the short axes ±1.
        assert!(close(b.min, [-5.0, -1.0, -1.0], 1e-3), "min {:?}", b.min);
        assert!(close(b.max, [5.0, 1.0, 1.0], 1e-3), "max {:?}", b.max);
    }

    #[test]
    fn cylinder_honors_radius_top_and_bottom() {
        // Widest ring (radiusBottom 7) sets the x/z half-extent; height 4 → y ±2.
        let b = part_bounds(json!({
            "shape": "cylinder", "radiusTop": 2.0, "radiusBottom": 7.0, "height": 4.0
        }))
        .expect("bounds");
        assert!(close(b.min, [-7.0, -2.0, -7.0], 1e-4), "min {:?}", b.min);
        assert!(close(b.max, [7.0, 2.0, 7.0], 1e-4), "max {:?}", b.max);
    }

    #[test]
    fn tube_default_radius_is_three_not_half() {
        // No `radii`/`radius` → the reference default 3 (NOT the 0.5 primitive
        // half). Straight two-point path along X.
        let b = part_bounds(json!({
            "shape": "tube", "path": [[0, 0, 0], [10, 0, 0]]
        }))
        .expect("bounds");
        assert!(close(b.min, [-3.0, -3.0, -3.0], 1e-3), "min {:?}", b.min);
        assert!(close(b.max, [13.0, 3.0, 3.0], 1e-3), "max {:?}", b.max);
    }

    /// N3 — a bare `{shape:"tube"}` with NO `path` defaults to the renderer's
    /// `[[0,0,0],[10,0,0]]` (375 real vertices), NOT `None`. It must yield the
    /// exact reference-default AABB: the straight X segment ± the default
    /// radius 3 → x∈[-3,13], y,z∈[-3,3] — identical to the explicit-path case.
    #[test]
    fn tube_absent_path_defaults_to_reference_segment() {
        let bare = part_bounds(json!({ "shape": "tube" })).expect("bare tube bounds");
        let explicit = part_bounds(json!({
            "shape": "tube", "path": [[0, 0, 0], [10, 0, 0]]
        }))
        .expect("explicit tube bounds");
        assert!(
            close(bare.min, [-3.0, -3.0, -3.0], 1e-3),
            "min {:?}",
            bare.min
        );
        assert!(
            close(bare.max, [13.0, 3.0, 3.0], 1e-3),
            "max {:?}",
            bare.max
        );
        // Bare == explicit reference default, exactly.
        assert!(
            close(bare.min, explicit.min, 1e-6) && close(bare.max, explicit.max, 1e-6),
            "bare tube must equal the explicit reference segment"
        );
    }

    #[test]
    fn tube_short_radii_clamp_last_not_point_five() {
        // radii has ONE entry for a three-point path; the reference clamps the
        // index to the last radius (8), so every ring is r=8 — the old 0.5
        // fallback (bug) would collapse the tail to a hairline.
        let b = part_bounds(json!({
            "shape": "tube",
            "path": [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
            "radii": [8.0]
        }))
        .expect("bounds");
        // Tail point at x=20 with radius 8 → max x ≈ 28 (clamp-last), never 20.5.
        assert!(
            b.max[0] > 27.0,
            "clamp-last tail radius missing: max x {}",
            b.max[0]
        );
        assert!(
            b.max[1] > 7.5 && b.min[1] < -7.5,
            "radius 8 not applied: y {:?}",
            (b.min[1], b.max[1])
        );
    }

    #[test]
    fn tube_bounds_cover_spline_sag_beyond_control_points() {
        // A bent path whose centripetal Catmull-Rom bulges PAST the control
        // polygon. Bounds sampled on the rendered curve must exceed the naive
        // control-point ± radius union.
        let path = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [11.0, 12.0, 0.0],
        ];
        let r = 0.5f32;
        let part = json!({
            "shape": "tube",
            "path": path,
            "radii": [r]
        });
        let sampled = part_bounds(part).expect("bounds");

        // Naive control-point union ± radius (the OLD, control-points-only rule).
        let mut naive: Option<Aabb> = None;
        for p in path {
            let a = Aabb::from_center_half([p[0], p[1], p[2]], [r, r, r]);
            naive = Some(match naive {
                Some(b) => b.union(&a),
                None => a,
            });
        }
        let naive = naive.unwrap();

        // The sampled bounds CONTAIN the naive union …
        assert!(
            sampled.min[0] <= naive.min[0] + 1e-4
                && sampled.max[0] >= naive.max[0] - 1e-4
                && sampled.min[1] <= naive.min[1] + 1e-4
                && sampled.max[1] >= naive.max[1] - 1e-4,
            "sampled must contain the control union; sampled {sampled:?} naive {naive:?}"
        );
        // … and strictly overshoot it (the spline sags past the polygon in x).
        assert!(
            sampled.max[0] > naive.max[0] + 0.05,
            "spline sag not captured: sampled max x {} vs control max x {}",
            sampled.max[0],
            naive.max[0]
        );

        // N2 — EXACT arc-length bound, with a DISCRIMINATING tolerance. The
        // renderer places rings by arc length (`getPointAt`), so the true
        // rendered max x on this sag case is 11.897224426 (reference measurement
        // in three.js f64). Two competing implementations differ measurably here:
        //   • THIS f32 arc-length reparameterization:   Δ = 1.7166e-5 m
        //   • the OLD uniform-`t` sampling (the bug):    Δ = 1.76376e-4 m
        // (uniform-`t` underbounds the sag because it spaces rings by parameter,
        // not by distance). TOLERANCE DERIVATION: pick TOL between the two so the
        // test is a real discriminator, not a rubber stamp. 5e-5 sits above the
        // fixed Δ (1.7e-5, PASS) and an order of magnitude below the old Δ
        // (1.76e-4, FAIL) — a ≈10× float-noise margin over the fixed result and a
        // ≈3× guard band under the old bug. (The previous 1e-3 was NOT
        // discriminating: the old uniform-`t` Δ of 1.76e-4 < 1e-3, so it passed
        // too — the comment claiming it "fell short" of 1e-3 was false.)
        // f64 reference (an f32 literal would round away the discriminating
        // digits); the f32 result is widened for the comparison.
        const REF_MAX_X: f64 = 11.897224426;
        const TOL: f64 = 5e-5;
        let delta = (sampled.max[0] as f64 - REF_MAX_X).abs();
        assert!(
            delta < TOL,
            "arc-length max x {} not the rendered {REF_MAX_X} (Δ {delta}, tol {TOL})",
            sampled.max[0]
        );
    }

    #[test]
    fn direct_mesh_is_one_implicit_part() {
        // `mesh.parts ?? [mesh]` — a mesh carrying `shape` directly is one part.
        let b = world_bounds(
            json!({ "position": [1, 2, 3] }),
            json!({ "shape": "box", "size": [2, 4, 6] }),
        )
        .expect("bounds");
        assert!(close(b.min, [0.0, 0.0, 0.0], 1e-4), "min {:?}", b.min);
        assert!(close(b.max, [2.0, 4.0, 6.0], 1e-4), "max {:?}", b.max);
    }

    #[test]
    fn visible_false_part_is_excluded() {
        // A hidden part contributes NO geometry; only the visible part counts.
        let b = world_bounds(
            json!({ "position": [0, 0, 0] }),
            json!({ "parts": [
                { "shape": "box", "size": [2, 2, 2], "position": [0, 0, 0] },
                { "shape": "box", "size": [100, 100, 100], "position": [0, 0, 0], "visible": false }
            ] }),
        )
        .expect("bounds");
        assert!(
            close(b.min, [-1.0, -1.0, -1.0], 1e-4),
            "hidden part leaked: min {:?}",
            b.min
        );
        assert!(
            close(b.max, [1.0, 1.0, 1.0], 1e-4),
            "hidden part leaked: max {:?}",
            b.max
        );
    }
}
