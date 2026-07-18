use std::collections::BTreeMap;

use crate::physics::{Body, BodyPose, Physics};
use bytemuck::{Pod, Zeroable};
use crystal::{
    EcsWorld, Environment, Mesh, MeshPart, NumberOrNumbers, Op, QuerySpec, Spawn, Transform,
};
use elements::Triangle;
use glam::{EulerRot, Mat3, Mat4, Quat, Vec3};
use homunculus::Pose;
use kami::{BindPose, CatMind, Registry, TickContext};
use sama::{Gait, GaitParams, Locomotion, LocomotionParams, gait_pose};

use crate::player::Ground;
use serde_json::{Number, json};
use transmutation::{
    Bounds, Cluster, Dag, Mesh as ChainMesh, TransmuteParams, Vertex as ChainVertex,
    transmute_default,
};

#[derive(Clone, Debug)]
pub struct SceneParameters {
    pub fov_y_degrees: f32,
    pub near: f32,
    pub far: f32,
    pub sky_top: String,
    pub sky_horizon: String,
    pub mesh_color: String,
    pub radial_segments: u32,
    pub camera_position: [f32; 3],
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    /// Great Chain cut threshold τ (screen-space error, ~pixels). A cluster is
    /// drawn where `parent_error > τ ≥ error` projected through its group's
    /// shared LOD sphere. Smaller = finer detail held longer. A PARAM (never
    /// hardcode): env `GAIA_NATIVE_CLUSTER_ERROR`.
    pub cluster_error_threshold: f32,
    /// World-clock tick delta (seconds) for the living layer's entropy tick.
    /// A PARAM (never hardcode), default 1/60: env `GAIA_NATIVE_TICK_DT`. The
    /// tick is closed-form on the tick INDEX (entropy), never wall time.
    pub tick_dt: f64,
    /// Sun + sky-ambient defaults, overridden per-scene by the `environment`
    /// component. These feed the TRACED integrator (Rite IV) — no fake shading.
    pub sun: SunDefaults,
    /// Emissive radiance = material colour × this intensity (a dial; env
    /// `GAIA_NATIVE_EMISSIVE_INTENSITY`). Lanterns/windows glow by this much.
    pub emission_intensity: f32,
}

/// A camera pose. `yaw` turns around +Y, `pitch` is negative looking down.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    /// Unit forward vector from yaw+pitch. yaw 0 looks down -Z; pitch<0 looks down.
    pub fn direction(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        Vec3::new(
            -self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            -self.yaw.cos() * cos_pitch,
        )
    }

    /// Camera-space basis for primary-ray generation (the traced view). Returns
    /// (right, up, forward) — a right-handed orthonormal frame, forward = the
    /// look direction. `right`/`up` span the image plane; a pixel ray is
    /// `forward + right·sx·tan(fov/2)·aspect + up·sy·tan(fov/2)` with
    /// sx,sy ∈ [-1,1].
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = self.direction();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let right = if right.length_squared() < 1e-8 {
            Vec3::X
        } else {
            right
        };
        let up = right.cross(forward).normalize_or_zero();
        (right, up, forward)
    }
}

/// Resolved sun + sky-ambient for the traced integrator. The sun is a
/// directional (delta) emitter reached by a shadow ray (next-event); the sky
/// ambient scales the sky-gradient environment gathered by escaped rays.
/// There is NO fake constant floor (GRIMOIRE: unlit is truly unlit).
#[derive(Clone, Copy, Debug)]
pub struct SunLight {
    /// Unit direction TOWARD the sun.
    pub direction: [f32; 3],
    /// Sun colour (linear rgb).
    pub color: [f32; 3],
    /// Sun radiance scale.
    pub intensity: f32,
    /// Sky-ambient scale applied to the sky-gradient environment radiance.
    pub ambient_intensity: f32,
}

/// Env-parameterised sun/sky defaults (never hardcoded at the shading site).
#[derive(Clone, Debug)]
pub struct SunDefaults {
    pub sun_color: String,
    pub sun_intensity: f32,
    pub sun_position: [f32; 3],
    pub ambient_intensity: f32,
}

impl SunLight {
    /// Read `environment.sun` / `environment.hemisphere` when present, else
    /// defaults. `sun.position` is a point the sun sits at → direction toward it.
    pub fn derive(
        environment: Option<&Environment>,
        defaults: &SunDefaults,
    ) -> Result<Self, String> {
        let sun = environment.and_then(|environment| environment.sun.as_ref());
        let hemisphere = environment.and_then(|environment| environment.hemisphere.as_ref());

        let sun_color = sun_string(sun, "color").unwrap_or(&defaults.sun_color);
        let color = linear_rgb(sun_color)?;
        let intensity =
            sun_number(sun, "intensity").unwrap_or(defaults.sun_intensity as f64) as f32;
        let position = sun_vec3(sun, "position").unwrap_or(Vec3::from_array(defaults.sun_position));
        let direction = position.normalize_or_zero();
        let direction = if direction.length_squared() < 1e-8 {
            Vec3::Y
        } else {
            direction
        };
        let ambient_intensity =
            sun_number(hemisphere, "intensity").unwrap_or(defaults.ambient_intensity as f64) as f32;

        Ok(Self {
            direction: direction.to_array(),
            color,
            intensity,
            ambient_intensity,
        })
    }
}

fn sun_string<'a>(value: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    value?.get(key)?.as_str()
}
fn sun_number(value: Option<&serde_json::Value>, key: &str) -> Option<f64> {
    value?.get(key)?.as_f64()
}
fn sun_vec3(value: Option<&serde_json::Value>, key: &str) -> Option<Vec3> {
    let array = value?.get(key)?.as_array()?;
    let numbers: Vec<Number> = array
        .iter()
        .filter_map(|item| item.as_f64().and_then(Number::from_f64))
        .collect();
    vec3(Some(&numbers))
}

/// One world-space leaf triangle carrying its material — the EXACT geometry the
/// traced integrator intersects (view-independent, error 0). `albedo` is the
/// lambertian reflectance (ZERO for a pure emitter, matching the Pleroma); `emission`
/// is the radiance the surface glows with (material colour × emission intensity,
/// ZERO for a non-emitter). `metallic`/`roughness` carry the L2 conductor lobe
/// (defaults 0/1 = pure lambertian — see the Pleroma `Material`).
#[derive(Clone, Copy, Debug)]
pub struct LeafTriangle {
    pub positions: [[f32; 3]; 3],
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
}

impl LeafTriangle {
    /// Construct with the default lambertian material dials (metallic 0,
    /// roughness 1) — the L0/L1 surface, so existing call sites read unchanged.
    pub fn lambertian(positions: [[f32; 3]; 3], albedo: [f32; 3], emission: [f32; 3]) -> Self {
        Self {
            positions,
            albedo,
            emission,
            metallic: 0.0,
            roughness: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub emissive: f32,
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 36,
                    shader_location: 3,
                },
            ],
        }
    }
}

/// One transmuted material batch: a Great Chain (the SOLE geometry path) plus
/// the flat colour/emissive its clusters draw with. Geometry stays generic —
/// the chain knows nothing of colour; colour rides the batch, not the vertex
/// stream, so identical geometry across colours never fragments the chain.
pub struct MaterialChain {
    pub dag: Dag,
    pub color: [f32; 3],
    pub emissive: f32,
    /// L2 conductor lobe: metallic `[0,1]` (0 = lambertian default).
    pub metallic: f32,
    /// L2 conductor lobe: roughness `[0,1]` (1 = lambertian default).
    pub roughness: f32,
}

pub struct RenderScene {
    /// Camera derived from the world `spawn`; the moving eye overrides it per request.
    pub camera: Camera,
    pub sky_top: [f32; 4],
    pub sky_horizon: [f32; 4],
    /// Traced sun + sky-ambient (Rite IV — replaces the deleted First Light).
    pub sun: SunLight,
    /// Per-material transmuted Great Chains. THE geometry path: every draw is a
    /// view-dependent cluster cut over these (the W1/W2 forward per-primitive
    /// path is gone).
    pub chains: Vec<MaterialChain>,
    /// Great Chain cut threshold τ (screen-space error), carried from params.
    pub error_threshold: f32,
    /// Emissive radiance scale (material colour × this = emission).
    pub emission_intensity: f32,
    /// The LIVING LAYER (Rite IV dynamics): every entity carrying a `behavior`
    /// component. Excluded from the static `chains` (and so from the STATIC BVH),
    /// each keeps its own bind-baked leaf triangles + a live model transform re-
    /// derived every world tick. Its transformed triangles form the per-tick
    /// DYNAMIC partition the traced BVH splices in ([`Dynamics`]).
    pub dynamics: Dynamics,
    /// RITE V — THE EMBODIED ONES. Every entity carrying a `body` sigil, skinned
    /// into world-space triangles at SAMA's canonical idle pose (tick 0). Like
    /// the living layer, these are excluded from the static chains and re-fed to
    /// the traced BVH every tick (`dynamic_leaf_triangles`) — the seam V1 drives
    /// with a per-tick sama pose. GENERIC: any creature that names a preset.
    pub bodies: Vec<BodyInstance>,
    /// RITE V FINAL WELD — the realm floor (the SAME post-transmute leaf ground
    /// the walker stands on, F1), kept so a walker-ATTACHED body re-grounds under
    /// the walker's roaming xz every tick (`command_bodies_walked`).
    floor: Ground,
    /// OWN-BODY CULL — the last WalkerPose xz fed to `command_bodies_walked`
    /// (`None` until the first walker-driven tick). The identity a render's
    /// `eye` is compared against in `dynamic_leaf_triangles_for_eye`: a
    /// walker-attached body is the walker's OWN body, so drawing it from that
    /// exact eye is drawing it inside the camera (the Architect-blinding bug,
    /// 31bae8b). Any OTHER eye (a moving `/scry?pos=...`, a diorama camera)
    /// still sees her — only identity with this pose culls.
    last_walker_eye: Option<Vec3>,
}

/// OWN-EYE CULL default epsilon (metres) for `dynamic_leaf_triangles_for_eye`:
/// how close a render `eye` must sit to the last walker pose to count as "the
/// walker's own eye" (vs. a moving eye that merely visits the same spot).
/// Sized to swallow f32 pose round-trip noise (sub-millimetre) while staying
/// far below a single skinned vessel's own scale (metres) — never confusable
/// with a real moving-eye shot taken from where the walker stands.
pub const OWN_EYE_EPSILON_M: f32 = 1e-3;

/// The walker's world pose fed to walker-ATTACHED bodies each tick (RITE V FINAL
/// WELD). `position` is the EYE pose (the body re-grounds under its xz, like
/// spawn); `yaw` is the facing. A body carrying `follows: "walker"` re-places
/// onto this pose and derives its gait speed from the per-tick displacement.
#[derive(Clone, Copy, Debug)]
pub struct WalkerPose {
    /// The walker's eye position this tick (xz drives placement; y is ignored —
    /// the body's y is re-derived from the floor).
    pub position: Vec3,
    /// The walker's facing yaw (radians about +Y).
    pub yaw: f32,
}

/// One embodied vessel: a realm entity's `body` skinned to world-space leaf
/// triangles. `preset` is the named vessel preset; the triangles carry
/// lambertian albedo from the body's per-vertex colours.
///
/// RITE V · V1 — SHE WALKS. The instance carries a SAMA [`Locomotion`] state
/// machine and the composed [`vessel::Body`]; [`BodyInstance::command`] advances
/// the machine one tick against a commanded speed (the walker's velocity),
/// takes the pose the state machine emits, and re-skins `world_tris` FROM THAT
/// EXACT POSE — sama is the sole pose source and its pose IS the skinning input
/// (the 0e0 ordeal). GENERIC: any creature that names a preset animates the same
/// way.
#[derive(Clone, Debug)]
pub struct BodyInstance {
    /// The owning entity's gaia id.
    pub gaia_id: String,
    /// The named vessel preset (`body.preset`).
    pub preset: String,
    /// World-space posed triangles carrying their albedo (the current tick).
    pub world_tris: Vec<LeafTriangle>,
    /// The composed body (skeleton + skinned vessel + colours + idle pose).
    body: vessel::Body,
    /// Per-vertex LINEAR-rgb albedo, parallel to the vessel mesh vertices.
    albedo: Vec<Vec3>,
    /// The GROUNDED world transform: authored rotation/scale/xz with the y
    /// derived so the lowest contact vertex rests on the realm floor.
    model: Mat4,
    /// The SAMA locomotion state machine — the SOLE pose source.
    locomotion: Locomotion,
    /// The pose the state machine emitted for the CURRENT `world_tris` (the
    /// skinning input — same object, so pose == skinning input is exact).
    pose: Pose,
    /// The commanded speed last fed to the state machine (walker velocity).
    commanded_speed: f32,
    /// RITE V · V2 — the behavior spirit, if this body carries a `behavior`
    /// `{kind:"cat"}`. A minded body drives its OWN commanded speed + world xz
    /// from the clock (its idle loop), ignoring the walker's velocity; a mindless
    /// body (nari) is still driven by the walker. `None` = no behavior.
    mind: Option<CatMind>,
    /// The DERIVED grounded y (paws on the floor) — held constant as the minded
    /// body walks its flat circuit, so grounding never drifts on the move.
    ground_y: f32,
    /// The authored scale — kept to rebuild the model each animated tick (the
    /// mind drives position + yaw; scale is authored).
    base_scale: Vec3,
    /// RITE V FINAL WELD — the ATTACHMENT sigil (`body.follows`), a plain-English
    /// parameter naming what this body tracks. `Some("walker")` binds the body to
    /// THE WALKER: its position/yaw track the walker each tick and its gait speed
    /// is DERIVED from actual horizontal displacement (not the global broadcast).
    /// The engine never special-cases a creature — any body may declare it.
    /// `None` = unattached (walker-broadcast, or minded).
    follows: Option<String>,
}

impl BodyInstance {
    /// The pose SAMA emitted for the current skinned triangles — the exact
    /// skinning input (the 0e0 ordeal proves `world_tris` skins THIS pose).
    pub fn pose(&self) -> &Pose {
        &self.pose
    }

    /// The commanded speed (walker velocity magnitude) last driven in.
    pub fn commanded_speed(&self) -> f32 {
        self.commanded_speed
    }

    /// The locomotion state (idle / walk / run) after the last command.
    pub fn gait(&self) -> Gait {
        self.locomotion.state()
    }

    /// Whether the body is animating this tick — a non-idle gait or a live
    /// cross-fade. An idle, settled body holds the bind pose (static geometry),
    /// so the dynamic BVH need not re-splice for it.
    pub fn is_animating(&self) -> bool {
        self.locomotion.state() != Gait::Idle || self.locomotion.blending()
    }

    /// Advance SAMA one fixed tick against `commanded_speed` (the walker's
    /// velocity) and re-skin `world_tris` from the pose it emits. The pose is
    /// taken ONCE and fed straight into the skinner — sama's pose IS the
    /// skinning input (Rite V·V1). Deterministic in the tick + command stream.
    pub fn command(&mut self, commanded_speed: f32) {
        self.commanded_speed = commanded_speed;
        self.pose = self.locomotion.step(&self.body.skeleton, commanded_speed);
        self.world_tris = skin_body(&self.body, &self.pose, self.model, &self.albedo);
    }

    /// RITE V · V2 — advance a MINDED body one tick against the world clock time
    /// `t = clock · dt`. The cat's idle loop derives (position, yaw, speed); the
    /// model is rebuilt from the grounded circuit position + heading, SAMA is
    /// commanded with the loop's speed (idle when sitting, walk on the circuit),
    /// and `world_tris` re-skin from the emitted pose. A no-op with no mind.
    /// Deterministic in `(t, mind)` — same clock, byte-identical body.
    pub fn animate(&mut self, t: f64) {
        let Some(mind) = self.mind else {
            return;
        };
        let drive = mind.drive(t);
        let position = Vec3::new(
            drive.position[0] as f32,
            self.ground_y,
            drive.position[2] as f32,
        );
        self.model = transform_matrix(
            position,
            Vec3::new(0.0, drive.yaw as f32, 0.0),
            self.base_scale,
        );
        self.commanded_speed = drive.speed as f32;
        self.pose = self
            .locomotion
            .step(&self.body.skeleton, self.commanded_speed);
        self.world_tris = skin_body(&self.body, &self.pose, self.model, &self.albedo);
    }

    /// Whether this body carries a behavior spirit (a minded body drives itself
    /// from the clock; a mindless one from the walker).
    pub fn is_minded(&self) -> bool {
        self.mind.is_some()
    }

    /// RITE V FINAL WELD — whether this body is ATTACHED to the walker
    /// (`follows: "walker"`): its pose tracks the walker and its gait derives
    /// from displacement, not the broadcast.
    pub fn follows_walker(&self) -> bool {
        self.follows.as_deref() == Some("walker")
    }

    /// RITE V FINAL WELD — track THE WALKER one tick: re-place onto the walker's
    /// world pose (re-grounded on `floor`), derive the gait speed from the
    /// horizontal displacement between where the body stood (its current model
    /// origin) and the walker's new xz over `dt`, step SAMA against it, re-skin.
    /// The N2 [`BodyInstance::drive`] exemplar with a WALKER-DISPLACEMENT velocity
    /// source — attachment is the only difference from a wired presence.
    pub fn follow_walker(&mut self, walker: WalkerPose, dt: f32, floor: Option<&Ground>) {
        let prev = self.model.w_axis.truncate();
        let speed = if dt <= 0.0 {
            0.0
        } else {
            let dx = walker.position.x - prev.x;
            let dz = walker.position.z - prev.z;
            (dx * dx + dz * dz).sqrt() / dt
        };
        let position = [
            walker.position.x as f64,
            walker.position.y as f64,
            walker.position.z as f64,
        ];
        self.drive(position, walker.yaw as f64, speed, floor);
    }

    /// The body's current world-space origin (the model's translation column) —
    /// where the grounded body stands this tick. A minded body's xz rides its
    /// idle-loop circuit; y is the derived grounded height.
    pub fn world_origin(&self) -> [f32; 3] {
        self.model.w_axis.truncate().to_array()
    }

    /// Re-skin the current pose into world-space triangles — the pure skinning
    /// step, exposed so an ordeal can prove `world_tris` IS this pose skinned
    /// (byte-identical, 0e0).
    pub fn skin_current(&self) -> Vec<LeafTriangle> {
        skin_body(&self.body, &self.pose, self.model, &self.albedo)
    }

    /// Compose a fresh body from a NAMED preset at an authored world pose — the
    /// GENERIC remote/presence constructor (no ECS, no realm `body` sigil). A
    /// wired presence (id + interpolated position/yaw) drives one of these so it
    /// renders as a real body in the traced world (Wired N2). `floor` grounds the
    /// y (lowest contact vertex rests on the surface under `(x, z)`); `None`
    /// keeps the authored y. Starts at SAMA's idle pose (commanded speed 0). An
    /// unknown preset is an error — nothing is invented.
    pub fn from_preset(
        gaia_id: impl Into<String>,
        preset_name: &str,
        position: [f64; 3],
        yaw: f64,
        floor: Option<&Ground>,
    ) -> Result<Self, String> {
        let preset = vessel::Preset::by_name(preset_name)
            .ok_or_else(|| format!("unknown preset {preset_name:?}"))?;
        let composed = vessel::Body::from_preset(&preset);
        let idle = composed.idle_pose.clone();
        let albedo = composed.vertex_albedo();
        let pos = Vec3::new(position[0] as f32, position[1] as f32, position[2] as f32);
        let model = ground_model(&composed, pos, yaw as f32, floor);
        let world_tris = skin_body(&composed, &idle, model, &albedo);
        Ok(Self {
            gaia_id: gaia_id.into(),
            preset: preset_name.to_string(),
            world_tris,
            body: composed,
            albedo,
            model,
            locomotion: Locomotion::new(LocomotionParams::default()),
            pose: idle,
            commanded_speed: 0.0,
            // A presence body is MINDLESS — the wire drives it, not the clock.
            mind: None,
            // The grounded y ground_model just derived (model translation y).
            ground_y: model.w_axis.y,
            // No authored scale on a wire-composed body.
            base_scale: Vec3::ONE,
            // A wire-composed presence body is not walker-attached (N2 drives it).
            follows: None,
        })
    }

    /// Drive a presence body to a NEW world pose this tick: re-place the model
    /// (re-grounded on `floor`), step SAMA against `commanded_speed` (the
    /// presence's derived ground speed), and re-skin. The SINGLE presence-drive
    /// path — the position tracks the interp buffer, the gait animates from the
    /// derived speed. Deterministic in the (pose, speed, floor) stream.
    pub fn drive(
        &mut self,
        position: [f64; 3],
        yaw: f64,
        commanded_speed: f32,
        floor: Option<&Ground>,
    ) {
        let pos = Vec3::new(position[0] as f32, position[1] as f32, position[2] as f32);
        self.model = ground_model(&self.body, pos, yaw as f32, floor);
        self.commanded_speed = commanded_speed;
        self.pose = self.locomotion.step(&self.body.skeleton, commanded_speed);
        self.world_tris = skin_body(&self.body, &self.pose, self.model, &self.albedo);
    }

    /// The current grounded world model (rotation/scale/translation) — exposed so
    /// an ordeal can read the body's world transform directly.
    pub fn model(&self) -> Mat4 {
        self.model
    }
}

/// The GROUNDED world model for a composed body at `(position, yaw)`: yaw about
/// +Y, unit scale, and the y derived so the lowest CONTACT vertex rests on
/// `floor` under `(x, z)`. `None` floor (body over the void) keeps the authored
/// y verbatim. The SINGLE grounding path shared by the presence constructor and
/// per-tick drive — the y is derived, never eye-nudged (Guardian finding 1).
fn ground_model(body: &vessel::Body, position: Vec3, yaw: f32, floor: Option<&Ground>) -> Mat4 {
    let rotation = Vec3::new(0.0, yaw, 0.0);
    let mesh = body.vessel.posed(&body.skeleton, &body.idle_pose);
    let orient = transform_matrix(Vec3::ZERO, rotation, Vec3::ONE);
    let contact_local = lowest_contact_y(body, &mesh, orient);
    let grounded_y = floor
        .and_then(|f| f.height_at(position.x, position.z, f32::INFINITY))
        .map(|g| g - contact_local)
        .unwrap_or(position.y);
    transform_matrix(
        Vec3::new(position.x, grounded_y, position.z),
        rotation,
        Vec3::ONE,
    )
}

/// Skin a composed body at `pose` into world-space [`LeafTriangle`]s. Pure: the
/// vessel deforms the bound mesh by `pose` (SAMA's output), the model places it
/// in the realm, each triangle takes the mean of its three vertices' linear
/// albedo. The SINGLE skinning path — compose and every per-tick command call it
/// with the pose SAMA emitted, so the pose is always the skinning input.
fn skin_body(body: &vessel::Body, pose: &Pose, model: Mat4, albedo: &[Vec3]) -> Vec<LeafTriangle> {
    let mesh = body.vessel.posed(&body.skeleton, pose);
    mesh.indices
        .chunks_exact(3)
        .map(|tri| {
            let corner = |i: u32| {
                model
                    .transform_point3(mesh.positions[i as usize])
                    .to_array()
            };
            let mean =
                (albedo[tri[0] as usize] + albedo[tri[1] as usize] + albedo[tri[2] as usize]) / 3.0;
            LeafTriangle::lambertian(
                [corner(tri[0]), corner(tri[1]), corner(tri[2])],
                mean.to_array(),
                [0.0, 0.0, 0.0],
            )
        })
        .collect()
}

/// Material batch key: quantised linear colour bits + emissive flag +
/// metallic/roughness bits. Ordered so the chain vector is deterministic
/// (byte-identical double builds); distinct metallic/roughness never merge.
type MatKey = ([u32; 3], u32, u32, u32);

struct MatBucket {
    /// World-space triangle soup (position/normal/uv); transmuted at seal.
    vertices: Vec<ChainVertex>,
    color: [f32; 3],
    emissive: f32,
    metallic: f32,
    roughness: f32,
}

impl RenderScene {
    /// Materialize the render scene from an OWNED ECS. The living layer
    /// ([`Dynamics`]) takes ownership of the world so its per-tick clock can read
    /// `behavior` and write the animated `transform`s back (senses/pose then read
    /// the moving world). Static geometry seals into the shared chains exactly as
    /// before; behavior-carriers split off into their own bind-baked chains.
    pub fn from_ecs(world: EcsWorld, parameters: &SceneParameters) -> Result<Self, String> {
        Self::from_ecs_at(world, parameters, [0.0, 0.0, 0.0])
    }

    /// As [`RenderScene::from_ecs`], but any `terrain` sigil's generated patch
    /// is placed relative to `render_origin` (world-space meters) rather than
    /// the world origin — VII-0b's COORDINATE SEAM. The code path is always
    /// `i64` tile coords -> `f64` tile origin ([`seed::tile_origin_m`]) ->
    /// subtract `render_origin` (still `f64`, exact regardless of either
    /// operand's magnitude — IEEE754 subtraction of two equal-magnitude
    /// values is exact) -> cast to `f32` ONLY once the residual offset is
    /// small (see `terrain_placement_offset`'s doc). A realm with no
    /// `terrain` sigil never reads `render_origin` at all. `from_ecs` is
    /// `render_origin = [0,0,0]` (the realm origin — every existing caller's
    /// unchanged behavior, and VII-0b's own "the realm sits near origin"
    /// case).
    pub fn from_ecs_at(
        mut world: EcsWorld,
        parameters: &SceneParameters,
        render_origin: [f64; 3],
    ) -> Result<Self, String> {
        let world = &mut world;
        if !(parameters.fov_y_degrees > 0.0 && parameters.fov_y_degrees < 180.0) {
            return Err("GAIA_NATIVE_FOV must be between 0 and 180 degrees".into());
        }
        if parameters.near <= 0.0 || parameters.far <= parameters.near {
            return Err("GAIA_NATIVE_NEAR must be positive and less than GAIA_NATIVE_FAR".into());
        }
        if parameters.radial_segments < 3 {
            return Err("GAIA_NATIVE_RADIAL_SEGMENTS must be at least 3".into());
        }

        let spawn = first_component::<Spawn>(world, "spawn")?;
        let eye = spawn
            .as_ref()
            .and_then(|spawn| vec3(spawn.position.as_ref()))
            .unwrap_or(Vec3::from_array(parameters.camera_position));
        let yaw = spawn
            .as_ref()
            .and_then(|spawn| number(spawn.yaw.as_ref()))
            .unwrap_or(parameters.camera_yaw);
        let camera = Camera {
            eye,
            yaw,
            pitch: parameters.camera_pitch,
            fov_y_radians: parameters.fov_y_degrees.to_radians(),
            near: parameters.near,
            far: parameters.far,
        };

        let environment = first_component::<Environment>(world, "environment")?;
        let sky_top = environment
            .as_ref()
            .and_then(|environment| environment.sky.as_ref())
            .and_then(|sky| sky.top.as_deref())
            .unwrap_or(&parameters.sky_top);
        let sky_horizon = environment
            .as_ref()
            .and_then(|environment| environment.sky.as_ref())
            .and_then(|sky| sky.horizon.as_deref())
            .unwrap_or(&parameters.sky_horizon);
        let sky_top = linear_rgba(sky_top)?;
        let sky_horizon = linear_rgba(sky_horizon)?;
        let sun = SunLight::derive(environment.as_ref(), &parameters.sun)?;
        let default_color = linear_rgb(&parameters.mesh_color)?;

        // Entities carrying a `behavior` component OR a `body` component are
        // DYNAMIC: split off from the shared static chains into their own
        // (bind-baked) chains + a live model transform. Generic — the split
        // reads only the `behavior`/`body` markers, never a realm's vocabulary.
        // `behavior` = KAMI kinematics; `body` = the Elements' rigid solver.
        let behavior_id = world.component_id("behavior");
        let body_id = world.component_id("body");
        // Physics declarations (id, body sigil, authored world centre) collected
        // as we walk the entities; installed into the living layer after the
        // static collider is sealed.
        let mut body_declarations: Vec<(String, Body, [f64; 3])> = Vec::new();
        let render_components = world
            .component_id("transform")
            .zip(world.component_id("mesh"));
        let mut entities = render_components
            .map(|(transform, mesh)| {
                world.query(&QuerySpec {
                    all: vec![transform, mesh],
                    ..Default::default()
                })
            })
            .unwrap_or_default();
        entities.sort_by(|a, b| world.gaia_id_for(*a).cmp(&world.gaia_id_for(*b)));

        // Tessellate every mesh part into world-space triangles. Static parts
        // pool into shared material buckets; each dynamic entity seals its OWN.
        let mut static_buckets = BTreeMap::<MatKey, MatBucket>::new();
        // The physics collision soup: every STATIC part's world-space triangles
        // with clean per-face outward normals (accumulated as we tessellate).
        let mut collider_triangles: Vec<Triangle> = Vec::new();
        let mut dynamics = Dynamics::new(parameters.emission_intensity);
        for entity in entities {
            let (transform_id, mesh_id) = render_components.expect("render query has components");
            let id = world.gaia_id_for(entity).unwrap_or("<unbound>").to_string();
            let transform: Transform =
                serde_json::from_value(world.get_component(entity, transform_id)?)
                    .map_err(|error| format!("entity {id:?} transform: {error}"))?;
            let mesh: Mesh = serde_json::from_value(world.get_component(entity, mesh_id)?)
                .map_err(|error| format!("entity {id:?} mesh: {error}"))?;
            let parts = parts_of(mesh).map_err(|error| format!("entity {id:?} mesh: {error}"))?;
            let bind_position = vec3(transform.position.as_ref()).unwrap_or(Vec3::ZERO);
            let bind_rotation = vec3(transform.rotation.as_ref()).unwrap_or(Vec3::ZERO);
            let bind_scale = scale(transform.scale.as_ref());
            let entity_model = transform_matrix(bind_position, bind_rotation, bind_scale);

            let has_behavior = behavior_id
                .map(|behavior| world.get_component(entity, behavior).is_ok())
                .unwrap_or(false);
            let body = body_id.and_then(|id| world.get_component(entity, id).ok());
            if let Some(raw) = &body {
                let has_preset = raw.get("preset").is_some();
                let has_shape = raw.get("shape").is_some();
                // F2 — a body is EITHER a skinned vessel (`preset`, driven by the
                // RITE V weld) OR a rigid solver body (`shape`), NEVER both.
                // `{preset, shape}` is a LOUD authoring error: name the entity,
                // refuse the realm (no silent double-driver).
                if has_preset && has_shape {
                    return Err(format!(
                        "entity {id:?} body declares BOTH `preset` (a skinned vessel) and \
                         `shape` (a rigid solver body) — a body is one or the other; \
                         refusing the realm"
                    ));
                }
                // preset ⇒ the rigid solver SKIPS it, ALWAYS. Only a non-preset
                // body enters `body_declarations`, so a meshed preset entity can
                // never become a double-driver (a default rigid AND a skinned
                // vessel) — the latent fork the adversary named is closed.
                if !has_preset {
                    let parsed: Body = serde_json::from_value(raw.clone())
                        .map_err(|error| format!("entity {id:?} body: {error}"))?;
                    body_declarations.push((
                        id.clone(),
                        parsed,
                        bind_position.as_dvec3().to_array(),
                    ));
                }
            }
            let is_dynamic = has_behavior || body.is_some();

            if is_dynamic {
                let mut buckets = BTreeMap::<MatKey, MatBucket>::new();
                for (index, part) in parts.iter().enumerate() {
                    append_part(
                        &mut buckets,
                        part,
                        entity_model,
                        default_color,
                        parameters.radial_segments,
                        None,
                    )
                    .map_err(|error| format!("entity {id:?} mesh part {index}: {error}"))?;
                }
                let chains =
                    seal_buckets(buckets).map_err(|error| format!("entity {id:?}: {error}"))?;
                let bind = BindPose {
                    position: bind_position.as_dvec3().to_array(),
                    rotation: bind_rotation.as_dvec3().to_array(),
                    scale: bind_scale.as_dvec3().to_array(),
                    intensity: 1.0,
                };
                dynamics.push(
                    &id,
                    chains,
                    bind,
                    entity_model,
                    parameters.emission_intensity,
                );
            } else {
                for (index, part) in parts.iter().enumerate() {
                    append_part(
                        &mut static_buckets,
                        part,
                        entity_model,
                        default_color,
                        parameters.radial_segments,
                        Some(&mut collider_triangles),
                    )
                    .map_err(|error| format!("entity {id:?} mesh part {index}: {error}"))?;
                }
            }
        }

        // VII-0b — THE FIRST GROUND, the render weld. A `terrain` sigil is
        // authored ONLY as (seed, tile_x, tile_y, optional dial overrides) —
        // NO stored geometry (the NO-STORAGE ordeal) — generated here, at
        // load, through VII-0a's `seed::tile_mesh`, and fed into the SAME
        // seal path every other static part rides (its own material chain;
        // sealed through the Great Chain like all matter). A terrain entity
        // carries no `mesh` component (asserted below); its world placement
        // is DERIVED from `tile_x`/`tile_y` × `tile_size_m`, never restated
        // by an authored `transform` (asserted-consistent if one exists).
        if let Some(terrain_id) = world.component_id("terrain") {
            let mesh_id = world.component_id("mesh");
            let transform_id = world.component_id("transform");
            let mut terrain_entities = world.query(&QuerySpec {
                all: vec![terrain_id],
                ..Default::default()
            });
            terrain_entities.sort_by(|a, b| world.gaia_id_for(*a).cmp(&world.gaia_id_for(*b)));
            for entity in terrain_entities {
                let id = world.gaia_id_for(entity).unwrap_or("<unbound>").to_string();
                if let Some(mesh_id) = mesh_id
                    && world.get_component(entity, mesh_id).is_ok()
                {
                    return Err(format!(
                        "entity {id:?} carries both `terrain` and `mesh` — a \
                         terrain patch is authored ONLY as a sigil (seed + \
                         tile coords + params); no stored geometry is \
                         permitted alongside it"
                    ));
                }
                let raw = world.get_component(entity, terrain_id)?;
                let sigil: seed::TerrainSigil = serde_json::from_value(raw)
                    .map_err(|error| format!("entity {id:?} terrain: {error}"))?;
                let tile = sigil.tile();
                let params = sigil.params();
                let world_seed = sigil.world_seed();
                let tile_origin = seed::tile_origin_m(tile, &params);

                // Position is DERIVED from tile coords — an authored
                // transform must agree with the derivation, never restate it
                // independently (silent divergence would desync the render
                // placement from the physics/oracle placement).
                if let Some(transform_id) = transform_id
                    && let Ok(raw_transform) = world.get_component(entity, transform_id)
                {
                    let transform: Transform = serde_json::from_value(raw_transform)
                        .map_err(|error| format!("entity {id:?} transform: {error}"))?;
                    if let Some(position) = vec3(transform.position.as_ref()) {
                        let expected = Vec3::new(tile_origin.0 as f32, 0.0, tile_origin.1 as f32);
                        // Derived tolerance: f32 ULP error on the largest
                        // magnitude coordinate involved, generously
                        // margined (8x) for the f64->f32 cast plus the
                        // subtraction below — never a plucked epsilon.
                        let scale = expected.abs().max_element().max(1.0);
                        let tolerance = scale * f32::EPSILON * 8.0;
                        if (position - expected).length() > tolerance {
                            return Err(format!(
                                "entity {id:?} terrain: authored transform.position \
                                 {:?} does not match the tile-derived origin {:?} \
                                 (tolerance {tolerance}) — the position is DERIVED \
                                 from tile_x/tile_y × tile_size_m, never restated \
                                 (drop the transform or fix the mismatch)",
                                position.to_array(),
                                expected.to_array()
                            ));
                        }
                    }
                }

                let mesh = seed::tile_mesh(world_seed, tile, &params);
                let offset = terrain_placement_offset(tile_origin, render_origin);
                let color = match sigil.color.as_deref() {
                    Some(hex) => linear_rgb(hex)?,
                    None => default_color,
                };
                append_terrain(
                    &mut static_buckets,
                    &mesh,
                    offset,
                    color,
                    &mut collider_triangles,
                );
            }
        }

        // Seal the shared static buckets into the transmuted Great Chains — THE
        // one geometry path — before anything grounds on the realm.
        let chains = seal_buckets(static_buckets)?;

        // RITE V weld (F1 — ONE FLOOR): read the embodied ones (`body` sigil)
        // into world-space skinned triangles BEFORE the dynamics consume the
        // ECS. The bodies GROUND onto the SAME POST-TRANSMUTE leaf floor the
        // walker walks on — `leaf_positions_of(&chains)` is byte-for-byte the
        // `Vec` `RenderScene::leaf_positions` hands the walker in `main.rs`
        // (same function, same sealed chains), so there is a SINGLE floor
        // source: no pre/post-transmute fork, no silent divergence under a
        // future weld tolerance. The y is derived, never eye-nudged.
        let floor = Ground::from_positions(&leaf_positions_of(&chains));
        let bodies = body_instances(world, &floor)?;

        // The dynamics take ownership of the ECS (its live tick reads and writes
        // the animated transforms).
        dynamics.install_world(std::mem::take(world), parameters);
        // Wire the Elements' rigid solver for every declared body (inert — stays
        // `None` — when no `body` is declared; a zero-physics realm is unchanged).
        dynamics.install_physics(body_declarations, collider_triangles, parameters.tick_dt);

        Ok(Self {
            camera,
            sky_top,
            sky_horizon,
            sun,
            chains,
            error_threshold: parameters.cluster_error_threshold,
            emission_intensity: parameters.emission_intensity,
            dynamics,
            bodies,
            // RITE V FINAL WELD — keep this SAME floor so a walker-attached body
            // re-grounds under the walker's roaming xz every tick (one floor, F1).
            floor,
            // OWN-BODY CULL — no walker pose has driven a tick yet; every body
            // draws (matches the pre-weld / headless-test fallback path).
            last_walker_eye: None,
        })
    }

    /// Advance the world clock one fixed tick (Flow of Data): KAMI reads the ECS
    /// → emits transform ops → they apply to the ECS → each dynamic entity's
    /// model transform is re-derived from its (now animated) transform.
    /// Deterministic in the tick count — never wall time. Call once per frame;
    /// the traced BVH then re-splices the dynamic partition (`main.rs`).
    pub fn tick(&mut self) {
        self.dynamics.tick();
    }

    /// [`RenderScene::tick`], plus a batch of incantation ops folded into the
    /// physics block before the solver steps this tick (see
    /// [`Dynamics::tick_with_ops`] — currently only [`Op::Impulse`] acts).
    pub fn tick_with_ops(&mut self, ops: &[Op]) {
        self.dynamics.tick_with_ops(ops);
    }

    /// RITE V · V1 — drive every embodied body one fixed tick against the
    /// walker's `commanded_speed` (the embodiment velocity magnitude). Each body
    /// steps its SAMA state machine, takes the emitted pose, and re-skins its
    /// `world_tris` from it. Call once per frame BEFORE re-splicing the dynamic
    /// partition. Returns whether any body is animating (so the caller knows the
    /// dynamic BVH must re-splice even when the living models are still).
    pub fn command_bodies(&mut self, commanded_speed: f32) -> bool {
        self.command_bodies_walked(commanded_speed, None)
    }

    /// RITE V FINAL WELD — drive every embodied body one fixed tick WITH the
    /// walker's pose, so walker-ATTACHED bodies (`follows: "walker"`) track it.
    /// Per-body velocity SOURCES, complete:
    /// - walker-ATTACHED + `Some(walker)` → [`BodyInstance::follow_walker`]: the
    ///   body re-places onto the walker's pose (re-grounded on the realm floor)
    ///   and its gait speed is DERIVED from the per-tick displacement — NOT the
    ///   `commanded_speed` broadcast (killed for attached bodies).
    /// - MINDED (the cat) → its own clock idle loop (V2, unchanged).
    /// - unattached MINDLESS (or an attached body with no walker pose, e.g. a
    ///   headless test) → the `commanded_speed` broadcast (legacy path).
    ///
    /// (A wired PRESENCE body is driven by [`BodyInstance::drive`] from the wire,
    /// N2 — that path never enters here.) Returns whether any body is animating.
    pub fn command_bodies_walked(
        &mut self,
        commanded_speed: f32,
        walker: Option<WalkerPose>,
    ) -> bool {
        // OWN-BODY CULL — remember THIS tick's walker eye (identity for
        // `dynamic_leaf_triangles_for_eye`'s own-eye test) before anything else
        // moves; a headless caller that never passes `walker` leaves it `None`
        // forever (every body draws — the pre-weld / broadcast-only fallback).
        if let Some(pose) = walker {
            self.last_walker_eye = Some(pose.position);
        }
        // The clock is the living layer's tick count × dt (read BEFORE `tick()`
        // increments it, so frame N uses tick N).
        let t = self.dynamics.clock as f64 * self.dynamics.dt;
        let dt = self.dynamics.dt as f32;
        let floor = &self.floor;
        let mut animating = false;
        for body in &mut self.bodies {
            match (
                body.follows_walker().then_some(walker).flatten(),
                body.is_minded(),
            ) {
                (Some(walker_pose), _) => body.follow_walker(walker_pose, dt, Some(floor)),
                (None, true) => body.animate(t),
                (None, false) => body.command(commanded_speed),
            }
            animating |= body.is_animating();
        }
        animating
    }

    /// The DYNAMIC partition: every living entity's leaf triangles, TRANSFORMED
    /// by its current model delta into world space, PLUS every embodied body's
    /// skinned triangles (Rite V) — the exact geometry the traced BVH splices in
    /// this tick (albedo/emission carried per triangle, same split as
    /// [`RenderScene::leaf_triangles`]). Empty with no behaviors and no bodies.
    pub fn dynamic_leaf_triangles(&self) -> Vec<LeafTriangle> {
        let mut out = self.dynamics.leaf_triangles();
        for body in &self.bodies {
            out.extend_from_slice(&body.world_tris);
        }
        out
    }

    /// [`RenderScene::dynamic_leaf_triangles`] plus the OWN-EYE CULL (RITE V
    /// FINAL WELD — nari's first-person mesh must not render INSIDE the camera
    /// that IS her own welded body, the Architect-blinding bug 31bae8b parked):
    /// a walker-ATTACHED body's triangles are OMITTED when `eye` sits within
    /// `epsilon` metres of the last walker pose `command_bodies_walked` was fed
    /// (xz+y identity — the same pose the body tracks), UNLESS `force_draw`
    /// overrides it. Any OTHER eye — a moving `/scry?pos=...`, a diorama camera,
    /// no walker pose ever fed — still collects her, unchanged from
    /// `dynamic_leaf_triangles`. Non-attached bodies (the minded cat) are never
    /// culled by this rule; `force_draw` is a plain parameter (the call site
    /// resolves it from `GAIA_NATIVE_DRAW_OWN_BODY`, default off — this fn never
    /// reads the environment itself).
    pub fn dynamic_leaf_triangles_for_eye(
        &self,
        eye: Vec3,
        epsilon: f32,
        force_draw: bool,
    ) -> Vec<LeafTriangle> {
        let mut out = self.dynamics.leaf_triangles();
        for body in &self.bodies {
            let is_own_eye = !force_draw
                && body.follows_walker()
                && self
                    .last_walker_eye
                    .is_some_and(|walker_eye| eye.distance(walker_eye) <= epsilon);
            if is_own_eye {
                continue;
            }
            out.extend_from_slice(&body.world_tris);
        }
        out
    }

    /// The realm's physics seam, if any body was declared (`None` otherwise —
    /// the zero-physics realm). Exposes the solver bindings, poses and state
    /// hash for verification.
    pub fn physics(&self) -> Option<&Physics> {
        self.dynamics.physics.as_ref()
    }

    /// Mutable access to the physics seam — for direct solver-only timing
    /// (measuring `Solver::step` wall-clock without the kami/BVH overhead
    /// `tick`/`tick_with_ops` also pay). Not part of the normal Flow of Data;
    /// callers that step the solver directly here are responsible for also
    /// writing the pose back if they need the render/ECS to see it move.
    pub fn physics_mut(&mut self) -> Option<&mut Physics> {
        self.dynamics.physics.as_mut()
    }

    /// The current world position of a declared body (its solver centroid),
    /// or `None` if the id names no body.
    pub fn body_position(&self, gaia_id: &str) -> Option<[f64; 3]> {
        let physics = self.dynamics.physics.as_ref()?;
        let binding = physics.bindings().iter().find(|b| b.gaia_id == gaia_id)?;
        Some(physics.pose(binding).position)
    }

    /// Select and expand the view-dependent cluster cut into draw vertices — the
    /// ONE geometry path. For each chain, every cluster is drawn where its
    /// group's projected `parent_error > τ ≥ error` (crack-free by the shared
    /// LOD metric); leaves carry error 0, roots carry parent_error ∞, so exactly
    /// one cut covers the surface. Colour/emissive come from the batch.
    pub fn select_vertices(&self, camera: &Camera, viewport_height: u32) -> Vec<Vertex> {
        let half_fov = (camera.fov_y_radians * 0.5).tan().max(1e-6);
        let projection_scale = viewport_height.max(1) as f32 / (2.0 * half_fov);
        let mut out = Vec::<Vertex>::new();
        for chain in &self.chains {
            select_chain(
                chain,
                camera,
                projection_scale,
                self.error_threshold,
                &mut out,
            );
        }
        out
    }

    /// Every leaf triangle's corner positions, world-space, view-independent —
    /// the EXACT geometry (error 0, the world itself). The Embodiment's floor:
    /// a body stands on the world, never on a camera's coarse cut.
    pub fn leaf_positions(&self) -> Vec<[f32; 3]> {
        leaf_positions_of(&self.chains)
    }

    /// Every STATIC leaf triangle carrying its material, world-space — the EXACT
    /// geometry the traced integrator's STATIC BVH is built over (view-
    /// independent, error 0, built once and cached; the living layer's triangles
    /// are the separate [`RenderScene::dynamic_leaf_triangles`] partition).
    /// Extends `leaf_positions` with per-triangle albedo/emission from
    /// the material batch: a pure emitter gets albedo 0 + emission colour×scale
    /// (matching the Pleroma), a non-emitter gets albedo colour + emission 0.
    pub fn leaf_triangles(&self) -> Vec<LeafTriangle> {
        let mut out = Vec::new();
        for chain in &self.chains {
            let emitter = chain.emissive > 0.5;
            let albedo = if emitter { [0.0; 3] } else { chain.color };
            let emission = if emitter {
                [
                    chain.color[0] * self.emission_intensity,
                    chain.color[1] * self.emission_intensity,
                    chain.color[2] * self.emission_intensity,
                ]
            } else {
                [0.0; 3]
            };
            if let Some(leaf_ids) = chain.dag.levels.first() {
                for &id in leaf_ids {
                    let cluster = chain.dag.cluster(id);
                    for triangle in cluster.indices.chunks_exact(3) {
                        out.push(LeafTriangle {
                            positions: [
                                cluster.vertices[triangle[0] as usize].position,
                                cluster.vertices[triangle[1] as usize].position,
                                cluster.vertices[triangle[2] as usize].position,
                            ],
                            albedo,
                            emission,
                            metallic: chain.metallic,
                            roughness: chain.roughness,
                        });
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy)]
struct PrimitiveVertex {
    position: Vec3,
    normal: Vec3,
}

fn first_component<T: serde::de::DeserializeOwned>(
    world: &EcsWorld,
    name: &str,
) -> Result<Option<T>, String> {
    let Some(component) = world.component_id(name) else {
        return Ok(None);
    };
    let mut entities = world.query(&QuerySpec {
        all: vec![component],
        ..Default::default()
    });
    entities.sort_by(|a, b| world.gaia_id_for(*a).cmp(&world.gaia_id_for(*b)));
    entities
        .first()
        .map(|entity| {
            serde_json::from_value(world.get_component(*entity, component)?)
                .map_err(|error| format!("component {name:?}: {error}"))
        })
        .transpose()
}

fn parts_of(mesh: Mesh) -> Result<Vec<MeshPart>, String> {
    if let Some(parts) = mesh.parts {
        return Ok(parts);
    }
    if mesh.extra.contains_key("shape") {
        return serde_json::from_value(serde_json::Value::Object(mesh.extra))
            .map(|part| vec![part])
            .map_err(|error| error.to_string());
    }
    Ok(Vec::new())
}

/// Read every VESSEL `body`-sigil entity into a skinned world-space
/// [`BodyInstance`] (Rite V). GENERIC: the compose reads only the
/// `body`/`transform` sigils and resolves the named vessel preset — the engine
/// never special-cases a creature. The body is composed at SAMA's idle pose
/// (via [`vessel::Body`]) and given a live [`Locomotion`] machine for V1; each
/// vertex is coloured by its region palette, a triangle's albedo the mean of
/// its three vertices' linear colours. `body.preset` names the preset; an
/// unknown preset is an authoring error (surfaced, never silently invented).
///
/// TWO BODY KINDS share the `body` sigil (the P3 merge): a VESSEL body names a
/// `preset` (skinned here); a PHYSICS body names a `shape` and is owned by the
/// Elements' rigid solver (see [`crate::physics`]) — that path collects it into
/// `body_declarations` during the mesh walk. This skinned pass is vessels ONLY,
/// so it SKIPS physics bodies; both dynamics paths then feed the traced BVH
/// splice side by side.
///
/// CONTACT (Rite V·V1, Guardian finding 1 — no hover): the world y is DERIVED,
/// not authored. The compose rotates/scales the idle mesh (translation excluded)
/// and finds the lowest CONTACT vertex (a `.foot`-bone vertex; the whole mesh's
/// lowest if a morphology has no foot bone); the realm `floor` height under the
/// body's `(x, z)` is read from the SAME static geometry the walker stands on,
/// and the y is set so that contact vertex rests exactly on the floor. Nothing
/// is nudged by eye.
fn body_instances(world: &EcsWorld, floor: &Ground) -> Result<Vec<BodyInstance>, String> {
    let Some(body_id) = world.component_id("body") else {
        return Ok(Vec::new());
    };
    let behavior_id = world.component_id("behavior");
    let Some(transform_id) = world.component_id("transform") else {
        return Ok(Vec::new());
    };
    let mut entities = world.query(&QuerySpec {
        all: vec![body_id, transform_id],
        ..Default::default()
    });
    entities.sort_by(|a, b| world.gaia_id_for(*a).cmp(&world.gaia_id_for(*b)));

    let mut out = Vec::new();
    for entity in entities {
        let id = world.gaia_id_for(entity).unwrap_or("<unbound>").to_string();
        let body_value = world.get_component(entity, body_id)?;
        // A PHYSICS body (names a `shape`, no `preset`) belongs to the rigid
        // solver, not the skinned path — skip it. A body with neither `preset`
        // nor `shape` is still an authoring error (surfaced below).
        if body_value.get("preset").is_none() && body_value.get("shape").is_some() {
            continue;
        }
        let preset_name = body_value
            .get("preset")
            .and_then(|value| value.as_str())
            .ok_or_else(|| format!("entity {id:?} body: missing string `preset`"))?
            .to_string();
        let preset = vessel::Preset::by_name(&preset_name)
            .ok_or_else(|| format!("entity {id:?} body: unknown preset {preset_name:?}"))?;

        let transform: Transform =
            serde_json::from_value(world.get_component(entity, transform_id)?)
                .map_err(|error| format!("entity {id:?} transform: {error}"))?;
        let position = vec3(transform.position.as_ref()).unwrap_or(Vec3::ZERO);
        let rotation = vec3(transform.rotation.as_ref()).unwrap_or(Vec3::ZERO);
        let body_scale = scale(transform.scale.as_ref());

        // Compose: skin + colour + idle pose (sama). The mesh is skeleton-local
        // (pelvis at origin); the entity transform places it in the world.
        let composed = vessel::Body::from_preset(&preset);
        let idle = composed.idle_pose.clone();
        let mesh = composed.vessel.posed(&composed.skeleton, &idle);
        let albedo = composed.vertex_albedo();

        // Derive the grounded y: the rotation+scale of the idle mesh, no
        // translation, then the lowest CONTACT vertex — the foot that must
        // touch the floor. The realm floor under (x,z) is the static geometry
        // the walker itself stands on; if none is found (body over the void)
        // the authored y is kept verbatim.
        let orient = transform_matrix(Vec3::ZERO, rotation, body_scale);
        let contact_local = lowest_contact_y(&composed, &mesh, orient);
        let ground_y = floor.height_at(position.x, position.z, f32::INFINITY);
        let grounded_y = ground_y.map(|g| g - contact_local).unwrap_or(position.y);
        let model = transform_matrix(
            Vec3::new(position.x, grounded_y, position.z),
            rotation,
            body_scale,
        );

        let world_tris = skin_body(&composed, &idle, model, &albedo);

        // RITE V · V2 — attach the behavior spirit if this body carries a
        // `behavior` component tagged `{kind:"cat", ...}`. The kind gate matters:
        // CatMind's fields all default, so it would silently absorb ANY behavior
        // JSON — only a `cat` kind may drive a cat. Other behaviors (or none)
        // leave the body mindless (walker-driven).
        let mind = behavior_id
            .and_then(|bid| world.get_component(entity, bid).ok())
            .filter(|raw| raw.get("kind").and_then(|k| k.as_str()) == Some("cat"))
            .and_then(|raw| serde_json::from_value::<CatMind>(raw).ok());

        // RITE V FINAL WELD — the ATTACHMENT sigil `body.follows` (plain English).
        // `"walker"` binds this body to THE WALKER; the engine reads the parameter
        // and never special-cases a name. Absent = unattached (walker-broadcast).
        let follows = body_value
            .get("follows")
            .and_then(|value| value.as_str())
            .map(str::to_string);

        out.push(BodyInstance {
            gaia_id: id,
            preset: preset_name,
            world_tris,
            body: composed,
            albedo,
            model,
            locomotion: Locomotion::new(LocomotionParams::default()),
            pose: idle,
            commanded_speed: 0.0,
            mind,
            ground_y: grounded_y,
            base_scale: body_scale,
            follows,
        });
    }
    Ok(out)
}

/// Two representative ticks of a `params` walk cycle, DERIVED from the gait
/// (never eye-picked): the CONTACT tick (the swing foot at its LOWEST — nearest
/// a plant, both feet down) and the PASSING tick (the swing foot at its HIGHEST
/// — mid-swing, one foot lifted). The metric is the highest `.foot` vertex of
/// the posed mesh over the cycle; contact = argmin, passing = argmax. The V1
/// ordeal and the proof forge both read the SAME two poses through this.
pub fn contact_passing_ticks(body: &vessel::Body, params: &GaitParams) -> (u64, u64) {
    let cycle = (1.0 / (params.cadence * params.dt)).round().max(1.0) as u64;
    let foot: Vec<usize> = (0..body.vessel.mesh.positions.len())
        .filter(|&vi| {
            body.vessel.weights.per_vertex[vi]
                .first()
                .map(|(bone, _)| body.skeleton.bones[*bone].name.ends_with(".foot"))
                .unwrap_or(false)
        })
        .collect();
    let lift = |tick: u64| -> f32 {
        let pose = gait_pose(&body.skeleton, params, tick);
        let mesh = body.vessel.posed(&body.skeleton, &pose);
        foot.iter()
            .map(|&vi| mesh.positions[vi].y)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut contact = 0u64;
    let mut passing = 0u64;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for tick in 0..cycle {
        let l = lift(tick);
        if l < lo {
            lo = l;
            contact = tick;
        }
        if l > hi {
            hi = l;
            passing = tick;
        }
    }
    (contact, passing)
}

/// The lowest CONTACT-vertex y of an idle mesh under an orientation (rotation +
/// scale, NO translation) — the foot point that must rest on the floor. A
/// vertex is a contact vertex when its dominant (max-weight) bone is a `.foot`;
/// a morphology with no foot bone falls back to the whole mesh's lowest vertex.
/// Deterministic and derived from the bind pose — the grounding never guesses.
fn lowest_contact_y(body: &vessel::Body, mesh: &vessel::Mesh, orient: Mat4) -> f32 {
    let skeleton = &body.skeleton;
    let mut foot_min = f32::INFINITY;
    let mut any_min = f32::INFINITY;
    for (vi, position) in mesh.positions.iter().enumerate() {
        let y = orient.transform_point3(*position).y;
        any_min = any_min.min(y);
        let dominant = body.vessel.weights.per_vertex[vi]
            .first()
            .map(|(bone, _)| *bone);
        if let Some(bone) = dominant
            && skeleton.bones[bone].name.ends_with(".foot")
        {
            foot_min = foot_min.min(y);
        }
    }
    if foot_min.is_finite() {
        foot_min
    } else {
        any_min
    }
}

/// A real emissive light source in the realm — one glowing mesh part: its
/// world-space centre and its emissive colour (linear rgb). The medium binds
/// its in-scatter light to one of these REAL sources (A2 true binding), never
/// an invented light. Read straight from the ECS entities, before the render
/// scene consumes the world.
#[derive(Clone, Debug, PartialEq)]
pub struct EmissiveSource {
    /// The owning entity's gaia id — the selection handle.
    pub id: String,
    /// World-space centre of the glowing part.
    pub position: [f32; 3],
    /// Emissive colour, linear rgb.
    pub color: [f32; 3],
    /// Effective emitter radius (world units): a sphere's radius, or the
    /// projected-area equivalent of a box's largest face (`√(A_max/π)`). Lets a
    /// caller derive a point light's radiant intensity as radiance × πr² — the
    /// emitter's real emitting area — never a plucked scale. A long thin strip
    /// gets its true (small) face area, not a fat bounding radius.
    pub radius: f32,
}

/// Every emissive mesh part in the realm as an [`EmissiveSource`] (entity id,
/// world-space centre, linear-rgb colour), sorted by gaia id (deterministic).
/// The medium's A2 binding selects one of these — the stall's lantern glow —
/// instead of inventing a light. Reads the SAME transform/mesh the render scene
/// tessellates, so the position is the exact authored world location.
pub fn emissive_sources(world: &EcsWorld) -> Result<Vec<EmissiveSource>, String> {
    let Some((transform_id, mesh_id)) = world
        .component_id("transform")
        .zip(world.component_id("mesh"))
    else {
        return Ok(Vec::new());
    };
    let mut entities = world.query(&QuerySpec {
        all: vec![transform_id, mesh_id],
        ..Default::default()
    });
    entities.sort_by(|a, b| world.gaia_id_for(*a).cmp(&world.gaia_id_for(*b)));
    let mut out = Vec::new();
    for entity in entities {
        let id = world.gaia_id_for(entity).unwrap_or("<unbound>").to_string();
        let transform: Transform =
            serde_json::from_value(world.get_component(entity, transform_id)?)
                .map_err(|error| format!("entity {id:?} transform: {error}"))?;
        let mesh: Mesh = serde_json::from_value(world.get_component(entity, mesh_id)?)
            .map_err(|error| format!("entity {id:?} mesh: {error}"))?;
        let parts = parts_of(mesh).map_err(|error| format!("entity {id:?} mesh: {error}"))?;
        let entity_model = transform_matrix(
            vec3(transform.position.as_ref()).unwrap_or(Vec3::ZERO),
            vec3(transform.rotation.as_ref()).unwrap_or(Vec3::ZERO),
            scale(transform.scale.as_ref()),
        );
        for part in &parts {
            let Some(hex) = part.emissive.as_deref() else {
                continue;
            };
            let color = linear_rgb(hex)?;
            let part_center = vec3(part.position.as_ref()).unwrap_or(Vec3::ZERO);
            let world_center = entity_model.transform_point3(part_center);
            // Effective radius: a sphere's radius, else the projected-area
            // equivalent of the box's largest face (√(A_max/π)), so a long thin
            // neon strip contributes its true face area, not a fat radius.
            let radius = match number(part.radius.as_ref()) {
                Some(r) => r,
                None => vec3(part.size.as_ref())
                    .map(|s| {
                        let a_max = (s.x * s.y).max((s.x * s.z).max(s.y * s.z));
                        (a_max / std::f32::consts::PI).sqrt()
                    })
                    .unwrap_or(0.5),
            };
            out.push(EmissiveSource {
                id: id.clone(),
                position: world_center.to_array(),
                color,
                radius,
            });
        }
    }
    Ok(out)
}

/// The world-space y of the topmost flat slab of an entity — the surface a prop
/// (or a steam plume) rests ON. A "flat slab" is a box part whose vertical
/// extent is its smallest dimension (a serving counter / table top / roof, not
/// an upright post). Returns the highest such top face in world space, DERIVED
/// from the realm geometry — the plume's y-min is grounded here, never plucked.
/// `None` when the entity has no flat box part (or does not exist).
pub fn top_flat_surface_y(world: &EcsWorld, gaia_id: &str) -> Result<Option<f32>, String> {
    let (Some(transform_id), Some(mesh_id)) =
        (world.component_id("transform"), world.component_id("mesh"))
    else {
        return Ok(None);
    };
    let Some(entity) = world.entity_for_gaia(gaia_id) else {
        return Ok(None);
    };
    let transform: Transform = serde_json::from_value(world.get_component(entity, transform_id)?)
        .map_err(|error| format!("entity {gaia_id:?} transform: {error}"))?;
    let mesh: Mesh = serde_json::from_value(world.get_component(entity, mesh_id)?)
        .map_err(|error| format!("entity {gaia_id:?} mesh: {error}"))?;
    let parts = parts_of(mesh).map_err(|error| format!("entity {gaia_id:?} mesh: {error}"))?;
    let entity_y = vec3(transform.position.as_ref()).unwrap_or(Vec3::ZERO).y;
    let mut best: Option<f32> = None;
    for part in &parts {
        if part.shape.as_deref().unwrap_or("box") != "box" {
            continue;
        }
        let Some(size) = vec3(part.size.as_ref()) else {
            continue;
        };
        // A flat slab: the vertical extent is the smallest of the three.
        if size.y > size.x.min(size.z) {
            continue;
        }
        let part_y = vec3(part.position.as_ref()).unwrap_or(Vec3::ZERO).y;
        let top = entity_y + part_y + size.y * 0.5;
        best = Some(best.map_or(top, |b| b.max(top)));
    }
    Ok(best)
}

/// The world-space CENTRE of the entity's highest flat serving surface, as
/// `[center_x, top_y, center_z]` — the same part [`top_flat_surface_y`] selects
/// (max top), but returning its horizontal centre too so a caller can GROUND
/// something on the surface (its `top_y`) at the surface's real footprint
/// (`center_x`/`center_z`) instead of inventing horizontal coordinates.
pub fn top_flat_surface_center(
    world: &EcsWorld,
    gaia_id: &str,
) -> Result<Option<[f32; 3]>, String> {
    let (Some(transform_id), Some(mesh_id)) =
        (world.component_id("transform"), world.component_id("mesh"))
    else {
        return Ok(None);
    };
    let Some(entity) = world.entity_for_gaia(gaia_id) else {
        return Ok(None);
    };
    let transform: Transform = serde_json::from_value(world.get_component(entity, transform_id)?)
        .map_err(|error| format!("entity {gaia_id:?} transform: {error}"))?;
    let mesh: Mesh = serde_json::from_value(world.get_component(entity, mesh_id)?)
        .map_err(|error| format!("entity {gaia_id:?} mesh: {error}"))?;
    let parts = parts_of(mesh).map_err(|error| format!("entity {gaia_id:?} mesh: {error}"))?;
    let entity_pos = vec3(transform.position.as_ref()).unwrap_or(Vec3::ZERO);
    let mut best: Option<(f32, [f32; 3])> = None;
    for part in &parts {
        if part.shape.as_deref().unwrap_or("box") != "box" {
            continue;
        }
        let Some(size) = vec3(part.size.as_ref()) else {
            continue;
        };
        if size.y > size.x.min(size.z) {
            continue;
        }
        let part_pos = vec3(part.position.as_ref()).unwrap_or(Vec3::ZERO);
        let top = entity_pos.y + part_pos.y + size.y * 0.5;
        let center = [entity_pos.x + part_pos.x, top, entity_pos.z + part_pos.z];
        if best.is_none_or(|(b, _)| top > b) {
            best = Some((top, center));
        }
    }
    Ok(best.map(|(_, center)| center))
}

/// Project a cluster's LOD error through its group's SHARED bounds sphere to a
/// screen-space error (~pixels). Error 0 (leaves) stays 0. Distance metric
/// (Rite III); hardware visibility lands later.
fn project_error(error: f32, bounds: &Bounds, camera: &Camera, projection_scale: f32) -> f32 {
    if error <= 0.0 {
        return 0.0;
    }
    let center = Vec3::from_array(bounds.center);
    let distance = ((center - camera.eye).length() - bounds.radius).max(camera.near);
    error * projection_scale / distance
}

/// Expand one chain's view-dependent cut into `out`. `error` side reads the
/// PRODUCING group's sphere (`cluster.group`; None = leaf, error 0); the
/// `parent_error` side reads the CONSUMING group's sphere (`cluster.parent_group`;
/// None = terminal/root, ∞). Draw where `parent_sse > τ ≥ self_sse`.
fn select_chain(
    chain: &MaterialChain,
    camera: &Camera,
    projection_scale: f32,
    tau: f32,
    out: &mut Vec<Vertex>,
) {
    let dag = &chain.dag;
    for cluster in &dag.clusters {
        let self_sse = match cluster.group {
            Some(group) => project_error(
                cluster.error,
                &dag.group(group).bounds,
                camera,
                projection_scale,
            ),
            None => 0.0,
        };
        let parent_sse = match cluster.parent_group {
            Some(group) => project_error(
                cluster.parent_error,
                &dag.group(group).bounds,
                camera,
                projection_scale,
            ),
            None => f32::INFINITY,
        };
        if parent_sse > tau && tau >= self_sse {
            emit_cluster(cluster, chain.color, chain.emissive, out);
        }
    }
}

fn emit_cluster(cluster: &Cluster, color: [f32; 3], emissive: f32, out: &mut Vec<Vertex>) {
    out.reserve(cluster.indices.len());
    for &index in &cluster.indices {
        let vertex = &cluster.vertices[index as usize];
        out.push(Vertex {
            position: vertex.position,
            normal: vertex.normal,
            color,
            emissive,
        });
    }
}

/// Every leaf triangle's corner positions, world-space, from a set of sealed
/// Great Chains — the POST-transmute EXACT geometry (error 0, the world itself).
/// The SINGLE floor source (F1): both the walker's ground
/// ([`RenderScene::leaf_positions`] → `main.rs`) and the RITE V bodies' ground
/// (`from_ecs` weld) construct their [`Ground`] from THIS function over the SAME
/// chains, so the two floors are byte-identical by construction — never a
/// pre/post-transmute fork.
fn leaf_positions_of(chains: &[MaterialChain]) -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for chain in chains {
        if let Some(leaf_ids) = chain.dag.levels.first() {
            for &id in leaf_ids {
                let cluster = chain.dag.cluster(id);
                for &index in &cluster.indices {
                    out.push(cluster.vertices[index as usize].position);
                }
            }
        }
    }
    out
}

/// Seal material buckets into transmuted Great Chains. `transmute` is
/// deterministic (BTree ordering + canonical welds), so two builds of one input
/// produce byte-identical chains. Shared by the static pool and every dynamic
/// entity's own chains.
fn seal_buckets(buckets: BTreeMap<MatKey, MatBucket>) -> Result<Vec<MaterialChain>, String> {
    let chain_params = TransmuteParams::default();
    let mut chains = Vec::<MaterialChain>::with_capacity(buckets.len());
    for bucket in buckets.into_values() {
        if bucket.vertices.is_empty() {
            continue;
        }
        let indices: Vec<u32> = (0..bucket.vertices.len() as u32).collect();
        let mesh = ChainMesh::new(bucket.vertices, indices);
        let dag = transmute_default(&mesh, &chain_params)
            .map_err(|error| format!("transmute material chain: {error}"))?;
        chains.push(MaterialChain {
            dag,
            color: bucket.color,
            emissive: bucket.emissive,
            metallic: bucket.metallic,
            roughness: bucket.roughness,
        });
    }
    Ok(chains)
}

/// A body's fitted rigid rotation (world-space column axes) as the transform's
/// `EulerRot::XYZ` triple — the same convention [`transform_matrix`] rebuilds,
/// so the write-back round-trips cleanly.
fn euler_xyz(pose: &BodyPose) -> [f32; 3] {
    let c = pose.rotation_columns;
    let matrix = Mat3::from_cols(
        Vec3::new(c[0][0] as f32, c[0][1] as f32, c[0][2] as f32),
        Vec3::new(c[1][0] as f32, c[1][1] as f32, c[1][2] as f32),
        Vec3::new(c[2][0] as f32, c[2][1] as f32, c[2][2] as f32),
    );
    let (x, y, z) = Quat::from_mat3(&matrix).to_euler(EulerRot::XYZ);
    [x, y, z]
}

/// One chain's LEAF triangles carrying their material (finest LOD, view-
/// independent) — the same albedo/emission split as [`RenderScene::leaf_triangles`],
/// applied to a single chain. Used to bake a dynamic entity's bind-pose geometry.
fn chain_leaf_triangles(chain: &MaterialChain, emission_intensity: f32) -> Vec<LeafTriangle> {
    let emitter = chain.emissive > 0.5;
    let albedo = if emitter { [0.0; 3] } else { chain.color };
    let emission = if emitter {
        [
            chain.color[0] * emission_intensity,
            chain.color[1] * emission_intensity,
            chain.color[2] * emission_intensity,
        ]
    } else {
        [0.0; 3]
    };
    let mut out = Vec::new();
    if let Some(leaf_ids) = chain.dag.levels.first() {
        for &id in leaf_ids {
            let cluster = chain.dag.cluster(id);
            for triangle in cluster.indices.chunks_exact(3) {
                out.push(LeafTriangle {
                    positions: [
                        cluster.vertices[triangle[0] as usize].position,
                        cluster.vertices[triangle[1] as usize].position,
                        cluster.vertices[triangle[2] as usize].position,
                    ],
                    albedo,
                    emission,
                    metallic: chain.metallic,
                    roughness: chain.roughness,
                });
            }
        }
    }
    out
}

/// One dynamic entity of the living layer: its bind-baked leaf triangles (world-
/// space, at the AUTHORED rest pose) plus the live model delta that animates
/// them. `model` = `M(animated) · M(bind)⁻¹`; applying it to the bind-baked
/// world-space corners yields the current world-space triangles the BVH splices.
pub struct DynamicEntity {
    pub gaia_id: String,
    /// The entity's material bands: (linear colour, emissive-flag), one per
    /// chain — the exact keys `RenderScene::from_ecs` bucketed by (no lossy
    /// reconstruction from baked emission).
    pub materials: Vec<([f32; 3], bool)>,
    /// Leaf triangles at the authored BIND world pose (material carried).
    pub bind_tris: Vec<LeafTriangle>,
    /// The bind transform matrix (world-space) the triangles were baked at.
    pub bind_model: Mat4,
    /// Current animated delta transform (identity until the first tick).
    pub model: Mat4,
}

impl DynamicEntity {
    /// This entity's leaf triangles transformed into current world space by the
    /// live `model` delta — the geometry the traced BVH sees this tick.
    pub fn world_triangles(&self) -> Vec<LeafTriangle> {
        self.bind_tris
            .iter()
            .map(|t| {
                let mut out = *t;
                for (k, p) in t.positions.iter().enumerate() {
                    out.positions[k] = self.model.transform_point3(Vec3::from_array(*p)).to_array();
                }
                out
            })
            .collect()
    }
}

/// The living layer of a scene: the ECS the world clock ticks, the dynamic
/// entities' bind geometry + live models, and the fixed-dt clock. The tick is
/// closed-form on the tick INDEX (entropy), never wall time — N ticks reproduce
/// byte-identical model transforms across runs (the determinism ordeal).
pub struct Dynamics {
    /// The live ECS: the tick reads `behavior` here and writes animated
    /// `transform`s back (so senses/pose read the moving world).
    world: EcsWorld,
    /// Registered `transform`/`behavior` handles; `None` until `install_world`.
    reg: Option<Registry>,
    /// Rest poses keyed by gaia id — the fixed origin each kind animates around,
    /// so re-reading the transform each tick never compounds.
    binds: BTreeMap<String, BindPose>,
    entities: Vec<DynamicEntity>,
    emission_intensity: f32,
    /// The Elements' rigid solver bound to the realm's declared bodies; `None`
    /// when no `body` is declared (the physics path is then wholly inert).
    physics: Option<Physics>,
    /// VI-2 — `(fragment_gaia_id, particle_indices)` for every fragment born
    /// so far, so its centroid can keep being read back every subsequent
    /// tick (a broken bonded body's `Physics` binding stops reporting once
    /// broken — this is where "shards settle" lives after birth).
    fragment_particles: Vec<(String, Vec<usize>)>,
    seed: u64,
    dt: f64,
    clock: u64,
}

impl Dynamics {
    fn new(emission_intensity: f32) -> Self {
        Self {
            world: EcsWorld::default(),
            reg: None,
            binds: BTreeMap::new(),
            entities: Vec::new(),
            emission_intensity,
            physics: None,
            fragment_particles: Vec::new(),
            seed: 0,
            dt: 1.0 / 60.0,
            clock: 0,
        }
    }

    fn push(
        &mut self,
        id: &str,
        chains: Vec<MaterialChain>,
        bind: BindPose,
        bind_model: Mat4,
        emission_intensity: f32,
    ) {
        let mut bind_tris = Vec::new();
        let mut materials = Vec::new();
        for chain in &chains {
            bind_tris.extend(chain_leaf_triangles(chain, emission_intensity));
            materials.push((chain.color, chain.emissive > 0.5));
        }
        self.binds.insert(id.to_string(), bind);
        self.entities.push(DynamicEntity {
            gaia_id: id.to_string(),
            materials,
            bind_tris,
            bind_model,
            model: Mat4::IDENTITY,
        });
    }

    /// Take ownership of the live ECS and lock in the tick's dt. Registration is
    /// idempotent — the loader already registered `transform`/`behavior`, so the
    /// existing ids are reused.
    fn install_world(&mut self, mut world: EcsWorld, parameters: &SceneParameters) {
        self.reg = Some(Registry::register(&mut world));
        self.world = world;
        self.dt = parameters.tick_dt;
        self.clock = 0;
    }

    /// Bind the Elements' rigid solver to the realm's declared bodies and the
    /// static collider. A no-op (leaves `physics = None`) when nothing declared
    /// a `body` — the zero-physics realm then ticks byte-identically to before.
    fn install_physics(
        &mut self,
        declarations: Vec<(String, Body, [f64; 3])>,
        collider_triangles: Vec<Triangle>,
        dt: f64,
    ) {
        self.physics = Physics::install(declarations, collider_triangles, dt, self.seed);
    }

    /// One world tick (Flow of Data): KAMI reads the ECS → emits transform ops →
    /// they apply to the ECS → each entity's model is re-derived from its now-
    /// animated transform. Increments the clock. Deterministic in the count.
    pub fn tick(&mut self) {
        self.tick_with_ops(&[]);
    }

    /// [`Dynamics::tick`], plus a batch of incantation ops folded into the
    /// physics block BEFORE `physics.step()` this tick — currently only
    /// [`Op::Impulse`] does anything here ("the op is the hand": the caller
    /// names an entity + a velocity delta, scrying-glass resolves it to the
    /// body's solver rigid index). Every other op variant is ignored (this is
    /// a physics-only seam, not the general op router).
    pub fn tick_with_ops(&mut self, incoming: &[Op]) {
        let Some(reg) = self.reg else {
            return;
        };
        if self.entities.is_empty() {
            return;
        }
        let ctx = TickContext {
            seed: self.seed,
            entropy: self.clock,
            dt: self.dt,
        };
        let ops = kami::tick_decorative(&self.world, reg, &self.binds, &ctx);
        for op in &ops {
            let Op::Set(set) = op else {
                continue;
            };
            if let Some(entity) = self.world.entity_for_gaia(&set.id) {
                let _ = self
                    .world
                    .set_component(entity, reg.transform, set.value.clone());
            }
        }
        // PHYSICS (the Elements bound into the realm): fold any incoming
        // impulses in first, advance every declared body one tick, then
        // write its pose (centroid + fitted rotation) back to the ECS
        // transform — the same Flow of Data KAMI uses, so the body's
        // triangles ride the dynamic BVH splice and the traced light sees it
        // move. Read all poses first (shared borrow), then write (world borrow).
        let mut body_poses: Vec<(String, [f64; 3], [f32; 3])> = match self.physics.as_mut() {
            Some(physics) => {
                for op in incoming {
                    if let Op::Impulse(impulse) = op {
                        physics.apply_impulse(&impulse.id, impulse.delta_velocity);
                    }
                }
                physics.step();
                physics
                    .bindings()
                    .iter()
                    .map(|binding| {
                        let pose = physics.pose(binding);
                        (binding.gaia_id.clone(), pose.position, euler_xyz(&pose))
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        // VI-2 — SOMETHING BREAKS: bonded bodies poll separately (no shape-
        // matched RigidBody pose to read). Still-whole bonded bodies ride
        // the same transform write-back as a rigid body (translation only,
        // rotation identity — see `Physics::poll_bonded`'s doc). A body that
        // broke THIS tick births its fragments as real ECS vessels — the
        // SAME wave: no separate tick, no lag — with a Great Chain re-mesh
        // (`fracture::fragment_mesh` -> `transmute_default`, the one
        // geometry path) spliced into the dynamic BVH by pushing a
        // `DynamicEntity` per fragment (picked up by `dynamic_leaf_
        // triangles` this exact tick, same as any other dynamic entity).
        if let Some(physics) = self.physics.as_mut() {
            let (still_whole, newly_broken) = physics.poll_bonded();
            for (id, position) in still_whole {
                body_poses.push((id, position, [0.0, 0.0, 0.0]));
            }
            for (parent_id, fragments, cube_size) in newly_broken {
                // Fragments inherit the parent's authored material (colour +
                // emissive flag) — fragments are made of the same stuff, no
                // invented colour.
                let inherited = self
                    .entities
                    .iter()
                    .find(|de| de.gaia_id == parent_id)
                    .and_then(|de| de.materials.first().copied())
                    .unwrap_or(([0.6, 0.6, 0.6], false));
                // The whole-body vessel is gone — its geometry is replaced
                // by its fragments (no double-draw of the same matter).
                self.entities.retain(|de| de.gaia_id != parent_id);

                let fragment_ids =
                    fracture::birth_fragment_entities(&mut self.world, &parent_id, &fragments);
                let params = TransmuteParams::default();
                for (frag_id, fragment) in fragment_ids.iter().zip(fragments.iter()) {
                    let mesh = fracture::fragment_mesh(physics.solver(), fragment, cube_size);
                    if mesh.indices.is_empty() {
                        continue;
                    }
                    let Ok(dag) = fracture::fragment_dag(&mesh, &params) else {
                        continue;
                    };
                    let chain = MaterialChain {
                        dag,
                        color: inherited.0,
                        emissive: if inherited.1 { 1.0 } else { 0.0 },
                        metallic: 0.0,
                        roughness: 1.0,
                    };
                    let tris = chain_leaf_triangles(&chain, self.emission_intensity);
                    let c = fragment.centroid;
                    let bind_model =
                        Mat4::from_translation(Vec3::new(c.x as f32, c.y as f32, c.z as f32));
                    self.entities.push(DynamicEntity {
                        gaia_id: frag_id.clone(),
                        materials: vec![inherited],
                        bind_tris: tris,
                        bind_model,
                        model: Mat4::IDENTITY,
                    });
                    self.fragment_particles
                        .push((frag_id.clone(), fragment.particles.clone()));
                }
            }
            // Fragments already born keep settling (translation-only
            // tracking of their own fixed particle set) every subsequent
            // tick — "shards at rest" needs them to keep moving after birth.
            for (id, particles) in &self.fragment_particles {
                let c = physics.group_centroid(particles);
                body_poses.push((id.clone(), c, [0.0, 0.0, 0.0]));
            }
        }
        for (id, position, rotation) in &body_poses {
            if let Some(entity) = self.world.entity_for_gaia(id) {
                let value = json!({
                    "position": [position[0], position[1], position[2]],
                    "rotation": [rotation[0], rotation[1], rotation[2]],
                });
                let _ = self.world.set_component(entity, reg.transform, value);
            }
        }
        for de in &mut self.entities {
            let Some(entity) = self.world.entity_for_gaia(&de.gaia_id) else {
                continue;
            };
            let Ok(value) = self.world.get_component(entity, reg.transform) else {
                continue;
            };
            let Ok(transform) = serde_json::from_value::<Transform>(value) else {
                continue;
            };
            let animated = transform_matrix(
                vec3(transform.position.as_ref()).unwrap_or(Vec3::ZERO),
                vec3(transform.rotation.as_ref()).unwrap_or(Vec3::ZERO),
                scale(transform.scale.as_ref()),
            );
            de.model = animated * de.bind_model.inverse();
        }
        self.clock += 1;
    }

    /// Every dynamic entity's leaf triangles transformed into current world
    /// space — the DYNAMIC partition the traced BVH splices this tick.
    pub fn leaf_triangles(&self) -> Vec<LeafTriangle> {
        let mut out = Vec::new();
        for de in &self.entities {
            out.extend(de.world_triangles());
        }
        out
    }

    /// The per-entity model transforms (column-major mat4), in `entities()`
    /// order. Byte-identical given the tick count (the tick-determinism ordeal
    /// reads exactly these bytes).
    pub fn model_matrices(&self) -> Vec<[f32; 16]> {
        self.entities
            .iter()
            .map(|entity| entity.model.to_cols_array())
            .collect()
    }

    pub fn entities(&self) -> &[DynamicEntity] {
        &self.entities
    }

    /// Consume this `Dynamics` and hand back the live, TICKED `EcsWorld` — the
    /// solver-rested realm, with every physics body's post-tick pose already
    /// written into its `transform` (see [`Dynamics::tick_with_ops`]'s
    /// pose-write-back). Exists for GUARDIAN RULING F6 ("senses read SOLVER
    /// TRUTH — the world as it is, not as authored"): external senses (the
    /// oracle) need the rested ECS to gaze at where the solver actually put a
    /// body, not just the `body_position` scalar or the authored load-pose.
    /// `EcsWorld` is not `Clone`, so this consumes rather than borrows — call it
    /// after the last tick. The smallest surface that lets a sibling crate wrap
    /// the ticked world into its own `oracle::World`.
    pub fn into_world(self) -> EcsWorld {
        self.world
    }
}

fn append_part(
    buckets: &mut BTreeMap<MatKey, MatBucket>,
    part: &MeshPart,
    entity_model: Mat4,
    default_color: [f32; 3],
    default_segments: u32,
    mut collider: Option<&mut Vec<Triangle>>,
) -> Result<(), String> {
    let position = vec3(part.position.as_ref()).unwrap_or(Vec3::ZERO);
    let rotation = vec3(part.rotation.as_ref()).unwrap_or(Vec3::ZERO);
    let model = entity_model * transform_matrix(position, rotation, scale(part.scale.as_ref()));
    let determinant = Mat3::from_mat4(model).determinant();
    if !determinant.is_finite() || determinant.abs() < f32::EPSILON {
        return Err("transform scale must be finite and non-zero".into());
    }
    let normal_matrix = Mat3::from_mat4(model).inverse().transpose();
    let segments = part.radial_segments.unwrap_or(default_segments).max(3);
    let primitive = match part.shape.as_deref().unwrap_or("box") {
        "box" => box_triangles(dimensions(part.size.as_ref(), [1.0, 1.0, 1.0])?),
        "sphere" => sphere_triangles(
            positive(number(part.radius.as_ref()).unwrap_or(0.5), "radius")?,
            segments,
        ),
        "cylinder" => {
            let radius = positive(number(part.radius.as_ref()).unwrap_or(0.5), "radius")?;
            let top = non_negative(
                number(part.radius_top.as_ref()).unwrap_or(radius),
                "radiusTop",
            )?;
            let bottom = non_negative(
                number(part.radius_bottom.as_ref()).unwrap_or(radius),
                "radiusBottom",
            )?;
            let height = positive(number(part.height.as_ref()).unwrap_or(1.0), "height")?;
            frustum_triangles(top, bottom, height, segments, true)
        }
        "cone" => {
            let radius = positive(number(part.radius.as_ref()).unwrap_or(0.5), "radius")?;
            let height = positive(number(part.height.as_ref()).unwrap_or(1.0), "height")?;
            frustum_triangles(0.0, radius, height, segments, true)
        }
        shape => return Err(format!("unsupported W1 primitive {shape:?}")),
    };

    let emissive = part.emissive.is_some();
    let color = match part.emissive.as_deref().or(part.color.as_deref()) {
        Some(color) => linear_rgb(color)?,
        None => default_color,
    };
    let emissive = f32::from(emissive);
    // L2 conductor dials (defaults 0/1 = pure lambertian). `metalness` carries
    // `metallic` too (crystal alias). Clamp to [0,1].
    let metallic = number(part.metalness.as_ref())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32;
    let roughness = number(part.roughness.as_ref())
        .unwrap_or(1.0)
        .clamp(0.0, 1.0) as f32;
    let key: MatKey = (
        [color[0].to_bits(), color[1].to_bits(), color[2].to_bits()],
        emissive.to_bits(),
        metallic.to_bits(),
        roughness.to_bits(),
    );
    let bucket = buckets.entry(key).or_insert_with(|| MatBucket {
        vertices: Vec::new(),
        color,
        emissive,
        metallic,
        roughness,
    });
    for triangle in &primitive {
        let mut positions = [Vec3::ZERO; 3];
        let mut normal_sum = Vec3::ZERO;
        for (k, vertex) in triangle.iter().enumerate() {
            let world_position = model.transform_point3(vertex.position);
            let normal = (normal_matrix * vertex.normal).normalize_or_zero();
            positions[k] = world_position;
            normal_sum += normal;
            bucket.vertices.push(ChainVertex::new(
                world_position.to_array(),
                normal.to_array(),
                [0.0, 0.0],
            ));
        }
        // The collision soup rides the SAME tessellation the light sees, but
        // with the CLEAN per-face outward normal (the mean of the triangle's
        // vertex normals) — never the transmuted/welded render geometry, whose
        // corner-welded normals would mis-push a resting body.
        if let Some(collider) = collider.as_deref_mut() {
            collider.push(Triangle::with_normal(
                glam_to_elements(positions[0]),
                glam_to_elements(positions[1]),
                glam_to_elements(positions[2]),
                glam_to_elements(normal_sum),
            ));
        }
    }
    Ok(())
}

/// Convert a glam world-space vector to the Elements' `f64` vector.
fn glam_to_elements(v: Vec3) -> elements::Vec3 {
    elements::Vec3::new(v.x as f64, v.y as f64, v.z as f64)
}

/// VII-0b's coordinate seam: a terrain tile's world-space offset RELATIVE TO
/// `render_origin`, computed `i64 -> f64 -> subtract -> f32` — never
/// `i64 -> f32` directly (which loses precision past `2^24` meters, VII-0a's
/// module doc). `tile_origin_xz` ([`seed::tile_origin_m`]) and
/// `render_origin` are both `f64`; the subtraction is exact IEEE754
/// arithmetic regardless of either operand's magnitude, so it's only the
/// RESULT — expected small whenever `render_origin` tracks near the tile,
/// the camera-relative-rendering guarantee — that gets cast down to `f32`.
/// A tile's own local mesh vertices ([`seed::tile_mesh`]) are already
/// tile-local `f32`; adding this offset places them without ever routing a
/// planet-scale magnitude through `f32`.
fn terrain_placement_offset(tile_origin_xz: (f64, f64), render_origin: [f64; 3]) -> Vec3 {
    Vec3::new(
        (tile_origin_xz.0 - render_origin[0]) as f32,
        (0.0 - render_origin[1]) as f32,
        (tile_origin_xz.1 - render_origin[2]) as f32,
    )
}

/// Feed one generated terrain tile's mesh ([`seed::tile_mesh`], tile-local
/// `f32` vertices) into a material bucket, placed at `offset`
/// ([`terrain_placement_offset`]) — the same seal path every other static
/// part rides ([`append_part`]), and the same collider-soup convention (a
/// clean per-face outward normal, the mean of the triangle's vertex
/// normals — never the transmuted/welded render geometry's corner-welded
/// normals, so a resting body isn't mis-pushed).
fn append_terrain(
    buckets: &mut BTreeMap<MatKey, MatBucket>,
    mesh: &ChainMesh,
    offset: Vec3,
    color: [f32; 3],
    collider: &mut Vec<Triangle>,
) {
    // Lambertian defaults (no metallic/roughness dial on the sigil yet — a
    // future wave's authoring surface, not VII-0b's scope).
    let emissive = 0.0f32;
    let metallic = 0.0f32;
    let roughness = 1.0f32;
    let key: MatKey = (
        [color[0].to_bits(), color[1].to_bits(), color[2].to_bits()],
        emissive.to_bits(),
        metallic.to_bits(),
        roughness.to_bits(),
    );
    let bucket = buckets.entry(key).or_insert_with(|| MatBucket {
        vertices: Vec::new(),
        color,
        emissive,
        metallic,
        roughness,
    });
    for triangle in mesh.indices.chunks_exact(3) {
        let mut positions = [Vec3::ZERO; 3];
        let mut normal_sum = Vec3::ZERO;
        for (k, &index) in triangle.iter().enumerate() {
            let vertex = &mesh.vertices[index as usize];
            let world_position = Vec3::from_array(vertex.position) + offset;
            let normal = Vec3::from_array(vertex.normal);
            positions[k] = world_position;
            normal_sum += normal;
            bucket.vertices.push(ChainVertex::new(
                world_position.to_array(),
                vertex.normal,
                vertex.uv,
            ));
        }
        collider.push(Triangle::with_normal(
            glam_to_elements(positions[0]),
            glam_to_elements(positions[1]),
            glam_to_elements(positions[2]),
            glam_to_elements(normal_sum),
        ));
    }
}

fn box_triangles(size: Vec3) -> Vec<[PrimitiveVertex; 3]> {
    let half = size * 0.5;
    let faces = [
        (
            Vec3::X * half.x,
            Vec3::NEG_Z * half.z,
            Vec3::Y * half.y,
            Vec3::X,
        ),
        (
            Vec3::NEG_X * half.x,
            Vec3::Z * half.z,
            Vec3::Y * half.y,
            Vec3::NEG_X,
        ),
        (
            Vec3::Y * half.y,
            Vec3::X * half.x,
            Vec3::NEG_Z * half.z,
            Vec3::Y,
        ),
        (
            Vec3::NEG_Y * half.y,
            Vec3::X * half.x,
            Vec3::Z * half.z,
            Vec3::NEG_Y,
        ),
        (
            Vec3::Z * half.z,
            Vec3::X * half.x,
            Vec3::Y * half.y,
            Vec3::Z,
        ),
        (
            Vec3::NEG_Z * half.z,
            Vec3::NEG_X * half.x,
            Vec3::Y * half.y,
            Vec3::NEG_Z,
        ),
    ];
    let mut triangles = Vec::with_capacity(12);
    for (center, u, v, normal) in faces {
        let a = PrimitiveVertex {
            position: center - u - v,
            normal,
        };
        let b = PrimitiveVertex {
            position: center + u - v,
            normal,
        };
        let c = PrimitiveVertex {
            position: center + u + v,
            normal,
        };
        let d = PrimitiveVertex {
            position: center - u + v,
            normal,
        };
        triangles.extend([[a, b, c], [a, c, d]]);
    }
    triangles
}

fn sphere_triangles(radius: f32, segments: u32) -> Vec<[PrimitiveVertex; 3]> {
    let stacks = (segments / 2).max(2);
    let mut triangles = Vec::with_capacity((segments * stacks * 2) as usize);
    let point = |latitude: u32, longitude: u32| {
        let theta = std::f32::consts::PI * latitude as f32 / stacks as f32;
        let phi = std::f32::consts::TAU * longitude as f32 / segments as f32;
        let normal = Vec3::new(
            theta.sin() * phi.sin(),
            theta.cos(),
            theta.sin() * phi.cos(),
        );
        PrimitiveVertex {
            position: normal * radius,
            normal,
        }
    };
    for latitude in 0..stacks {
        for longitude in 0..segments {
            let next = longitude + 1;
            let a = point(latitude, longitude);
            let b = point(latitude + 1, longitude);
            let c = point(latitude + 1, next);
            let d = point(latitude, next);
            if latitude > 0 {
                triangles.push([a, b, c]);
            }
            if latitude + 1 < stacks {
                triangles.push([a, c, d]);
            }
        }
    }
    triangles
}

fn frustum_triangles(
    top_radius: f32,
    bottom_radius: f32,
    height: f32,
    segments: u32,
    capped: bool,
) -> Vec<[PrimitiveVertex; 3]> {
    let mut triangles = Vec::with_capacity((segments * 4) as usize);
    let half = height * 0.5;
    let slope = (bottom_radius - top_radius) / height;
    let ring = |angle: f32, radius: f32, y: f32| {
        let radial = Vec3::new(angle.sin(), 0.0, angle.cos());
        PrimitiveVertex {
            position: radial * radius + Vec3::Y * y,
            normal: Vec3::new(radial.x, slope, radial.z).normalize(),
        }
    };
    for segment in 0..segments {
        let a = std::f32::consts::TAU * segment as f32 / segments as f32;
        let b = std::f32::consts::TAU * (segment + 1) as f32 / segments as f32;
        let bottom_a = ring(a, bottom_radius, -half);
        let bottom_b = ring(b, bottom_radius, -half);
        let top_a = ring(a, top_radius, half);
        let top_b = ring(b, top_radius, half);
        triangles.push([bottom_a, bottom_b, top_b]);
        if top_radius > 0.0 {
            triangles.push([bottom_a, top_b, top_a]);
        }
        if capped && top_radius > 0.0 {
            let center = PrimitiveVertex {
                position: Vec3::Y * half,
                normal: Vec3::Y,
            };
            let mut edge_a = top_a;
            let mut edge_b = top_b;
            edge_a.normal = Vec3::Y;
            edge_b.normal = Vec3::Y;
            triangles.push([center, edge_a, edge_b]);
        }
        if capped && bottom_radius > 0.0 {
            let center = PrimitiveVertex {
                position: Vec3::NEG_Y * half,
                normal: Vec3::NEG_Y,
            };
            let mut edge_a = bottom_a;
            let mut edge_b = bottom_b;
            edge_a.normal = Vec3::NEG_Y;
            edge_b.normal = Vec3::NEG_Y;
            triangles.push([center, edge_b, edge_a]);
        }
    }
    triangles
}

fn transform_matrix(position: Vec3, rotation: Vec3, scale: Vec3) -> Mat4 {
    Mat4::from_scale_rotation_translation(
        scale,
        Quat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z),
        position,
    )
}

fn dimensions(value: Option<&Vec<Number>>, default: [f32; 3]) -> Result<Vec3, String> {
    let size = vec3(value).unwrap_or(Vec3::from_array(default));
    if !size.is_finite() || size.min_element() <= 0.0 {
        return Err("box size must contain three positive finite numbers".into());
    }
    Ok(size)
}

fn vec3(value: Option<&Vec<Number>>) -> Option<Vec3> {
    let value = value?;
    (value.len() == 3).then(|| {
        Vec3::new(
            number(value.first()).unwrap_or(0.0),
            number(value.get(1)).unwrap_or(0.0),
            number(value.get(2)).unwrap_or(0.0),
        )
    })
}

fn scale(value: Option<&NumberOrNumbers>) -> Vec3 {
    match value {
        Some(NumberOrNumbers::Number(value)) => Vec3::splat(number(Some(value)).unwrap_or(1.0)),
        Some(NumberOrNumbers::Numbers(value)) => vec3(Some(value)).unwrap_or(Vec3::ONE),
        None => Vec3::ONE,
    }
}

fn number(value: Option<&Number>) -> Option<f32> {
    value.and_then(Number::as_f64).map(|value| value as f32)
}

fn positive(value: f32, name: &str) -> Result<f32, String> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("{name} must be positive and finite"))
    }
}

fn non_negative(value: f32, name: &str) -> Result<f32, String> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(format!("{name} must be non-negative and finite"))
    }
}

fn linear_rgba(hex: &str) -> Result<[f32; 4], String> {
    let [r, g, b] = linear_rgb(hex)?;
    Ok([r, g, b, 1.0])
}

fn linear_rgb(hex: &str) -> Result<[f32; 3], String> {
    let hex = hex
        .strip_prefix('#')
        .ok_or_else(|| format!("color {hex:?} must start with #"))?;
    let bytes = match hex.len() {
        3 => [
            u8::from_str_radix(&hex[0..1].repeat(2), 16),
            u8::from_str_radix(&hex[1..2].repeat(2), 16),
            u8::from_str_radix(&hex[2..3].repeat(2), 16),
        ],
        6 => [
            u8::from_str_radix(&hex[0..2], 16),
            u8::from_str_radix(&hex[2..4], 16),
            u8::from_str_radix(&hex[4..6], 16),
        ],
        _ => return Err(format!("color #{hex} must contain 3 or 6 hex digits")),
    };
    let bytes = bytes.map(|value| value.map_err(|_| format!("invalid hex color #{hex}")));
    let [r, g, b] = [bytes[0].clone()?, bytes[1].clone()?, bytes[2].clone()?];
    Ok([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)])
}

fn srgb_to_linear(channel: u8) -> f32 {
    let value = channel as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crystal::{ComponentDescriptor, EcsWorld, FieldSpec};
    use serde_json::json;
    use std::collections::BTreeMap;

    fn buffer_component(world: &mut EcsWorld, name: &str) -> crystal::ComponentId {
        world
            .register_component(ComponentDescriptor {
                name: name.into(),
                fields: BTreeMap::<String, FieldSpec>::new(),
                enableable: false,
                buffer: true,
                default: None,
            })
            .unwrap()
    }

    fn test_parameters() -> SceneParameters {
        SceneParameters {
            fov_y_degrees: 60.0,
            near: 0.1,
            far: 4000.0,
            sky_top: "#20152f".into(),
            sky_horizon: "#9a627d".into(),
            mesh_color: "#9aa0a6".into(),
            radial_segments: 24,
            camera_position: [0.0, 2.0, 22.0],
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            cluster_error_threshold: 1.0,
            tick_dt: 1.0 / 60.0,
            sun: SunDefaults {
                sun_color: "#ffe2b0".into(),
                sun_intensity: 1.1,
                sun_position: [60.0, 90.0, 30.0],
                ambient_intensity: 0.32,
            },
            emission_intensity: 2.5,
        }
    }

    #[test]
    fn from_ecs_derives_world_space_box_vertices_and_spawn_camera() {
        let mut world = EcsWorld::default();
        let transform = buffer_component(&mut world, "transform");
        let mesh = buffer_component(&mut world, "mesh");
        let spawn = buffer_component(&mut world, "spawn");

        let spawn_entity = world
            .create_entity(vec![(spawn, json!({"position": [0, 2, 10], "yaw": 0}))])
            .unwrap();
        world.bind_gaia_id("known_spawn", spawn_entity).unwrap();

        // A 2×2×2 box centred at world (3, 0, -4): corners span [2,-1,-5]..[4,1,-3].
        let box_entity = world
            .create_entity(vec![
                (transform, json!({"position": [3, 0, -4]})),
                (
                    mesh,
                    json!({"parts": [{"shape": "box", "size": [2, 2, 2], "color": "#804020"}]}),
                ),
            ])
            .unwrap();
        world.bind_gaia_id("known_box", box_entity).unwrap();

        let scene = RenderScene::from_ecs(world, &test_parameters()).unwrap();

        // One box = one material chain; 12 tris ≤ shard budget → a single leaf.
        assert_eq!(scene.chains.len(), 1);
        assert_eq!(scene.chains[0].dag.leaf_tri_sum(), 12);

        // Camera reads the spawn pose verbatim.
        assert_eq!(scene.camera.eye, Vec3::new(0.0, 2.0, 10.0));
        assert_eq!(scene.camera.yaw, 0.0);

        // The Great Chain draw path expands the cut back to the box: 6 faces ×
        // 2 triangles × 3 vertices, world-space (a single leaf is always drawn).
        let vertices = scene.select_vertices(&scene.camera, 640);
        assert_eq!(vertices.len(), 36);

        // World-space AABB matches the authored box exactly (no camera-relative bake).
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for vertex in &vertices {
            let position = Vec3::from_array(vertex.position);
            min = min.min(position);
            max = max.max(position);
        }
        assert!(
            (min - Vec3::new(2.0, -1.0, -5.0)).length() < 1e-5,
            "min {min:?}"
        );
        assert!(
            (max - Vec3::new(4.0, 1.0, -3.0)).length() < 1e-5,
            "max {max:?}"
        );
    }

    #[test]
    fn sun_reads_environment_over_defaults() {
        let mut world = EcsWorld::default();
        let environment = buffer_component(&mut world, "environment");
        let env_entity = world
            .create_entity(vec![(
                environment,
                json!({"sun": {"color": "#ff0000", "intensity": 2.0, "position": [0, 10, 0]}}),
            )])
            .unwrap();
        world.bind_gaia_id("env", env_entity).unwrap();

        let scene = RenderScene::from_ecs(world, &test_parameters()).unwrap();
        // #ff0000 → linear red 1.0, others 0.0; intensity read from the env.
        assert!((scene.sun.color[0] - 1.0).abs() < 1e-6);
        assert!(scene.sun.color[1] < 1e-6 && scene.sun.color[2] < 1e-6);
        assert!((scene.sun.intensity - 2.0).abs() < 1e-6);
        // Sun at +Y → direction toward sun is +Y.
        assert!((scene.sun.direction[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn leaf_triangles_split_albedo_and_emission_by_material() {
        let scene = naruko_scene();
        let tris = scene.leaf_triangles();
        assert!(!tris.is_empty(), "the realm has leaf geometry");
        // Leaf triangle sum equals the summed leaf triangles of every chain
        // (loss-free — the BVH sees the whole exact surface).
        let leaf_tris: usize = scene.chains.iter().map(|c| c.dag.leaf_tri_sum()).sum();
        assert_eq!(tris.len(), leaf_tris);
        // Emitters carry emission and zero albedo; non-emitters the reverse.
        let emitters = tris.iter().filter(|t| t.emission != [0.0; 3]).count();
        let reflectors = tris.iter().filter(|t| t.albedo != [0.0; 3]).count();
        assert!(emitters > 0, "lanterns/windows glow");
        assert!(reflectors > 0, "piers/terra reflect");
        for t in &tris {
            let is_emitter = t.emission != [0.0; 3];
            assert_eq!(is_emitter, t.albedo == [0.0; 3], "emitter xor reflector");
        }
    }

    // ---- Rite III ordeals: the Great Chain is THE geometry path ----

    use crystal::load_world_dir;
    use std::path::{Path, PathBuf};

    fn naruko_world() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worlds/naruko")
    }

    fn naruko_scene() -> RenderScene {
        let mut world = EcsWorld::default();
        load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
        RenderScene::from_ecs(world, &test_parameters()).expect("transmute the realm")
    }

    // ── A2 · the medium light is BOUND to real realm entities ──

    /// `emissive_sources` reads the real emitters from the realm: the stall's
    /// lantern is present at its authored world position with its emissive
    /// colour — the medium binds to THIS, never an invented light.
    #[test]
    fn emissive_sources_read_the_real_lantern() {
        let mut world = EcsWorld::default();
        load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
        let sources = emissive_sources(&world).expect("emissive sources");
        assert!(!sources.is_empty(), "the realm has emitters");
        let lantern = sources
            .iter()
            .find(|s| s.id == "naruko_lantern")
            .expect("the stall lantern is an emissive source");
        // Authored: entity [-7.5,0,20] + emissive sphere part [0,3.5,0].
        assert!((lantern.position[0] - (-7.5)).abs() < 1e-4, "{lantern:?}");
        assert!((lantern.position[1] - 3.5).abs() < 1e-4, "{lantern:?}");
        assert!((lantern.position[2] - 20.0).abs() < 1e-4, "{lantern:?}");
        // #ff9db0 → linear: red channel saturates to 1.0, warm pink.
        assert!((lantern.color[0] - 1.0).abs() < 1e-4, "{lantern:?}");
        assert!(lantern.color[1] < lantern.color[0] && lantern.color[2] < lantern.color[0]);
        // The emissive SPHERE radius is read verbatim (0.55) for the πr² area.
        assert!((lantern.radius - 0.55).abs() < 1e-4, "{lantern:?}");
    }

    /// `top_flat_surface_y` derives the plume's grounding from the stall's
    /// geometry — the top face of its highest flat slab (the serving surface),
    /// not a plucked constant. The upright posts (vertical extent largest) are
    /// correctly rejected.
    #[test]
    fn stall_top_flat_surface_is_derived_from_geometry() {
        let mut world = EcsWorld::default();
        load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
        let y = top_flat_surface_y(&world, "naruko_stall_massing")
            .expect("query")
            .expect("the stall has a flat serving surface");
        // The roof slab [5.6,0.3,4] at entity y=0, part y=2.75 → top 2.75+0.15.
        assert!((y - 2.9).abs() < 1e-4, "derived surface y = {y}");
        // A realm without the entity yields None (no plucked fallback).
        assert!(
            top_flat_surface_y(&world, "no_such_entity")
                .expect("query")
                .is_none()
        );
    }

    /// Mean y of every corner of a triangle set — a cheap centroid to watch a
    /// bob move geometry through the pipeline.
    fn centroid_y(tris: &[LeafTriangle]) -> f64 {
        if tris.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f64;
        for t in tris {
            for p in &t.positions {
                sum += p[1] as f64;
            }
        }
        sum / (tris.len() as f64 * 3.0)
    }

    fn mat_key(color: &str, emissive: bool) -> MatKey {
        let rgb = linear_rgb(color).unwrap();
        let emissive = f32::from(emissive);
        // Default lambertian dials (metallic 0, roughness 1) — this presence
        // test cares only about the colour/emissive band.
        (
            [rgb[0].to_bits(), rgb[1].to_bits(), rgb[2].to_bits()],
            emissive.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
        )
    }

    /// Two independent transmutations of the realm produce identical Great
    /// Chains — same cluster count, byte-identical serialization (FORMAT.md
    /// determinism invariant). Cluster count is READ from the build, never
    /// hardcoded (it grows as the realm does).
    #[test]
    fn naruko_chain_is_deterministic_and_double_builds_byte_identical() {
        let first = naruko_scene();
        let second = naruko_scene();
        assert_eq!(
            first.chains.len(),
            second.chains.len(),
            "chain count stable"
        );
        assert!(!first.chains.is_empty(), "the realm has geometry");

        let mut total_clusters = 0usize;
        for (a, b) in first.chains.iter().zip(&second.chains) {
            assert_eq!(a.color, b.color, "chain material order stable");
            let bytes_a = transmutation::serialize(&a.dag).expect("serialize chain A");
            let bytes_b = transmutation::serialize(&b.dag).expect("serialize chain B");
            assert_eq!(bytes_a, bytes_b, "double build must be byte-identical");
            total_clusters += a.dag.clusters.len();
        }
        eprintln!(
            "[ordeal] Naruko Great Chain: {} chains, {} clusters",
            first.chains.len(),
            total_clusters
        );
        assert!(
            total_clusters >= first.chains.len(),
            "each chain has ≥1 cluster"
        );
    }

    /// Draw-parity band assert: the WHOLE traced surface (static cut ∪ the living
    /// layer) still carries every signature material of the keyart. The lantern
    /// rose and the lit beacon now ride the DYNAMIC partition (they carry
    /// behaviors), so the UNION — not the static cut alone — must preserve them,
    /// and the dynamic materials must have LEFT the static chains (clean split).
    #[test]
    fn naruko_selected_cut_preserves_every_material_band() {
        let scene = naruko_scene();
        let vertices = scene.select_vertices(&scene.camera, 640);
        assert!(!vertices.is_empty(), "the cut drew geometry");

        // The cut's `Vertex` carries only colour/emissive; pad with the default
        // lambertian dials so the key type matches (this is a band-presence test).
        let vkey = |v: &Vertex| -> MatKey {
            (
                [
                    v.color[0].to_bits(),
                    v.color[1].to_bits(),
                    v.color[2].to_bits(),
                ],
                v.emissive.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
            )
        };
        let static_present: std::collections::BTreeSet<MatKey> =
            vertices.iter().map(vkey).collect();
        let mut present = static_present.clone();
        // The living layer contributes its own material bands — fold them in.
        for entity in scene.dynamics.entities() {
            for (color, emissive) in &entity.materials {
                present.insert((
                    [color[0].to_bits(), color[1].to_bits(), color[2].to_bits()],
                    f32::from(*emissive).to_bits(),
                    0.0f32.to_bits(),
                    1.0f32.to_bits(),
                ));
            }
        }

        for (label, color, emissive) in [
            ("pier brown", "#4a3626", false),
            ("lantern rose", "#ff9db0", true),
            ("warm window", "#ffb46b", true),
            ("lit beacon", "#f3e9ff", true),
        ] {
            assert!(
                present.contains(&mat_key(color, emissive)),
                "the traced surface lost the {label} band ({color}, emissive={emissive})"
            );
        }

        // The dynamic materials must NOT leak into the static cut (split clean).
        assert!(
            !static_present.contains(&mat_key("#ff9db0", true)),
            "lantern rose must have left the static chains (it is dynamic)"
        );
        assert!(
            !static_present.contains(&mat_key("#f3e9ff", true)),
            "lit beacon must have left the static chains (it is dynamic)"
        );

        // Sky gradient endpoints intact (linear sRGB of the night preset).
        assert_eq!(scene.sky_top, linear_rgba("#2a1a3e").unwrap());
        assert_eq!(scene.sky_horizon, linear_rgba("#d98ba8").unwrap());
    }

    /// DYNAMIC SPLIT correctness + leaf parity through the TRACED path: entities
    /// carrying a `behavior` are excluded from the static BVH triangles and kept
    /// as the dynamic partition, with NO triangle lost or duplicated. Naruko
    /// carries the lantern (bob) + beacon (pulse) + the three SIGNAL RINGS
    /// (pulse — the lighthouse broadcasts) + the Mirror Proof's kami orb
    /// (orbit) + RITE VI · VI-1's four `body` vessels (naruko_crate + the
    /// three stack crates) + REALM SHINE's three orbiting emitters
    /// (naruko_show_light_a/b/c — orbit; the show_chrome sphere and show_mirror
    /// panel carry NO behavior/body ⇒ they stay STATIC). PLAYGROUND adds nine
    /// more `body` vessels the Architect can push — the 5-crate stack, the
    /// bonded break crate, and the 3-crate pyramid — all dynamic (13 + 9 = 22).
    #[test]
    fn dynamic_split_leaf_parity_holds() {
        let scene = naruko_scene();
        assert_eq!(
            scene.dynamics.entities().len(),
            22,
            "the realm breath: lantern + beacon + ring_a/b/c + kami orb + show_light_a/b/c (behaviors) + crate + stack_crate_0/1/2 + playground stack(5)/bonded/pyramid(3) (bodies) are dynamic"
        );

        // STATIC BVH triangles (built once) and the DYNAMIC partition triangles.
        // Rite V: the dynamic partition is now the living layer PLUS the embodied
        // bodies (nari's `body` sigil) — account for the body triangles apart.
        let static_tris = scene.leaf_triangles().len();
        let dyn_tris = scene.dynamic_leaf_triangles().len();
        let body_tris: usize = scene.bodies.iter().map(|b| b.world_tris.len()).sum();
        assert!(dyn_tris > 0, "dynamic entities carry geometry");
        // The living-layer partition equals the bind triangle sum (transform
        // preserves count); the body is the separate Rite-V partition on top.
        let bind_sum: usize = scene
            .dynamics
            .entities()
            .iter()
            .map(|e| e.bind_tris.len())
            .sum();
        assert_eq!(
            scene.dynamics.leaf_triangles().len(),
            bind_sum,
            "transform never drops a living-layer triangle"
        );
        assert_eq!(
            dyn_tris,
            bind_sum + body_tris,
            "dynamic partition = living layer + embodied bodies"
        );

        // INDEPENDENT total: rebuild the SAME realm with every `behavior` AND
        // `body` stripped — now everything is static. static + dynamic must equal
        // this undivided leaf count EXACTLY (the split neither drops nor dups).
        let undivided = {
            let mut world = EcsWorld::default();
            load_world_dir(naruko_world(), &mut world).expect("load the Naruko realm");
            for marker in ["behavior", "body"] {
                if let Some(component) = world.component_id(marker) {
                    let carriers = world.query(&QuerySpec {
                        all: vec![component],
                        ..Default::default()
                    });
                    for entity in carriers {
                        world.remove_component(entity, component).unwrap();
                    }
                }
            }
            let all_static = RenderScene::from_ecs(world, &test_parameters()).unwrap();
            assert!(
                all_static.dynamics.entities().is_empty(),
                "stripping behaviors leaves no dynamic entities"
            );
            all_static.leaf_triangles().len()
        };
        // The undivided rebuild counts static MESH leaves only — a `body` sigil
        // carries no mesh, so nari's skinned triangles live solely in the
        // dynamic partition. The conservation identity therefore carries the
        // body count explicitly: static + dynamic == undivided-mesh + body.
        assert_eq!(
            static_tris + dyn_tris,
            undivided + body_tris,
            "static BVH tris + dynamic tris == undivided mesh leaves + body (no loss, no dup)"
        );

        // And the MERGED BVH the GPU walks carries exactly that total.
        use crate::bvh::{Bvh, BvhParams};
        let params = BvhParams::default();
        let static_bvh = Bvh::build(&scene.leaf_triangles(), &params);
        let dyn_bvh = Bvh::build(&scene.dynamic_leaf_triangles(), &params);
        let merged = Bvh::merge(&static_bvh, &dyn_bvh);
        assert_eq!(
            merged.tris.len(),
            static_tris + dyn_tris,
            "the spliced BVH carries every static + dynamic triangle"
        );
        eprintln!(
            "[ordeal] dynamic split: {static_tris} static tris + {dyn_tris} dynamic tris = {} total (merged BVH {} nodes)",
            static_tris + dyn_tris,
            merged.nodes.len(),
        );
    }

    /// TICK DETERMINISM: the clock counts ticks, never wall time. Two runs of N
    /// ticks produce byte-identical dynamic model-transform buffers at every step.
    #[test]
    fn tick_determinism_byte_identical_model_buffer() {
        let run = || {
            let mut scene = naruko_scene();
            let mut bytes = Vec::new();
            for _ in 0..300u64 {
                scene.tick();
                for m in scene.dynamics.model_matrices() {
                    bytes.extend_from_slice(bytemuck::bytes_of(&m));
                }
            }
            bytes
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len(), "model-buffer stream length");
        assert_eq!(a, b, "model transforms must be byte-identical across runs");
        eprintln!(
            "[ordeal] tick determinism: 2 runs × 300 ticks, {} bytes, byte-identical",
            a.len()
        );
    }

    /// BOB math matches KAMI's formula THROUGH the full traced pipeline: data →
    /// `tick_decorative` → ECS transform → model delta → world-space triangle.
    /// The lantern's model y-translation must equal `sin(t·speed+phase)·amplitude`
    /// at every tick, and its transformed triangles must ride that same offset.
    #[test]
    fn bob_matches_kami_through_the_pipeline() {
        let dt = 1.0f64 / 60.0;
        let bob = kami::Decorative::Bob {
            speed: 0.8,
            phase: 0.0,
            amplitude: 0.12,
        };
        let bind = kami::BindPose {
            position: [-7.5, 0.0, 20.0],
            ..kami::BindPose::default()
        };
        let mut scene = naruko_scene();
        // Capture the lantern's bind triangle centroid (before any tick).
        let bind_centroid = {
            let lantern = scene
                .dynamics
                .entities()
                .iter()
                .find(|e| e.gaia_id == "naruko_lantern")
                .expect("the lantern is a dynamic entity");
            centroid_y(&lantern.bind_tris)
        };
        let mut worst = 0.0f64;
        for k in 0..240u64 {
            scene.tick(); // after this call clock==k+1, model reflects t = k*dt
            let t = k as f64 * dt;
            let want_dy = bob.eval(t, bind).position[1]; // == sin(t*0.8)*0.12
            let lantern = scene
                .dynamics
                .entities()
                .iter()
                .find(|e| e.gaia_id == "naruko_lantern")
                .expect("the lantern is a dynamic entity");
            // model = M(animated) * M(bind)⁻¹ = pure y-translation for a bob.
            let got_dy = lantern.model.to_cols_array()[13] as f64;
            worst = worst.max((got_dy - want_dy).abs());
            assert!(
                (got_dy - want_dy).abs() <= 1e-5,
                "tick {k}: model dy {got_dy} != kami bob {want_dy}"
            );
            assert!(got_dy.abs() <= 0.12 + 1e-6, "bob within amplitude 0.12");
            // The TRANSFORMED triangles ride the same offset (the traced proof in
            // miniature: geometry the BVH sees actually moved by want_dy).
            let world_centroid = centroid_y(&lantern.world_triangles());
            assert!(
                (world_centroid - (bind_centroid + want_dy)).abs() <= 1e-4,
                "tick {k}: world triangle centroid didn't follow the bob"
            );
        }
        eprintln!("[ordeal] bob pipeline parity: 240 ticks vs kami eval, worst err={worst:.3e}");
    }

    /// At τ → 0 the cut selects the finest LOD everywhere: the emitted triangle
    /// count equals the summed leaf triangles of every chain (geometry parity —
    /// leaves are the loss-free shardized input).
    #[test]
    fn finest_threshold_reproduces_leaf_geometry() {
        let mut scene = naruko_scene();
        scene.error_threshold = 0.0;
        let leaf_tris: usize = scene.chains.iter().map(|c| c.dag.leaf_tri_sum()).sum();
        let vertices = scene.select_vertices(&scene.camera, 640);
        assert_eq!(vertices.len(), leaf_tris * 3, "finest cut == all leaves");
    }

    // ---- VII-0b ordeal (c): THE COORDINATE SEAM — translation invariance ----

    /// `terrain_placement_offset` MUST route `i64 -> f64 -> subtract ->
    /// f32` (never `i64 -> f32` directly, which collapses past `2^24` m —
    /// VII-0a's module doc), so that a residual offset between a tile's
    /// origin and the render origin stays EXACT regardless of how far either
    /// magnitude sits from the world origin — THE camera-relative-rendering
    /// guarantee, provable here without a far camera anywhere in a realm.
    ///
    /// Two regimes: NEAR (tile origin and render origin both close to zero)
    /// and FAR (tile_x = 10_000_000, render_origin CO-LOCATED with that same
    /// huge tile origin, both magnitudes in the hundreds of millions of
    /// meters). In both regimes the render-relative residual is engineered
    /// to be the identical small vector `(1.5, 0.0, -2.25)` (an arbitrary
    /// small "camera not quite at the tile origin" offset). If the far
    /// regime's f64 subtraction lost precision, the resulting f32 offset
    /// would drift from the near regime's.
    ///
    /// SCOPE (post-adversary-review correction): this is the OFFSET
    /// ARITHMETIC in isolation — it does not, by itself, prove the
    /// PRODUCTION `from_ecs_at` weld actually threads `render_origin`
    /// through correctly, nor does composing one shared local mesh with two
    /// offsets prove anything about DIFFERENT tiles' generated content (two
    /// genuinely different tiles, e.g. `(0,0)` vs `(10_000_000,-10_000_000)`,
    /// legitimately generate DIFFERENT height content — different global
    /// grid indices sample different noise, VII-0a's per-tile independence
    /// by design — so "bit-identical leaf triangles between near and far
    /// tiles" is not even a true claim to make). The real end-to-end proof,
    /// through `from_ecs_at` -> `append_terrain` on an actual loaded realm,
    /// with the content-vs-placement distinction kept honest, lives in
    /// `packages/scrying-glass/tests/vii0b_terrain.rs`'s
    /// `render_origin_at_planetary_tile_magnitude_reproduces_the_local_mesh_through_the_real_weld`.
    #[test]
    fn terrain_placement_offset_is_translation_invariant_at_planetary_tile_magnitude() {
        let params = seed::TerrainParams::default();

        // NEAR regime: tile (0,0), tile_origin = (0,0). Render origin offset
        // by the small residual so tile_origin - render_origin = -residual.
        let near_tile = seed::TerrainTile::new(0, 0);
        let near_tile_origin = seed::tile_origin_m(near_tile, &params);
        let residual = [1.5_f64, 0.0, -2.25];
        let near_render_origin = [
            near_tile_origin.0 - residual[0],
            0.0 - residual[1],
            near_tile_origin.1 - residual[2],
        ];
        let near_offset = terrain_placement_offset(near_tile_origin, near_render_origin);

        // FAR regime: tile (10_000_000, -10_000_000) — a planet-scale tile
        // coordinate (Ruling 4's own worked example). Render origin tracks
        // the SAME residual relative to THIS tile's (huge) origin.
        let far_tile = seed::TerrainTile::new(10_000_000, -10_000_000);
        let far_tile_origin = seed::tile_origin_m(far_tile, &params);
        let far_render_origin = [
            far_tile_origin.0 - residual[0],
            0.0 - residual[1],
            far_tile_origin.1 - residual[2],
        ];
        let far_offset = terrain_placement_offset(far_tile_origin, far_render_origin);

        assert_eq!(
            near_offset, far_offset,
            "the SAME residual, at planetary tile magnitude, must produce a \
             BIT-IDENTICAL f32 placement offset — {near_offset:?} vs {far_offset:?}"
        );
        assert_eq!(
            near_offset,
            Vec3::new(residual[0] as f32, residual[1] as f32, residual[2] as f32),
            "the offset must equal the residual exactly (both magnitudes cancel \
             in f64 before the f32 cast)"
        );
    }
}
