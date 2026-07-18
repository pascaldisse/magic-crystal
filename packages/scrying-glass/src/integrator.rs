//! The traced integrator (Rite IV, L1) — the GPU half of the Pleroma in the
//! glass. Wraps the compute path tracer (`integrator.wgsl`) and its present blit
//! into one reusable spirit the window (`main.rs`) and the ordeals both drive.
//!
//! The window builds one [`Integrator`] over the realm's BVH, keeps a persistent
//! accumulation buffer, dispatches one frame per tick (accumulating while the
//! camera is still, reset on move), then blits the resolved radiance to the
//! surface. The ordeals build the same integrator headlessly and read the
//! accumulation buffer back for parity/determinism.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::bvh::Bvh;
use crate::scene::{Camera, SunLight};

pub const INTEGRATOR_SHADER: &str = include_str!("integrator.wgsl");

/// GPU integrator uniform. Field layout matches the WGSL `Uniform` exactly
/// (all vec4-aligned blocks).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IntegratorUniform {
    pub eye: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
    pub sky_top: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    /// width, height, spp, max_bounces.
    pub params: [u32; 4],
    /// seed, samples_before, node_count, tri_count.
    pub counters: [u32; 4],
    /// ambient_intensity, eps, rr_start, unused.
    pub misc: [f32; 4],
    /// Rite VI A1 medium: xyz grid world origin, w = voxel_size.
    pub med_origin: [f32; 4],
    /// sigma_a, sigma_s, g (HG anisotropy), far cap.
    pub med_params: [f32; 4],
    /// march_steps, shadow_steps, shadow_dist, enabled (0/1).
    pub med_march: [f32; 4],
    /// grid dims xyz, w unused.
    pub med_dims: [u32; 4],
    /// The medium's bound scene light: for a directional light xyz = unit
    /// direction TOWARD it; for a point light xyz = world POSITION. w = the
    /// scalar intensity (radiance scale, or radiant intensity fed through
    /// 1/dist²).
    pub med_light: [f32; 4],
    /// rgb = the light's colour tint (multiplied outside the scalar scatter);
    /// w = light KIND (0 = directional sun/moon, 1 = point emitter glow).
    pub med_light_color: [f32; 4],
    /// Display target: target_w, target_h, nearest=1, unused. Rendering stays
    /// at `params.xy`; the shader applies nearest integer display scaling only.
    pub surface: [u32; 4],
    // ── LIGHT-NOT-DOTS: temporal accumulation with reprojection ──
    /// Previous frame's eye (xyz) for reprojection.
    pub prev_eye: [f32; 4],
    /// Previous frame's scaled image-plane right.
    pub prev_right: [f32; 4],
    /// Previous frame's scaled image-plane up.
    pub prev_up: [f32; 4],
    /// Previous frame's unit forward.
    pub prev_forward: [f32; 4],
    /// alpha_min (moving EMA floor), depth_tol, normal_tol (cos), clamp_k.
    pub temporal: [f32; 4],
    /// history_valid (0/1), max_history frames, unused, unused.
    pub temporal_flags: [u32; 4],
}

/// Temporal accumulation dials (LIGHT-NOT-DOTS) — env-parameterised at the call
/// site, never hardcoded. See `temporal_resolve` in integrator.wgsl.
#[derive(Clone, Copy, Debug)]
pub struct TemporalParams {
    /// EMA floor for the CURRENT frame while the camera MOVES (caps effective
    /// history so moving content stays responsive; a still camera ignores it and
    /// converges with a pure 1/n running average).
    pub alpha_min: f32,
    /// Relative depth agreement for accepting reprojected history (|d-d'|/d).
    pub depth_tol: f32,
    /// Minimum normal agreement (cosine) for accepting reprojected history.
    pub normal_tol: f32,
    /// Neighbourhood variance clamp width (±k·sigma). Applied EVERY frame now
    /// (gateless), so a still-camera relight re-converges too.
    pub clamp_k: f32,
    /// Hard cap on accumulated frame count (avoids unbounded 1/n stagnation).
    pub max_history: u32,
    /// Sub-pixel image-motion budget (in PIXELS) below which a frame is treated
    /// as numerically STILL: identity reproject + pure 1/n running average +
    /// clamp off (deep, exact convergence). ABOVE it the frame reprojects, floors
    /// alpha, and clamps. This must be NEAR ZERO, not a generous sub-pixel box:
    /// a SUSTAINED sub-pixel pan (0.1°/frame is already >0.05px at trace res)
    /// smears just as badly under identity reproject as a fast one, because the
    /// displacement GROWS every frame while a 1/n average keeps old frames
    /// weighted — that was the Architect's ghost. 0.05px only snaps true stillness
    /// (float noise) and a tremor that oscillates in place. Derived against the
    /// pixel angular size in the shader — replaces the frozen 0.99999 gate.
    pub still_px: f32,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            alpha_min: 0.1,
            depth_tol: 0.05,
            normal_tol: 0.85,
            clamp_k: 1.5,
            max_history: 512,
            still_px: 0.05,
        }
    }
}

/// A participating medium uploaded to the GPU: the density volume (Aether's
/// rasterized grid values, f32) plus its optical + march parameters. Built from
/// the SAME `aether` types the CPU reference marches, so the two paths share one
/// artifact (the parity ordeal's whole point). Plain primitives — the crate
/// stays aether-free at runtime; the tests do the conversion.
#[derive(Clone, Debug)]
pub struct MediumGpu {
    /// Grid resolution in cells (x, y, z).
    pub dims: [u32; 3],
    /// Cubic cell edge length (world units).
    pub voxel_size: f32,
    /// Grid box minimum corner (world space).
    pub world_origin: [f32; 3],
    /// Absorption coefficient (per unit density).
    pub sigma_a: f32,
    /// Scattering coefficient (per unit density).
    pub sigma_s: f32,
    /// Henyey-Greenstein anisotropy g.
    pub g: f32,
    /// Far cap for the primary march when the camera ray escapes.
    pub far: f32,
    /// Camera-ray march step count.
    pub march_steps: u32,
    /// Shadow-ray march step count (self-shadowing toward the sun).
    pub shadow_steps: u32,
    /// Bound on the occlusion march toward a DIRECTIONAL light (a point light
    /// uses its true source distance instead).
    pub shadow_dist: f32,
    /// The bound scene light (A2): a real realm entity, never invented.
    pub light: MediumLightGpu,
    /// Light colour tint (linear rgb), multiplied outside the scalar scatter.
    pub light_color: [f32; 3],
    /// Scalar intensity: incident radiance (directional) or radiant intensity
    /// fed through 1/dist² (point).
    pub light_intensity: f32,
    /// Density values (x-fastest, then y, z), length = dims.x*dims.y*dims.z.
    pub density: Vec<f32>,
}

/// The medium's bound scene light on the GPU — the SAME split the CPU
/// `pleroma::MediumLight` carries (A2 true binding). The vector field means a
/// unit direction (directional) or a world position (point); the `kind` byte
/// selects which in the shader.
#[derive(Clone, Copy, Debug)]
pub enum MediumLightGpu {
    /// Sun / moon: `to_light` is a unit direction TOWARD the light.
    Directional { to_light: [f32; 3] },
    /// An emissive entity's glow: `position` is its world-space position,
    /// radiance falls off as intensity/dist².
    Point { position: [f32; 3] },
}

impl MediumLightGpu {
    /// (xyz vector, kind flag): 0 = directional, 1 = point.
    fn pack(&self) -> ([f32; 3], f32) {
        match *self {
            MediumLightGpu::Directional { to_light } => (to_light, 0.0),
            MediumLightGpu::Point { position } => (position, 1.0),
        }
    }
}

/// Integrator dials (never hardcode — env-parameterised at the call site).
#[derive(Clone, Copy, Debug)]
pub struct IntegratorParams {
    /// Samples per pixel per accumulation frame.
    pub spp: u32,
    /// Hard path-length cap.
    pub max_bounces: u32,
    /// Bounce index at which russian roulette begins.
    pub rr_start: u32,
    /// Master seed (ENTROPY origin).
    pub seed: u32,
    /// Ray self-intersection epsilon.
    pub eps: f32,
}

impl Default for IntegratorParams {
    fn default() -> Self {
        Self {
            spp: 2,
            max_bounces: 4,
            rr_start: 2,
            seed: 0x5eed,
            eps: 1e-3,
        }
    }
}

impl IntegratorUniform {
    /// Build a uniform for one accumulation frame. `samples_before` is the total
    /// sample count already in the accumulation buffer (0 on reset).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        camera: &Camera,
        sun: &SunLight,
        sky_top: [f32; 4],
        sky_horizon: [f32; 4],
        width: u32,
        height: u32,
        node_count: u32,
        tri_count: u32,
        samples_before: u32,
        params: &IntegratorParams,
        medium: Option<&MediumGpu>,
    ) -> Self {
        let (right, up, forward) = camera.basis();
        let aspect = width as f32 / height.max(1) as f32;
        let half = (camera.fov_y_radians * 0.5).tan();
        let right = right * (half * aspect);
        let up = up * half;
        IntegratorUniform {
            eye: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            forward: [forward.x, forward.y, forward.z, 0.0],
            sky_top,
            sky_horizon,
            sun_dir: [sun.direction[0], sun.direction[1], sun.direction[2], 0.0],
            sun_color: [sun.color[0], sun.color[1], sun.color[2], sun.intensity],
            params: [width, height, params.spp, params.max_bounces],
            counters: [params.seed, samples_before, node_count, tri_count],
            misc: [
                sun.ambient_intensity,
                params.eps,
                params.rr_start as f32,
                0.0,
            ],
            med_origin: match medium {
                Some(m) => [
                    m.world_origin[0],
                    m.world_origin[1],
                    m.world_origin[2],
                    m.voxel_size,
                ],
                None => [0.0; 4],
            },
            med_params: match medium {
                Some(m) => [m.sigma_a, m.sigma_s, m.g, m.far],
                None => [0.0; 4],
            },
            med_march: match medium {
                Some(m) => [
                    m.march_steps as f32,
                    m.shadow_steps as f32,
                    m.shadow_dist,
                    1.0,
                ],
                None => [0.0; 4],
            },
            med_dims: match medium {
                Some(m) => [m.dims[0], m.dims[1], m.dims[2], 0],
                None => [0; 4],
            },
            med_light: match medium {
                Some(m) => {
                    let (v, _kind) = m.light.pack();
                    [v[0], v[1], v[2], m.light_intensity]
                }
                None => [0.0; 4],
            },
            med_light_color: match medium {
                Some(m) => {
                    let (_v, kind) = m.light.pack();
                    [m.light_color[0], m.light_color[1], m.light_color[2], kind]
                }
                None => [0.0; 4],
            },
            // Default: canvas-sized nearest identity present.
            surface: [width, height, 1, 0],
            // Temporal defaults: history invalid (identity — output = current),
            // so every non-temporal caller is byte-unchanged. The window loop
            // overrides these when temporal accumulation is enabled.
            prev_eye: [0.0; 4],
            prev_right: [0.0; 4],
            prev_up: [0.0; 4],
            prev_forward: [0.0; 4],
            temporal: [0.1, 0.05, 0.85, 1.5],
            temporal_flags: [0, 512, 0, 0],
        }
    }
}

/// The GPU tracer resources: compute + blit pipelines, the immutable BVH buffers,
/// and a shared uniform buffer.
pub struct Integrator {
    pub compute_pipeline: wgpu::ComputePipeline,
    pub blit_pipeline: wgpu::RenderPipeline,
    pub compute_layout: wgpu::BindGroupLayout,
    pub blit_layout: wgpu::BindGroupLayout,
    pub uniform_buf: wgpu::Buffer,
    node_buf: wgpu::Buffer,
    tri_buf: wgpu::Buffer,
    density_buf: wgpu::Buffer,
    pub node_count: u32,
    pub tri_count: u32,
    // ── VIII-0 AOV EXPORT BEGIN ──────────────────────────────────────────
    // A SECOND compute pipeline (own bind group layout at @group(1)) so the
    // AOV export dial is a split pipeline, not a new binding threaded onto
    // the existing one-light-pass `compute_layout`/`compute_pipeline` — the
    // pre-existing path above is untouched by this wave (see integrator.wgsl
    // "VIII-0 AOV EXPORT" block and the AOV-off golden-hash ordeal).
    pub aov_pipeline: wgpu::ComputePipeline,
    pub aov_layout: wgpu::BindGroupLayout,
    // ── VIII-0 AOV EXPORT END ────────────────────────────────────────────
    // ── LIGHT-NOT-DOTS: temporal accumulation ──
    // Two more split pipelines over the SAME @group(0) plus a shared @group(1)
    // of five storage buffers. When temporal accumulation is off neither is
    // dispatched — zero cost, and `integrate`/`blit` are byte-for-byte unchanged.
    pub temporal_integrate_pipeline: wgpu::ComputePipeline,
    pub temporal_resolve_pipeline: wgpu::ComputePipeline,
    pub temporal_layout: wgpu::BindGroupLayout,
}

/// Bytes one accumulation cell occupies (vec4<f32>).
const ACCUM_CELL: u64 = 16;

/// Bytes one AOV pixel occupies: 2 vec4<f32> cells (albedo+depth,
/// normal+hit) — see integrator.wgsl "VIII-0 AOV EXPORT" for the packing.
const AOV_CELL: u64 = 32;

impl Integrator {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        bvh: &Bvh,
        medium: Option<&MediumGpu>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L1 traced integrator"),
            source: wgpu::ShaderSource::Wgsl(INTEGRATOR_SHADER.into()),
        });

        // An empty realm still needs a nonzero buffer so bindings are valid.
        let node_bytes = bytemuck::cast_slice(&bvh.nodes);
        let tri_bytes = bytemuck::cast_slice(&bvh.tris);
        let node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh nodes"),
            contents: if node_bytes.is_empty() {
                &[0u8; 32]
            } else {
                node_bytes
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh triangles"),
            contents: if tri_bytes.is_empty() {
                &[0u8; 80]
            } else {
                tri_bytes
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        // The medium density volume (binding 4). Always present so the binding
        // is valid; a disabled medium uploads a single zero and the shader's
        // `enabled` flag short-circuits the march.
        let density_bytes: Vec<u8> = match medium {
            Some(m) if !m.density.is_empty() => bytemuck::cast_slice(&m.density).to_vec(),
            _ => bytemuck::cast_slice(&[0.0f32]).to_vec(),
        };
        let density_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("medium density volume"),
            contents: &density_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integrator uniform"),
            size: std::mem::size_of::<IntegratorUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<IntegratorUniform>() as u64
                ),
            },
            count: None,
        };
        let storage_entry = |binding, read_only, vis| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrator compute layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(4, true, wgpu::ShaderStages::COMPUTE),
            ],
        });
        // The blit shares the single `accum` global (declared read_write for the
        // compute pass), so its layout must also expose binding 3 as read_write
        // even though the fragment only reads it (WGSL global access must match).
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrator blit layout"),
            entries: &[
                uniform_entry,
                storage_entry(3, false, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        // ── VIII-0 AOV EXPORT BEGIN ────────────────────────────────────────
        // Own bind group layout at @group(1): a single read_write storage
        // buffer for the packed albedo/normal/depth cells. `compute_layout`
        // (@group(0)) is reused UNCHANGED as the AOV pipeline's group 0 —
        // nothing above this block was touched to add AOV export.
        let aov_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrator aov layout"),
            entries: &[storage_entry(0, false, wgpu::ShaderStages::COMPUTE)],
        });
        // ── VIII-0 AOV EXPORT END ──────────────────────────────────────────

        // The three pipelines (compute / blit / aov) are built by the SHARED
        // `build_pipelines` — the SAME path das Blutbändigen's shader bend
        // (`reload_shader`) re-runs on a live WGSL edit, so a hot-swapped
        // module goes through byte-identical pipeline construction (law 1: the
        // layouts/buffers OUTLIVE the swapped module).
        let (compute_pipeline, blit_pipeline, aov_pipeline) = Self::build_pipelines(
            device,
            &shader,
            &compute_layout,
            &blit_layout,
            &aov_layout,
            target_format,
        );

        // ── LIGHT-NOT-DOTS temporal pipelines ──
        // @group(1): FOUR read_write storage buffers (t_cur packed, t_prev
        // packed, hist_prev, hist_out) shared by both temporal entry points
        // — four here + four in group(0) = the 8-storage-per-stage limit.
        let temporal_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrator temporal layout"),
            entries: &[
                storage_entry(0, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let temporal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrator temporal pipeline layout"),
                bind_group_layouts: &[Some(&compute_layout), Some(&temporal_layout)],
                immediate_size: 0,
            });
        let temporal_integrate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("integrator temporal integrate pipeline"),
                layout: Some(&temporal_pipeline_layout),
                module: &shader,
                entry_point: Some("integrate_temporal"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let temporal_resolve_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("integrator temporal resolve pipeline"),
                layout: Some(&temporal_pipeline_layout),
                module: &shader,
                entry_point: Some("temporal_resolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            compute_pipeline,
            blit_pipeline,
            compute_layout,
            blit_layout,
            uniform_buf,
            node_buf,
            tri_buf,
            density_buf,
            node_count: bvh.nodes.len() as u32,
            tri_count: bvh.tris.len() as u32,
            aov_pipeline,
            aov_layout,
            temporal_integrate_pipeline,
            temporal_resolve_pipeline,
            temporal_layout,
        }
    }

    /// Build the three integrator pipelines (compute `integrate`, blit
    /// `blit_vs`/`blit_fs`, AOV `integrate_aov`) over an already-created shader
    /// module and the persistent bind-group layouts. Shared by `new` and by
    /// das Blutbändigen's `reload_shader` so a live WGSL swap reconstructs the
    /// pipelines by the identical path — only the module changes; the layouts,
    /// buffers and bind groups persist untouched (BLOODBEND law 1).
    fn build_pipelines(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        compute_layout: &wgpu::BindGroupLayout,
        blit_layout: &wgpu::BindGroupLayout,
        aov_layout: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> (
        wgpu::ComputePipeline,
        wgpu::RenderPipeline,
        wgpu::ComputePipeline,
    ) {
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrator compute pipeline layout"),
                bind_group_layouts: &[Some(compute_layout)],
                immediate_size: 0,
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integrator compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: shader,
            entry_point: Some("integrate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("integrator blit pipeline layout"),
            bind_group_layouts: &[Some(blit_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("integrator blit pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("blit_vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("blit_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let aov_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("integrator aov pipeline layout"),
            bind_group_layouts: &[Some(compute_layout), Some(aov_layout)],
            immediate_size: 0,
        });
        let aov_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integrator aov pipeline"),
            layout: Some(&aov_pipeline_layout),
            module: shader,
            entry_point: Some("integrate_aov"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        (compute_pipeline, blit_pipeline, aov_pipeline)
    }

    /// DAS BLUTBÄNDIGEN — the SHADER SUB-DOOR (B0). Recompile the integrator's
    /// WGSL from a live source string and swap the pipelines ONLY on a clean
    /// validation. A wgpu Validation error scope wraps the whole module +
    /// pipeline construction (FULL-MOON RULE: reject-before-apply, zero partial
    /// effect); on ANY compile/validation error the new module is discarded and
    /// the OLD pipelines keep rendering untouched — the caller gets the error
    /// string for a police report. On success the three pipeline fields are
    /// swapped; every buffer, layout and bind group persists (the API separates
    /// state from code — law 1).
    pub fn reload_shader(
        &mut self,
        device: &wgpu::Device,
        source: &str,
        target_format: wgpu::TextureFormat,
    ) -> Result<(), String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("L1 traced integrator (bent)"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let (compute, blit, aov) = Self::build_pipelines(
            device,
            &module,
            &self.compute_layout,
            &self.blit_layout,
            &self.aov_layout,
            target_format,
        );
        // The scope captures every validation error raised above (parse, naga
        // validation, pipeline creation). If any fired, the new pipelines are
        // invalid — DISCARD them, keep the old ones live.
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(format!("{error}"));
        }
        self.compute_pipeline = compute;
        self.blit_pipeline = blit;
        self.aov_pipeline = aov;
        Ok(())
    }

    /// Re-upload the acceleration structure after the living layer re-splices the
    /// dynamic partition (Rite IV dynamics). Recreates the node/tri storage
    /// buffers (their sizes track the moving geometry) and updates the counts;
    /// the caller must then rebuild any bind groups that reference them (they
    /// bind these buffers) and reset accumulation (moved geometry invalidates it).
    pub fn update_bvh(&mut self, device: &wgpu::Device, bvh: &Bvh) {
        let node_bytes = bytemuck::cast_slice(&bvh.nodes);
        let tri_bytes = bytemuck::cast_slice(&bvh.tris);
        self.node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh nodes"),
            contents: if node_bytes.is_empty() {
                &[0u8; 32]
            } else {
                node_bytes
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.tri_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh triangles"),
            contents: if tri_bytes.is_empty() {
                &[0u8; 80]
            } else {
                tri_bytes
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.node_count = bvh.nodes.len() as u32;
        self.tri_count = bvh.tris.len() as u32;
    }

    /// Allocate a fresh accumulation buffer for a `width×height` frame, zeroed.
    pub fn make_accum(&self, device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let cells = (width as u64) * (height as u64);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integrator accumulation"),
            size: (cells.max(1)) * ACCUM_CELL,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    // ── LIGHT-NOT-DOTS temporal helpers ─────────────────────────────────────
    /// Allocate one PACKED temporal frame buffer: 2 `vec4<f32>` cells per pixel
    /// (radiance + primary gbuffer), zeroed. The window ping-pongs two of these.
    pub fn make_temporal_packed(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> wgpu::Buffer {
        let cells = (width as u64) * (height as u64);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integrator temporal packed"),
            size: (cells.max(1)) * ACCUM_CELL * 2,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Allocate one history buffer (rgb + accumulated frame count), 1 cell/pixel.
    pub fn make_temporal_buffer(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> wgpu::Buffer {
        let cells = (width as u64) * (height as u64);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integrator temporal history"),
            size: (cells.max(1)) * ACCUM_CELL,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A @group(1) bind group binding the four temporal buffers in order:
    /// (t_cur packed, t_prev packed, hist_prev, hist_out). The window ping-pongs
    /// the packed pair and the history pair across frames by building two.
    pub fn temporal_bind_group(
        &self,
        device: &wgpu::Device,
        cur_packed: &wgpu::Buffer,
        prev_packed: &wgpu::Buffer,
        hist_prev: &wgpu::Buffer,
        hist_out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrator temporal bind group"),
            layout: &self.temporal_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cur_packed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prev_packed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hist_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hist_out.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the two temporal passes (trace-this-frame + reproject/resolve)
    /// into one encoder. `compute_bg` is the ordinary @group(0) (uniform/nodes/
    /// tris/accum/density); `temporal_bg` is the @group(1) built above. The
    /// resolve writes the accumulated radiance into the accum bound in
    /// `compute_bg`, so the existing blit presents it unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_temporal(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        uniform: &IntegratorUniform,
        compute_bg: &wgpu::BindGroup,
        temporal_bg: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniform));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temporal integrate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_integrate_pipeline);
            pass.set_bind_group(0, compute_bg, &[]);
            pass.set_bind_group(1, temporal_bg, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temporal resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_resolve_pipeline);
            pass.set_bind_group(0, compute_bg, &[]);
            pass.set_bind_group(1, temporal_bg, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
    }

    // ── VIII-0 AOV EXPORT BEGIN ────────────────────────────────────────────
    /// Allocate a fresh AOV buffer for a `width×height` frame: 2 `vec4<f32>`
    /// cells per pixel (see integrator.wgsl "VIII-0 AOV EXPORT" for the
    /// packing). Zeroed, like `make_accum` — but note the AOV shader writes
    /// EVERY pixel unconditionally (hit or miss), so the zero fill is never
    /// actually read back; it exists only so the buffer is valid before the
    /// first dispatch.
    pub fn make_aov_buffer(&self, device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let cells = (width as u64) * (height as u64);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integrator aov export"),
            size: (cells.max(1)) * AOV_CELL,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Bind group for the AOV pipeline's @group(1) (the packed output buffer
    /// alone — @group(0) is the ordinary `compute_bind_group`).
    pub fn aov_bind_group(&self, device: &wgpu::Device, aov: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrator aov bind group"),
            layout: &self.aov_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: aov.as_entire_binding(),
            }],
        })
    }

    /// Dispatch one AOV export pass: current-frame-only (see
    /// integrate_aov in integrator.wgsl) — there is no `frames` loop here
    /// because there is nothing to accumulate; one dispatch is the complete,
    /// deterministic answer for this camera pose.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_aov(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        uniform: &IntegratorUniform,
        compute_bg: &wgpu::BindGroup,
        aov_bg: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniform));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("integrate aov"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.aov_pipeline);
        pass.set_bind_group(0, compute_bg, &[]);
        pass.set_bind_group(1, aov_bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    // ── VIII-0 AOV EXPORT END ──────────────────────────────────────────────

    pub fn compute_bind_group(
        &self,
        device: &wgpu::Device,
        accum: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrator compute bind group"),
            layout: &self.compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tri_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: accum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.density_buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn blit_bind_group(&self, device: &wgpu::Device, accum: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrator blit bind group"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: accum.as_entire_binding(),
                },
            ],
        })
    }

    /// Write the uniform and dispatch one accumulation frame into `accum`.
    pub fn dispatch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        uniform: &IntegratorUniform,
        compute_bg: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniform));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("integrate frame"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, compute_bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    /// Present the accumulation buffer to `view` (a *Srgb target).
    pub fn blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        blit_bg: &wgpu::BindGroup,
        label: &'static str,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, blit_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A headless GPU device for the ordeals (no surface). Returns None when no
/// adapter is available (documented: the ordeal then cannot run on this host).
pub fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

/// Trace `frames` accumulation frames of a scene headlessly and read the
/// accumulation buffer back (per-pixel sum in xyz, sample count in w). The
/// ordeals' workhorse: parity vs the Pleroma, determinism, shadow correctness.
#[allow(clippy::too_many_arguments)]
pub fn trace_headless(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    width: u32,
    height: u32,
    frames: u32,
    params: &IntegratorParams,
    medium: Option<&MediumGpu>,
) -> Vec<[f32; 4]> {
    let integrator = Integrator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, bvh, medium);
    let accum = integrator.make_accum(device, width, height);
    let compute_bg = integrator.compute_bind_group(device, &accum);

    let mut samples_before = 0u32;
    for _ in 0..frames {
        let uniform = IntegratorUniform::build(
            camera,
            sun,
            sky_top,
            sky_horizon,
            width,
            height,
            integrator.node_count,
            integrator.tri_count,
            samples_before,
            params,
            medium,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless integrate"),
        });
        integrator.dispatch(queue, &mut encoder, &uniform, &compute_bg, width, height);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        samples_before += params.spp;
    }

    // Copy accum → a mappable readback buffer.
    let cells = (width as u64) * (height as u64);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless readback"),
        size: cells * ACCUM_CELL,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless copy"),
    });
    encoder.copy_buffer_to_buffer(&accum, 0, &readback, 0, cells * ACCUM_CELL);
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback channel").expect("map readback");
    let mapped = readback.get_mapped_range(..).expect("mapped readback");
    let out: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    out
}

/// Run the LIVE temporal path headlessly over a SEQUENCE of camera poses (the
/// motion the ordeal replays), driving the exact ping-pong + prev-camera wiring
/// `main.rs`'s render loop uses. Returns the final frame's resolved radiance
/// (one Vec3 per pixel) — the accumulated light, not raw dots. Each frame's
/// Monte-Carlo samples are decorrelated by advancing `samples_before`.
#[allow(clippy::too_many_arguments)]
pub fn trace_headless_temporal(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    cameras: &[Camera],
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    width: u32,
    height: u32,
    params: &IntegratorParams,
    temporal: &TemporalParams,
) -> Vec<Vec3> {
    let integrator = Integrator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, bvh, None);
    let accum = integrator.make_accum(device, width, height);
    let compute_bg = integrator.compute_bind_group(device, &accum);
    let packed = [
        integrator.make_temporal_packed(device, width, height),
        integrator.make_temporal_packed(device, width, height),
    ];
    let hist = [
        integrator.make_temporal_buffer(device, width, height),
        integrator.make_temporal_buffer(device, width, height),
    ];
    // Two ping-pong bind groups. Parity p: t_cur=packed[p] (written this frame),
    // t_prev=packed[1-p] (last frame's gbuffer), hist_prev=hist[p], hist_out=hist[1-p].
    let bind = [
        integrator.temporal_bind_group(device, &packed[0], &packed[1], &hist[0], &hist[1]),
        integrator.temporal_bind_group(device, &packed[1], &packed[0], &hist[1], &hist[0]),
    ];

    let mut prev: Option<IntegratorUniform> = None;
    for (i, camera) in cameras.iter().enumerate() {
        let mut uniform = IntegratorUniform::build(
            camera,
            sun,
            sky_top,
            sky_horizon,
            width,
            height,
            integrator.node_count,
            integrator.tri_count,
            (i as u32) * params.spp,
            params,
            None,
        );
        uniform.temporal = [
            temporal.alpha_min,
            temporal.depth_tol,
            temporal.normal_tol,
            temporal.clamp_k,
        ];
        let valid = if let Some(p) = prev {
            uniform.prev_eye = p.eye;
            uniform.prev_right = p.right;
            uniform.prev_up = p.up;
            uniform.prev_forward = p.forward;
            1
        } else {
            0
        };
        uniform.temporal_flags = [valid, temporal.max_history, temporal.still_px.to_bits(), 0];
        let parity = i % 2;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("headless temporal"),
        });
        integrator.dispatch_temporal(
            queue,
            &mut encoder,
            &uniform,
            &compute_bg,
            &bind[parity],
            width,
            height,
        );
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        prev = Some(uniform);
    }

    let cells = (width as u64) * (height as u64);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless temporal readback"),
        size: cells * ACCUM_CELL,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless temporal copy"),
    });
    encoder.copy_buffer_to_buffer(&accum, 0, &readback, 0, cells * ACCUM_CELL);
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback channel").expect("map readback");
    let mapped = readback.get_mapped_range(..).expect("mapped readback");
    let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    raw.iter().map(|c| Vec3::new(c[0], c[1], c[2])).collect()
}

// ── VIII-0 AOV EXPORT BEGIN ────────────────────────────────────────────────
/// Trace ONE AOV export pass headlessly and read the packed buffer back.
/// Current-frame-only (see integrate_aov in integrator.wgsl): this function
/// takes a single camera pose and the realm's geometry alone — no frame
/// index, no previous-frame buffer, no accumulation-across-frames parameter.
/// Calling it twice with identical arguments is the AOV determinism ordeal.
#[allow(clippy::too_many_arguments)]
pub fn trace_headless_aov(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bvh: &Bvh,
    camera: &Camera,
    sun: &SunLight,
    sky_top: [f32; 4],
    sky_horizon: [f32; 4],
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    let integrator = Integrator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, bvh, None);
    // The AOV pipeline's @group(0) is the same layout as the ordinary
    // compute pass, so it needs a valid (if here unused) accum buffer too.
    let accum = integrator.make_accum(device, width, height);
    let aov_buf = integrator.make_aov_buffer(device, width, height);
    let compute_bg = integrator.compute_bind_group(device, &accum);
    let aov_bg = integrator.aov_bind_group(device, &aov_buf);

    let uniform = IntegratorUniform::build(
        camera,
        sun,
        sky_top,
        sky_horizon,
        width,
        height,
        integrator.node_count,
        integrator.tri_count,
        0,
        &IntegratorParams::default(),
        None,
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless aov"),
    });
    integrator.dispatch_aov(
        queue,
        &mut encoder,
        &uniform,
        &compute_bg,
        &aov_bg,
        width,
        height,
    );
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    let cells = (width as u64) * (height as u64);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless aov readback"),
        size: cells * AOV_CELL,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless aov copy"),
    });
    encoder.copy_buffer_to_buffer(&aov_buf, 0, &readback, 0, cells * AOV_CELL);
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r.map(|_| ()));
    });
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("readback channel").expect("map readback");
    let mapped = readback.get_mapped_range(..).expect("mapped readback");
    let out: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    out
}

/// Split a raw AOV readback (2 cells/pixel, see `trace_headless_aov`) into
/// three per-pixel images: albedo (rgb), world normal (xyz, RAW `[-1,1]`
/// range — callers remap to `[0,1]` for display, see `viii0_truth.rs`), and
/// hit distance ("depth", 0.0 on a primary miss).
pub fn split_aov(raw: &[[f32; 4]]) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>) {
    let n = raw.len() / 2;
    let mut albedo = Vec::with_capacity(n);
    let mut normal = Vec::with_capacity(n);
    let mut depth = Vec::with_capacity(n);
    for i in 0..n {
        let a = raw[2 * i];
        let b = raw[2 * i + 1];
        albedo.push(Vec3::new(a[0], a[1], a[2]));
        depth.push(a[3]);
        normal.push(Vec3::new(b[0], b[1], b[2]));
    }
    (albedo, normal, depth)
}
// ── VIII-0 AOV EXPORT END ──────────────────────────────────────────────────

/// Convenience: the linear resolved image (radiance per pixel) from an accum
/// readback (sum ÷ samples).
pub fn resolve(accum: &[[f32; 4]]) -> Vec<Vec3> {
    accum
        .iter()
        .map(|c| {
            let s = c[3].max(1.0);
            Vec3::new(c[0] / s, c[1] / s, c[2] / s)
        })
        .collect()
}
