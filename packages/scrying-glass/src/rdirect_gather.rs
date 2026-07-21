//! NEURAL-LIVE N0.b — FEATURE GATHER driver. Builds the net's `[N, 23]` input
//! tensor on the GPU straight from the live frame's existing buffers (the
//! integrator's trace-resolution `accum` radiance + native-resolution `aov`
//! G-buffer), writing it into a caller-owned STORAGE buffer. That destination
//! is, on the live Metal path, the pooled MTLBuffer the MPSGraph forward reads
//! zero-copy (see `rdirect_live::RdirectLive`): the gather is one compute
//! dispatch, no per-frame allocation.
//!
//! The shader (`rdirect_gather.wgsl`) is bit-for-bit the CPU
//! `rdirect::pixel_features` — parity-gated in tests/rdirect_gather_ordeals.rs.
//!
//! CURRENT-FRAME ONLY. BAN-SCOPED

use bytemuck::{Pod, Zeroable};

use crate::rdirect::{CamPose, HIST_FEATURES_SPLIT, INPUT_FEATURES, INPUT_FEATURES_SPLIT};

/// The gather uniform: `dims = (low_w, low_h, target_w, target_h)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherUniform {
    dims: [u32; 4],
}

pub const GATHER_SHADER: &str = include_str!("rdirect_gather.wgsl");

/// A built feature-gather compute pass. Construct once; `encode` any number of
/// frames (all buffers are supplied by the caller — this owns only the pipeline
/// and a small reusable uniform buffer, so there is zero per-frame allocation).
pub struct FeatureGather {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
}

impl FeatureGather {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rdirect gather"),
            source: wgpu::ShaderSource::Wgsl(GATHER_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rdirect gather layout"),
            entries: &[
                // uniform dims
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GatherUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),  // accum (read)
                storage_entry(2, true),  // aov (read)
                storage_entry(3, false), // feats (read_write)
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rdirect gather pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rdirect gather pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gather"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect gather uniform"),
            size: std::mem::size_of::<GatherUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            uniform_buf,
        }
    }

    /// Bytes the destination feature buffer must hold for `n` target pixels.
    pub fn feature_bytes(n: usize) -> u64 {
        (n * INPUT_FEATURES * std::mem::size_of::<f32>()) as u64
    }

    /// Encode one gather dispatch. `accum` = trace-res accumulation cells,
    /// `aov` = native-res AOVs (2 cells/px), `feats` = the `[N,23]` destination
    /// STORAGE buffer (≥ `feature_bytes(target_w*target_h)`). All current-frame.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        accum: &wgpu::Buffer,
        aov: &wgpu::Buffer,
        feats: &wgpu::Buffer,
        low_w: u32,
        low_h: u32,
        target_w: u32,
        target_h: u32,
    ) {
        let uniform = GatherUniform {
            dims: [low_w, low_h, target_w, target_h],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rdirect gather bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: aov.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: feats.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rdirect gather pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let gx = target_w.div_ceil(8);
        let gy = target_h.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

// ── V7-LIVE LANE STAGE 1: split (E/D) feature gather ───────────────────────
// Sibling of `FeatureGather` above, over the SAME house pattern, for the
// 35-feature `INPUT_FEATURES_SPLIT` layout (no history — that's Stage 2).
// Reads the integrator's `accum_ed` split buffer instead of the composite
// `accum`. Additive: never constructed unless the live path opts in
// (`GAIA_NATIVE_EVIDENCE_SPLIT`), so the 23-in `FeatureGather` path above is
// byte-untouched when this type is never built.
pub const GATHER_SPLIT_SHADER: &str = include_str!("rdirect_gather_split.wgsl");

pub struct FeatureGatherSplit {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
}

impl FeatureGatherSplit {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rdirect gather split"),
            source: wgpu::ShaderSource::Wgsl(GATHER_SPLIT_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rdirect gather split layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GatherUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),  // accum_ed (read)
                storage_entry(2, true),  // aov (read)
                storage_entry(3, false), // feats (read_write)
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rdirect gather split pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rdirect gather split pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gather_split"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect gather split uniform"),
            size: std::mem::size_of::<GatherUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            uniform_buf,
        }
    }

    /// Bytes the destination feature buffer must hold for `n` target pixels
    /// (35-feature `INPUT_FEATURES_SPLIT` rows, no history).
    pub fn feature_bytes(n: usize) -> u64 {
        (n * INPUT_FEATURES_SPLIT * std::mem::size_of::<f32>()) as u64
    }

    /// Encode one split-gather dispatch. `accum_ed` = the integrator's split
    /// trace-res accumulation (2 vec4 cells/px, E then D — see
    /// `integrator::dispatch_split`), `aov` = SAME native-res AOVs the 23-in
    /// gather reads, `feats` = the `[N,35]` destination STORAGE buffer
    /// (≥ `feature_bytes(target_w*target_h)`). All current-frame.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        accum_ed: &wgpu::Buffer,
        aov: &wgpu::Buffer,
        feats: &wgpu::Buffer,
        low_w: u32,
        low_h: u32,
        target_w: u32,
        target_h: u32,
    ) {
        let uniform = GatherUniform {
            dims: [low_w, low_h, target_w, target_h],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rdirect gather split bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum_ed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: aov.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: feats.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rdirect gather split pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        let gx = target_w.div_ceil(8);
        let gy = target_h.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

// ── V7-LIVE LANE STAGE 2: recurrent history (39-in) ─────────────────────────
// `FeatureGatherHistSplit` mirrors `FeatureGatherSplit` (@group(0): dims/
// accum_ed/aov/feats, feats now 39-wide) plus a SECOND bind group
// (@group(1)) carrying the ping-pong history: the previous frame's net
// output (demod-log, bilinearly resampled) + its AOV (for the depth/normal
// reject test) + both cameras (current, for the reprojection ray; previous,
// for the reprojection target) — see `rdirect_gather_split.wgsl`'s
// `gather_hist_split` entry, a bit-for-bit port of CPU
// `direct_render_sequence_hist_split`'s reprojection block. Additive: never
// constructed unless the live path opts in (same `GAIA_NATIVE_EVIDENCE_SPLIT`
// gate as Stage 1 — Stage 3 wires it into `NetPresent`), so `FeatureGather`/
// `FeatureGatherSplit` above stay byte-untouched.

/// GPU-layout camera pose — field-for-field mirror of CPU [`CamPose`], packed
/// into 4 `vec4<f32>`s (`std140`-friendly, no reordering games):
/// `eye_ht.xyz`=eye, `.w`=half_tan; `right_asp.xyz`=right, `.w`=aspect;
/// `up.xyz`=up; `fwd.xyz`=forward.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CamGpu {
    pub eye_ht: [f32; 4],
    pub right_asp: [f32; 4],
    pub up: [f32; 4],
    pub fwd: [f32; 4],
}

impl From<CamPose> for CamGpu {
    fn from(c: CamPose) -> Self {
        Self {
            eye_ht: [c.eye.x, c.eye.y, c.eye.z, c.half_tan],
            right_asp: [c.right.x, c.right.y, c.right.z, c.aspect],
            up: [c.up.x, c.up.y, c.up.z, 0.0],
            fwd: [c.forward.x, c.forward.y, c.forward.z, 0.0],
        }
    }
}

/// The `@group(1)` history uniform — mirrors `HistU` in the WGSL exactly.
/// `params = (prev_w, prev_h, has_prev, depth_tol)`,
/// `params2 = (normal_thresh, sky_reject, 0, 0)` — `sky_reject` is
/// `GAIA_V7_SKY_HISTORY=reject` as 1.0/0.0 (see
/// `rdirect::sky_history_reject`), read once per `encode` call, mirroring
/// the CPU reference's `direct_render_sequence_hist_split`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HistUniform {
    cur: CamGpu,
    prev: CamGpu,
    params: [f32; 4],
    params2: [f32; 4],
}

pub const GATHER_HIST_SPLIT_SHADER: &str = GATHER_SPLIT_SHADER;

pub struct FeatureGatherHistSplit {
    pipeline: wgpu::ComputePipeline,
    layout0: wgpu::BindGroupLayout,
    layout1: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    hist_uniform_buf: wgpu::Buffer,
}

impl FeatureGatherHistSplit {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rdirect gather hist split"),
            source: wgpu::ShaderSource::Wgsl(GATHER_HIST_SPLIT_SHADER.into()),
        });
        let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rdirect gather hist split layout0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GatherUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),  // accum_ed (read)
                storage_entry(2, true),  // aov (read)
                storage_entry(3, false), // feats (read_write, 39-wide rows)
            ],
        });
        let layout1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rdirect gather hist split layout1"),
            entries: &[
                storage_entry(0, true), // prev_out_dl (read)
                storage_entry(1, true), // prev_aov (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<HistUniform>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rdirect gather hist split pipeline layout"),
            bind_group_layouts: &[Some(&layout0), Some(&layout1)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rdirect gather hist split pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gather_hist_split"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect gather hist split uniform"),
            size: std::mem::size_of::<GatherUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hist_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect gather hist split hist-uniform"),
            size: std::mem::size_of::<HistUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, layout0, layout1, uniform_buf, hist_uniform_buf }
    }

    /// Bytes the destination feature buffer must hold for `n` target pixels
    /// (39-feature `HIST_FEATURES_SPLIT` rows).
    pub fn feature_bytes(n: usize) -> u64 {
        (n * HIST_FEATURES_SPLIT * std::mem::size_of::<f32>()) as u64
    }

    /// Bytes a `prev_out_dl` ping-pong buffer must hold for `n` native pixels
    /// (one `vec4<f32>` per pixel — xyz used, w padding).
    pub fn out_dl_bytes(n: usize) -> u64 {
        (n * 16) as u64
    }

    /// Encode one hist-gather dispatch. `accum_ed`/`aov` are this frame's own
    /// (same as `FeatureGatherSplit`); `prev_out_dl` is the previous frame's
    /// net output (demod-log, `out_dl_bytes(n)` sized) and `prev_aov` its AOV
    /// buffer (`prev_w × prev_h`, same 2-cell/px layout) — both `None` (or
    /// `has_prev=false`) on the first frame / after a history reset, in which
    /// case history features are zeroed exactly as the CPU reference zeroes
    /// them. `feats` is the `[N,39]` destination STORAGE buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        accum_ed: &wgpu::Buffer,
        aov: &wgpu::Buffer,
        feats: &wgpu::Buffer,
        prev_out_dl: &wgpu::Buffer,
        prev_aov: &wgpu::Buffer,
        cur_cam: CamPose,
        prev_cam: CamPose,
        has_prev: bool,
        prev_w: u32,
        prev_h: u32,
        depth_tol: f32,
        normal_thresh: f32,
        low_w: u32,
        low_h: u32,
        target_w: u32,
        target_h: u32,
    ) {
        let uniform = GatherUniform { dims: [low_w, low_h, target_w, target_h] };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let sky_reject = if crate::rdirect::sky_history_reject() { 1.0 } else { 0.0 };
        let hist_uniform = HistUniform {
            cur: cur_cam.into(),
            prev: prev_cam.into(),
            params: [prev_w as f32, prev_h as f32, if has_prev { 1.0 } else { 0.0 }, depth_tol],
            params2: [normal_thresh, sky_reject, 0.0, 0.0],
        };
        queue.write_buffer(&self.hist_uniform_buf, 0, bytemuck::bytes_of(&hist_uniform));
        let bind0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rdirect gather hist split bind0"),
            layout: &self.layout0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum_ed.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: aov.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: feats.as_entire_binding() },
            ],
        });
        let bind1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rdirect gather hist split bind1"),
            layout: &self.layout1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: prev_out_dl.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prev_aov.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.hist_uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rdirect gather hist split pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind0, &[]);
        pass.set_bind_group(1, &bind1, &[]);
        let gx = target_w.div_ceil(8);
        let gy = target_h.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }
}

/// GPU-resident ping-pong history state: the previous frame's net output
/// (demod-log) + its AOV (depth/normal, for the reject test) + camera pose.
/// `swap` (called once a frame, AFTER that frame's `gather_hist_split` has
/// consumed the OLD `prev_*`) copies the CURRENT frame's output/AOV into the
/// history buffers on the GPU (no CPU round-trip) and remembers the current
/// camera for next frame's reprojection target. `reset` drops history (scene
/// cut / resize) — the next `encode` then sees `has_prev=false` and the CPU
/// reference's own first-frame rule (prev_dl=0, valid=0) takes over.
pub struct HistoryBuffers {
    pub prev_out_dl: wgpu::Buffer,
    pub prev_aov: wgpu::Buffer,
    pub has_prev: bool,
    pub prev_cam: Option<CamPose>,
    pub w: u32,
    pub h: u32,
}

impl HistoryBuffers {
    pub fn new(device: &wgpu::Device, w: u32, h: u32) -> Self {
        let n = (w as usize) * (h as usize);
        let prev_out_dl = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect history prev_out_dl"),
            size: FeatureGatherHistSplit::out_dl_bytes(n).max(1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prev_aov = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect history prev_aov"),
            size: ((n as u64).max(1)) * 32, // 2 vec4<f32> cells/px, same as integrator AOV_CELL*2
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { prev_out_dl, prev_aov, has_prev: false, prev_cam: None, w, h }
    }

    /// Drop history (next frame's gather sees `has_prev=false`).
    pub fn reset(&mut self) {
        self.has_prev = false;
        self.prev_cam = None;
    }

    /// GPU-side copy of this frame's own output (`cur_out_dl`, `out_dl_bytes`
    /// sized) + AOV (`cur_aov`, `w×h` sized) into the history buffers, and
    /// remember `cur_cam` — call once per frame AFTER this frame's own
    /// `gather_hist_split` dispatch has read the OLD history (the ping-pong
    /// swap point).
    pub fn swap(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        cur_out_dl: &wgpu::Buffer,
        cur_aov: &wgpu::Buffer,
        cur_cam: CamPose,
        w: u32,
        h: u32,
    ) {
        let n = (w as usize) * (h as usize);
        encoder.copy_buffer_to_buffer(
            cur_out_dl, 0, &self.prev_out_dl, 0, FeatureGatherHistSplit::out_dl_bytes(n),
        );
        encoder.copy_buffer_to_buffer(cur_aov, 0, &self.prev_aov, 0, (n as u64) * 32);
        self.prev_cam = Some(cur_cam);
        self.has_prev = true;
        self.w = w;
        self.h = h;
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
