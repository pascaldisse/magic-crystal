//! R-DIRECT — THE NET IS THE RENDERER, GPU driver. The compute port of the CPU
//! reference direct renderer (src/rdirect.rs::direct_render_image). Uploads the
//! SAME hash-pinned weights (via [`crate::rdirect::deserialize_weights`], never
//! re-derived), the current frame's LOW-resolution radiance plus this frame's
//! TARGET-resolution G-buffer (albedo/normal/depth/motion), dispatches the
//! fused single-pass kernel (`rdirect.wgsl` f32, or `rdirect_fast.wgsl` fp16
//! MODE A), and reads the target-resolution image back. Per-TARGET-pixel MLP,
//! the exact house pattern of the VIII-3 upscaler port (`upscaler_gpu.rs`).
//!
//! THE BAN: current-frame buffers only. The shaders carry `// BAN-SCOPED` and
//! are scanned; nothing here indexes/stores/reads any earlier frame's output.
//! BAN-SCOPED

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::rdirect::Mlp;

/// Fixed ceilings mirrored from the shaders (`MAX_LAYERS`, `MAX_WIDTH`).
pub const MAX_LAYERS: usize = 16;
pub const MAX_WIDTH: u32 = 64;

/// The compute uniform. Layout matches the WGSL `RdirectU` exactly (all
/// vec4-aligned): `dims` = (low_w, low_h, target_w, target_h); `info` =
/// (layer_count, weight_count, _, _); each `layers[l]` = (in_dim, out_dim,
/// weights_off, bias_off).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RdirectUniform {
    dims: [u32; 4],
    info: [u32; 4],
    layers: [[u32; 4]; MAX_LAYERS],
}

pub const RDIRECT_SHADER: &str = include_str!("rdirect.wgsl");
/// The fp16-threadgroup-prefix-cached fast port (MODE A: f16 storage, f32
/// accumulate). Requires a device created with [`wgpu::Features::SHADER_F16`].
pub const RDIRECT_FAST_SHADER: &str = include_str!("rdirect_fast.wgsl");

/// A GPU direct-renderer bound to one net's weights. Build once (weights
/// upload + pipeline), then [`Self::render`] any number of current frames.
pub struct GpuRdirect {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: RdirectUniform,
    weights_buf: wgpu::Buffer,
}

impl GpuRdirect {
    /// Build the naive per-pixel port (device-storage weights, f32).
    pub fn new(device: &wgpu::Device, mlp: &Mlp) -> Self {
        Self::build(device, mlp, RDIRECT_SHADER)
    }

    /// Build the fp16-threadgroup-prefix-cached FAST port (MODE A). The device
    /// MUST have [`wgpu::Features::SHADER_F16`]. Output/inputs identical; only
    /// the on-GPU weight residence + f16 storage differ.
    pub fn new_fast(device: &wgpu::Device, mlp: &Mlp) -> Self {
        Self::build(device, mlp, RDIRECT_FAST_SHADER)
    }

    fn build(device: &wgpu::Device, mlp: &Mlp, shader_src: &str) -> Self {
        let dims = mlp.layer_dims();
        assert!(
            dims.len() <= MAX_LAYERS,
            "rdirect has {} layers, GPU port ceiling is {MAX_LAYERS}",
            dims.len()
        );
        let flat = mlp.flat_weights();
        let mut uniform = RdirectUniform {
            dims: [0, 0, 0, 0],
            info: [dims.len() as u32, flat.len() as u32, 0, 0],
            layers: [[0; 4]; MAX_LAYERS],
        };
        let mut offset: u32 = 0;
        for (l, &(in_dim, out_dim)) in dims.iter().enumerate() {
            assert!(
                out_dim <= MAX_WIDTH && in_dim <= MAX_WIDTH,
                "layer {l} width {in_dim}->{out_dim} exceeds GPU port ceiling {MAX_WIDTH}"
            );
            let w_off = offset;
            let b_off = offset + in_dim * out_dim;
            uniform.layers[l] = [in_dim, out_dim, w_off, b_off];
            offset = b_off + out_dim;
        }
        assert_eq!(
            flat.len() as u32,
            offset,
            "flat weight length disagrees with computed layer offsets"
        );

        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rdirect weights"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("R-Direct kernel"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<RdirectUniform>() as u64
                ),
            },
            count: None,
        };
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rdirect layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, true),  // weights
                storage_entry(2, true),  // low radiance
                storage_entry(3, true),  // hi albedo
                storage_entry(4, true),  // hi normal
                storage_entry(5, true),  // hi depth
                storage_entry(6, true),  // hi motion
                storage_entry(7, false), // out
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rdirect pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rdirect pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("render"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout,
            uniform,
            weights_buf,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn upload_inputs(
        &self,
        device: &wgpu::Device,
        low_radiance: &[Vec3],
        hi_albedo: &[Vec3],
        hi_normal: &[Vec3],
        hi_depth: &[f32],
        hi_motion: &[Vec2],
        low_n: usize,
        target_n: usize,
    ) -> InputBuffers {
        assert_eq!(low_radiance.len(), low_n);
        assert_eq!(hi_albedo.len(), target_n);
        assert_eq!(hi_normal.len(), target_n);
        assert_eq!(hi_depth.len(), target_n);
        assert_eq!(hi_motion.len(), target_n);

        let vec3_pack =
            |v: &[Vec3]| -> Vec<[f32; 4]> { v.iter().map(|c| [c.x, c.y, c.z, 0.0]).collect() };
        let low_p = vec3_pack(low_radiance);
        let albedo_p = vec3_pack(hi_albedo);
        let normal_p = vec3_pack(hi_normal);
        let depth_p: Vec<[f32; 4]> = hi_depth.iter().map(|&d| [d, 0.0, 0.0, 0.0]).collect();
        let motion_p: Vec<[f32; 4]> = hi_motion.iter().map(|m| [m.x, m.y, 0.0, 0.0]).collect();

        let mk = |label, data: &[[f32; 4]]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect out"),
            size: (target_n.max(1) * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        InputBuffers {
            low: mk("rdirect low radiance", &low_p),
            albedo: mk("rdirect hi albedo", &albedo_p),
            normal: mk("rdirect hi normal", &normal_p),
            depth: mk("rdirect hi depth", &depth_p),
            motion: mk("rdirect hi motion", &motion_p),
            out: out_buf,
        }
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        uniform_buf: &wgpu::Buffer,
        b: &InputBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rdirect bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.low.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: b.albedo.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: b.normal.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: b.depth.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: b.motion.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: b.out.as_entire_binding() },
            ],
        })
    }

    fn make_uniform(&self, low_w: u32, low_h: u32, target_w: u32, target_h: u32) -> RdirectUniform {
        let mut uniform = self.uniform;
        uniform.dims = [low_w, low_h, target_w, target_h];
        uniform
    }

    fn uniform_buf(&self, device: &wgpu::Device, u: &RdirectUniform, label: &str) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(u),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Direct-render one current frame on the GPU and read the result back. The
    /// returned image is the GPU transcription of
    /// [`crate::rdirect::direct_render_image`].
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        low_radiance: &[Vec3],
        low_w: u32,
        low_h: u32,
        hi_albedo: &[Vec3],
        hi_normal: &[Vec3],
        hi_depth: &[f32],
        hi_motion: &[Vec2],
        target_w: u32,
        target_h: u32,
    ) -> Vec<Vec3> {
        let low_n = (low_w * low_h) as usize;
        let target_n = (target_w * target_h) as usize;
        let b = self.upload_inputs(
            device, low_radiance, hi_albedo, hi_normal, hi_depth, hi_motion, low_n, target_n,
        );
        let uniform = self.make_uniform(low_w, low_h, target_w, target_h);
        let uniform_buf = self.uniform_buf(device, &uniform, "rdirect uniform");
        let bg = self.bind_group(device, &uniform_buf, &b);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rdirect encode"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rdirect pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(target_w.div_ceil(8), target_h.div_ceil(8), 1);
        }

        let bytes = (target_n * 16) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&b.out, 0, &readback, 0, bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r.map(|_| ()));
        });
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("readback channel").expect("map readback");
        let mapped = readback.get_mapped_range(..).expect("mapped readback");
        let cells: Vec<[f32; 4]> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        cells.iter().map(|c| Vec3::new(c[0], c[1], c[2])).collect()
    }

    /// Time `repeats` GPU render dispatches via GPU timestamp queries, returning
    /// per-dispatch milliseconds. The device MUST have
    /// [`wgpu::Features::TIMESTAMP_QUERY`]; returns `None` otherwise. Only the
    /// compute pass is bracketed — upload/readback excluded, so this is the
    /// pass cost against the frame budget, not the round-trip.
    #[allow(clippy::too_many_arguments)]
    pub fn time_dispatches_ms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        low_radiance: &[Vec3],
        low_w: u32,
        low_h: u32,
        hi_albedo: &[Vec3],
        hi_normal: &[Vec3],
        hi_depth: &[f32],
        hi_motion: &[Vec2],
        target_w: u32,
        target_h: u32,
        repeats: u32,
    ) -> Option<Vec<f64>> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let period_ns = queue.get_timestamp_period() as f64;
        let low_n = (low_w * low_h) as usize;
        let target_n = (target_w * target_h) as usize;
        let b = self.upload_inputs(
            device, low_radiance, hi_albedo, hi_normal, hi_depth, hi_motion, low_n, target_n,
        );
        let uniform = self.make_uniform(low_w, low_h, target_w, target_h);
        let uniform_buf = self.uniform_buf(device, &uniform, "rdirect uniform (timed)");
        let bg = self.bind_group(device, &uniform_buf, &b);

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("rdirect timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let ts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ts_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut out = Vec::with_capacity(repeats as usize);
        for _ in 0..repeats {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rdirect timed encode"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rdirect timed pass"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(target_w.div_ceil(8), target_h.div_ceil(8), 1);
            }
            encoder.resolve_query_set(&query_set, 0..2, &ts_buf, 0);
            encoder.copy_buffer_to_buffer(&ts_buf, 0, &ts_read, 0, 16);
            let (tx, rx) = std::sync::mpsc::channel();
            encoder.map_buffer_on_submit(&ts_read, wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r.map(|_| ()));
            });
            queue.submit(Some(encoder.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("ts channel").expect("map ts");
            let mapped = ts_read.get_mapped_range(..).expect("mapped ts");
            let ticks: &[u64] = bytemuck::cast_slice(&mapped);
            let delta = ticks[1].saturating_sub(ticks[0]) as f64;
            drop(mapped);
            ts_read.unmap();
            out.push(delta * period_ns / 1.0e6);
        }
        Some(out)
    }
}

struct InputBuffers {
    low: wgpu::Buffer,
    albedo: wgpu::Buffer,
    normal: wgpu::Buffer,
    depth: wgpu::Buffer,
    motion: wgpu::Buffer,
    out: wgpu::Buffer,
}

/// A headless GPU device requesting `SHADER_F16` (for the fast kernel) plus
/// `TIMESTAMP_QUERY` when available (for pass timing). Returns `None` when no
/// adapter is available OR the adapter lacks `SHADER_F16`.
pub fn headless_device_f16_timed() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    if !adapter.features().contains(wgpu::Features::SHADER_F16) {
        return None;
    }
    let wanted = wgpu::Features::SHADER_F16 | wgpu::Features::TIMESTAMP_QUERY;
    let features = adapter.features() & wanted;
    let mut desc = wgpu::DeviceDescriptor::default();
    desc.required_features = features;
    pollster::block_on(adapter.request_device(&desc)).ok()
}

/// A headless GPU device requesting only `TIMESTAMP_QUERY` (for timing the f32
/// kernel on hosts without `SHADER_F16`). Returns `None` when no adapter.
pub fn headless_device_timed() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
    let mut desc = wgpu::DeviceDescriptor::default();
    desc.required_features = features;
    pollster::block_on(adapter.request_device(&desc)).ok()
}
