//! RITE VIII-3 wave (b) — THE UPSCALER AT SPEED: the GPU driver for the compute
//! port of the VIII-3 CPU reference upscaler. Uploads the SAME hash-pinned
//! weights (loaded via [`crate::upscaler::deserialize_weights`], never
//! re-derived), the current frame's LOW-resolution radiance plus this frame's
//! TARGET-resolution auxiliary buffers (albedo/normal/depth), dispatches
//! `upscaler.wgsl`, and reads the target-resolution image back. A plain
//! compute MLP per TARGET pixel — the exact house pattern of the VIII-2
//! denoiser port (`denoiser_gpu.rs`); wgpu has no tensor surface (RENDER.md §8).
//!
//! THE BAN: this module takes current-frame buffers only. Its name matches the
//! `upscaler*.rs` grep-gate glob (viii0_ordeals), so it is scanned whole for
//! the forbidden cross-frame vocabulary; the shader carries its own
//! `// BAN-SCOPED` marker and is scanned by the VIII-3 ordeals. Nothing here
//! indexes, stores, or reads any earlier frame's output.
//!
//! BAN-SCOPED

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::upscaler::Mlp;

/// Fixed ceilings mirrored from `upscaler.wgsl` (`MAX_LAYERS`, `MAX_WIDTH`).
pub const MAX_LAYERS: usize = 16;
pub const MAX_WIDTH: u32 = 64;

/// The compute uniform. Layout matches the WGSL `UpscaleU` exactly (all
/// vec4-aligned): `dims` = (low_w, low_h, target_w, target_h); `info` =
/// (layer_count, _, _, _); each `layers[l]` = (in_dim, out_dim, weights_off,
/// bias_off).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpscaleUniform {
    dims: [u32; 4],
    info: [u32; 4],
    layers: [[u32; 4]; MAX_LAYERS],
}

pub const UPSCALER_SHADER: &str = include_str!("upscaler.wgsl");
/// The fp16-threadgroup-cached fast port (MODE A: f16 storage, f32 accumulate).
/// Requires a device created with [`wgpu::Features::SHADER_F16`].
pub const UPSCALER_FAST_SHADER: &str = include_str!("upscaler_fast.wgsl");

/// A GPU upscaler bound to one net's weights. Build once (weights upload +
/// pipeline), then [`Self::upscale`] any number of current frames.
pub struct GpuUpscaler {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: UpscaleUniform,
    weights_buf: wgpu::Buffer,
}

impl GpuUpscaler {
    /// Build the naive per-pixel port (device-storage weights, f32).
    pub fn new(device: &wgpu::Device, mlp: &Mlp) -> Self {
        Self::build(device, mlp, UPSCALER_SHADER)
    }

    /// Build the fp16-threadgroup-cached FAST port (MODE A). The device MUST
    /// have [`wgpu::Features::SHADER_F16`]. Output/inputs identical; only the
    /// on-GPU weight residence + f16 storage differ (parity ordeal derives the
    /// fp16 bound and asserts it).
    pub fn new_fast(device: &wgpu::Device, mlp: &Mlp) -> Self {
        Self::build(device, mlp, UPSCALER_FAST_SHADER)
    }

    /// Build the pipeline and upload `mlp`'s flat weights. The per-layer
    /// offsets into the flat buffer are computed here from the net geometry
    /// (weights block then bias block per layer, in evaluation order) — the
    /// exact layout [`Mlp::flat_weights`] produces.
    fn build(device: &wgpu::Device, mlp: &Mlp, shader_src: &str) -> Self {
        let dims = mlp.layer_dims();
        assert!(
            dims.len() <= MAX_LAYERS,
            "upscaler has {} layers, GPU port ceiling is {MAX_LAYERS}",
            dims.len()
        );
        let mut uniform = UpscaleUniform {
            dims: [0, 0, 0, 0],
            // info.y = total weight count (the fast shader's cooperative
            // threadgroup load bound); harmless/unused by the naive shader.
            info: [dims.len() as u32, mlp.flat_weights().len() as u32, 0, 0],
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

        let flat = mlp.flat_weights();
        assert_eq!(
            flat.len() as u32,
            offset,
            "flat weight length disagrees with computed layer offsets"
        );
        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upscaler weights"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VIII-3 upscaler"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<UpscaleUniform>() as u64),
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
            label: Some("upscaler layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, true),  // weights
                storage_entry(2, true),  // low radiance
                storage_entry(3, true),  // hi albedo
                storage_entry(4, true),  // hi normal
                storage_entry(5, true),  // hi depth
                storage_entry(6, false), // out
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upscaler pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("upscaler pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("upscale"),
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

    /// Pack one frame's buffers into the input storage buffers the shader
    /// reads: low radiance (low res) + hi albedo/normal/depth (target res).
    /// Panics on a length mismatch (a caller bug, like the CPU `upscale_image`).
    #[allow(clippy::too_many_arguments)]
    fn upload_inputs(
        &self,
        device: &wgpu::Device,
        low_radiance: &[Vec3],
        hi_albedo: &[Vec3],
        hi_normal: &[Vec3],
        hi_depth: &[f32],
        low_n: usize,
        target_n: usize,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
    ) {
        assert_eq!(low_radiance.len(), low_n);
        assert_eq!(hi_albedo.len(), target_n);
        assert_eq!(hi_normal.len(), target_n);
        assert_eq!(hi_depth.len(), target_n);

        let vec3_pack = |v: &[Vec3]| -> Vec<[f32; 4]> {
            v.iter().map(|c| [c.x, c.y, c.z, 0.0]).collect()
        };
        let low_p = vec3_pack(low_radiance);
        let albedo_p = vec3_pack(hi_albedo);
        let normal_p = vec3_pack(hi_normal);
        let depth_p: Vec<[f32; 4]> = hi_depth.iter().map(|&d| [d, 0.0, 0.0, 0.0]).collect();

        let mk = |label, data: &[[f32; 4]]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let low_buf = mk("upscaler low radiance", &low_p);
        let albedo_buf = mk("upscaler hi albedo", &albedo_p);
        let normal_buf = mk("upscaler hi normal", &normal_p);
        let depth_buf = mk("upscaler hi depth", &depth_p);
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upscaler out"),
            size: (target_n.max(1) * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        (low_buf, albedo_buf, normal_buf, depth_buf, out_buf)
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_group(
        &self,
        device: &wgpu::Device,
        uniform_buf: &wgpu::Buffer,
        low_buf: &wgpu::Buffer,
        albedo_buf: &wgpu::Buffer,
        normal_buf: &wgpu::Buffer,
        depth_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscaler bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: low_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: albedo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: normal_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: depth_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: out_buf.as_entire_binding() },
            ],
        })
    }

    fn make_uniform(&self, low_w: u32, low_h: u32, target_w: u32, target_h: u32) -> UpscaleUniform {
        let mut uniform = self.uniform;
        uniform.dims = [low_w, low_h, target_w, target_h];
        uniform
    }

    /// Upscale one current frame on the GPU and read the result back. The
    /// low-resolution radiance is upsampled to (target_w, target_h) guided by
    /// this frame's target-resolution AOVs; the returned image is the GPU
    /// transcription of [`crate::upscaler::upscale_image`].
    #[allow(clippy::too_many_arguments)]
    pub fn upscale(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        low_radiance: &[Vec3],
        low_w: u32,
        low_h: u32,
        hi_albedo: &[Vec3],
        hi_normal: &[Vec3],
        hi_depth: &[f32],
        target_w: u32,
        target_h: u32,
    ) -> Vec<Vec3> {
        let low_n = (low_w * low_h) as usize;
        let target_n = (target_w * target_h) as usize;
        let (low_buf, albedo_buf, normal_buf, depth_buf, out_buf) = self.upload_inputs(
            device, low_radiance, hi_albedo, hi_normal, hi_depth, low_n, target_n,
        );
        let uniform = self.make_uniform(low_w, low_h, target_w, target_h);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upscaler uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.bind_group(device, &uniform_buf, &low_buf, &albedo_buf, &normal_buf, &depth_buf, &out_buf);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("upscale encode"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("upscale pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(target_w.div_ceil(8), target_h.div_ceil(8), 1);
        }

        let bytes = (target_n * 16) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upscale readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, bytes);
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

    /// Time `repeats` GPU upscale dispatches via GPU timestamp queries,
    /// returning per-dispatch milliseconds. The device MUST have
    /// [`wgpu::Features::TIMESTAMP_QUERY`]; returns `None` otherwise so the
    /// caller can report the gap honestly. Only the compute pass is bracketed
    /// — upload/readback excluded, so this is the pass cost against the frame
    /// budget, not the round-trip.
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
        let (low_buf, albedo_buf, normal_buf, depth_buf, out_buf) = self.upload_inputs(
            device, low_radiance, hi_albedo, hi_normal, hi_depth, low_n, target_n,
        );
        let uniform = self.make_uniform(low_w, low_h, target_w, target_h);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upscaler uniform (timed)"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.bind_group(device, &uniform_buf, &low_buf, &albedo_buf, &normal_buf, &depth_buf, &out_buf);

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("upscale timestamps"),
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
                label: Some("upscale timed encode"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("upscale timed pass"),
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

/// A headless GPU device requesting `SHADER_F16` (for the fast upscaler) plus
/// `TIMESTAMP_QUERY` when available (for pass timing). Returns `None` when no
/// adapter is available OR the adapter lacks `SHADER_F16` (the fast port
/// cannot run on this host — the caller reports the gap honestly).
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
