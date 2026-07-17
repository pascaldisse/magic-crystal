//! RITE VIII-2 — THE DREAM AT SPEED: the GPU driver for the compute port of
//! the VIII-1 CPU reference denoiser. Uploads the SAME hash-pinned weights
//! (loaded via [`crate::denoiser::deserialize_weights`], never re-derived),
//! the current-frame input buffers (noisy radiance, albedo, normal, depth),
//! dispatches `denoiser.wgsl`, and reads the denoised image back. A plain
//! compute MLP per pixel — wgpu has no tensor surface (RENDER.md §8).
//!
//! THE BAN: this module takes current-frame buffers only. Its name matches
//! the `denoiser*.rs` grep-gate glob (viii0_ordeals), so it is scanned whole
//! for the forbidden cross-pass vocabulary; the shader carries its own
//! `// BAN-SCOPED` marker and is scanned by the VIII-2 ordeals. Nothing here
//! indexes, stores, or reads any earlier pass's output.
//!
//! BAN-SCOPED

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::denoiser::Mlp;

/// Fixed ceilings mirrored from `denoiser.wgsl` (`MAX_LAYERS`, `MAX_WIDTH`).
/// The shipped net (4 hidden layers, width 32) fits well under both; the live
/// geometry is passed as data, never assumed.
pub const MAX_LAYERS: usize = 16;
pub const MAX_WIDTH: u32 = 64;

/// The compute uniform. Layout matches the WGSL `DenoiseU` exactly (all
/// vec4-aligned): `dims` = (width, height, layer_count, _pad); each
/// `layers[l]` = (in_dim, out_dim, weights_offset, bias_offset).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenoiseUniform {
    dims: [u32; 4],
    layers: [[u32; 4]; MAX_LAYERS],
}

pub const DENOISER_SHADER: &str = include_str!("denoiser.wgsl");

/// A GPU denoiser bound to one net's weights. Build once (weights upload +
/// pipeline), then [`Self::denoise`] any number of current frames.
pub struct GpuDenoiser {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: DenoiseUniform,
    weights_buf: wgpu::Buffer,
}

impl GpuDenoiser {
    /// Build the pipeline and upload `mlp`'s flat weights. The per-layer
    /// offsets into the flat buffer are computed here from the net geometry
    /// (weights block then bias block per layer, in evaluation order) — the
    /// exact layout [`Mlp::flat_weights`] produces.
    pub fn new(device: &wgpu::Device, mlp: &Mlp) -> Self {
        let dims = mlp.layer_dims();
        assert!(
            dims.len() <= MAX_LAYERS,
            "denoiser has {} layers, GPU port ceiling is {MAX_LAYERS}",
            dims.len()
        );
        let mut uniform = DenoiseUniform {
            dims: [0, 0, dims.len() as u32, 0],
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
            label: Some("denoiser weights"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VIII-2 denoiser"),
            source: wgpu::ShaderSource::Wgsl(DENOISER_SHADER.into()),
        });

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<DenoiseUniform>() as u64
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
            label: Some("denoiser layout"),
            entries: &[
                uniform_entry,
                storage_entry(1, true),  // weights
                storage_entry(2, true),  // noisy
                storage_entry(3, true),  // albedo
                storage_entry(4, true),  // normal
                storage_entry(5, false), // out
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("denoiser pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("denoiser pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("denoise"),
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

    /// Pack one frame's current-frame buffers into the three vec4/pixel input
    /// storage buffers the shader reads. Panics on a length mismatch (a caller
    /// bug, like the CPU `denoise_image`).
    fn upload_inputs(
        &self,
        device: &wgpu::Device,
        noisy: &[Vec3],
        albedo: &[Vec3],
        normal: &[Vec3],
        depth: &[f32],
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        usize,
    ) {
        let n = noisy.len();
        assert_eq!(albedo.len(), n);
        assert_eq!(normal.len(), n);
        assert_eq!(depth.len(), n);

        let mut noisy_p = Vec::with_capacity(n);
        let mut albedo_p = Vec::with_capacity(n);
        let mut normal_p = Vec::with_capacity(n);
        for i in 0..n {
            noisy_p.push([noisy[i].x, noisy[i].y, noisy[i].z, depth[i]]);
            albedo_p.push([albedo[i].x, albedo[i].y, albedo[i].z, 0.0f32]);
            normal_p.push([normal[i].x, normal[i].y, normal[i].z, 0.0f32]);
        }
        let mk = |label, data: &[[f32; 4]]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let noisy_buf = mk("denoiser noisy", &noisy_p);
        let albedo_buf = mk("denoiser albedo", &albedo_p);
        let normal_buf = mk("denoiser normal", &normal_p);
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("denoiser out"),
            size: (n.max(1) * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        (noisy_buf, albedo_buf, normal_buf, out_buf, n)
    }

    fn bind_group(
        &self,
        device: &wgpu::Device,
        uniform_buf: &wgpu::Buffer,
        noisy_buf: &wgpu::Buffer,
        albedo_buf: &wgpu::Buffer,
        normal_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("denoiser bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: noisy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: albedo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: normal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Denoise one current frame on the GPU and read the result back. All
    /// inputs are this frame's buffers; the returned image is the per-pixel
    /// MLP output, the GPU transcription of [`crate::denoiser::denoise_image`].
    pub fn denoise(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        noisy: &[Vec3],
        albedo: &[Vec3],
        normal: &[Vec3],
        depth: &[f32],
        width: u32,
        height: u32,
    ) -> Vec<Vec3> {
        let (noisy_buf, albedo_buf, normal_buf, out_buf, n) =
            self.upload_inputs(device, noisy, albedo, normal, depth);
        assert_eq!(n as u32, width * height, "pixel count != width*height");

        let mut uniform = self.uniform;
        uniform.dims[0] = width;
        uniform.dims[1] = height;
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("denoiser uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.bind_group(
            device,
            &uniform_buf,
            &noisy_buf,
            &albedo_buf,
            &normal_buf,
            &out_buf,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("denoise encode"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("denoise pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        let bytes = (n * 16) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("denoise readback"),
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

    /// Time `repeats` GPU denoise dispatches of one current frame via GPU
    /// timestamp queries, returning per-dispatch milliseconds (one entry per
    /// repeat). The device MUST have been created with
    /// [`wgpu::Features::TIMESTAMP_QUERY`]; returns `None` if it was not, so
    /// the caller can fall back or report the gap honestly. Only the compute
    /// pass is bracketed — upload/readback are excluded, so this is the pass
    /// cost against the frame budget, not the round-trip.
    pub fn time_dispatches_ms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        noisy: &[Vec3],
        albedo: &[Vec3],
        normal: &[Vec3],
        depth: &[f32],
        width: u32,
        height: u32,
        repeats: u32,
    ) -> Option<Vec<f64>> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let period_ns = queue.get_timestamp_period() as f64; // ns per tick
        let (noisy_buf, albedo_buf, normal_buf, out_buf, _n) =
            self.upload_inputs(device, noisy, albedo, normal, depth);
        let mut uniform = self.uniform;
        uniform.dims[0] = width;
        uniform.dims[1] = height;
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("denoiser uniform (timed)"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.bind_group(
            device,
            &uniform_buf,
            &noisy_buf,
            &albedo_buf,
            &normal_buf,
            &out_buf,
        );

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("denoise timestamps"),
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
                label: Some("denoise timed encode"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("denoise timed pass"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
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
            out.push(delta * period_ns / 1.0e6); // ns -> ms
        }
        Some(out)
    }
}

/// A headless GPU device that requests `TIMESTAMP_QUERY` when the adapter
/// supports it (so [`GpuDenoiser::time_dispatches_ms`] can measure the pass);
/// falls back to a plain device otherwise. Returns `None` when no adapter is
/// available (the ordeal then cannot run on this host).
pub fn headless_device_timed() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    let wanted = wgpu::Features::TIMESTAMP_QUERY;
    let features = adapter.features() & wanted;
    let mut desc = wgpu::DeviceDescriptor::default();
    desc.required_features = features;
    pollster::block_on(adapter.request_device(&desc)).ok()
}
