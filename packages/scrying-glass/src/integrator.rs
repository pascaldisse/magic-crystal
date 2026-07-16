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
    pub node_count: u32,
    pub tri_count: u32,
}

/// Bytes one accumulation cell occupies (vec4<f32>).
const ACCUM_CELL: u64 = 16;

impl Integrator {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, bvh: &Bvh) -> Self {
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

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrator compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                immediate_size: 0,
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integrator compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("integrate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("integrator blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("integrator blit pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("blit_vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        Self {
            compute_pipeline,
            blit_pipeline,
            compute_layout,
            blit_layout,
            uniform_buf,
            node_buf,
            tri_buf,
            node_count: bvh.nodes.len() as u32,
            tri_count: bvh.tris.len() as u32,
        }
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
) -> Vec<[f32; 4]> {
    let integrator = Integrator::new(device, wgpu::TextureFormat::Rgba8UnormSrgb, bvh);
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
