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

use crate::rdirect::INPUT_FEATURES;

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
