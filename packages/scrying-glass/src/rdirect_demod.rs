//! NEURAL-LIVE N0 CUT 2 — GPU DEMOD driver. One compute dispatch that undoes
//! the net's log-demod on the GPU (net output MTLBuffer + native AOV albedo →
//! present accum), killing the N0.d present stage's GPU→CPU AOV readback + CPU
//! per-pixel demod + re-upload round-trip. Same math as the CPU
//! `undo_log_demod_px` (main.rs) — the pixels are UNCHANGED, only the path.
//!
//! Construct once; `encode` any number of frames. Owns only the pipeline and a
//! tiny reusable uniform buffer — zero per-frame allocation beyond the bind
//! group (handle-weight; the net output buffer is pooled, the AOV/present
//! buffers are pooled in `NetPresent`).

use bytemuck::{Pod, Zeroable};

pub const DEMOD_SHADER: &str = include_str!("rdirect_demod.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DemodUniform {
    n: u32,
    /// S12.5 AI DEBUG DOOR: 0 = presented (undo the albedo log-demod, the
    /// shipped final), 1 = belief (the net's RAW radiance `exp(dl)-1`, no
    /// albedo multiply — the accum-belief eye owed since n0e).
    mode: u32,
    _pad: [u32; 2],
}

/// A built GPU-demod compute pass.
pub struct DemodPass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
}

impl DemodPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rdirect demod"),
            source: wgpu::ShaderSource::Wgsl(DEMOD_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rdirect demod layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<DemodUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),  // net_out (read)
                storage_entry(2, true),  // aov (read)
                storage_entry(3, false), // present accum (read_write)
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rdirect demod pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rdirect demod pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("demod"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rdirect demod uniform"),
            size: std::mem::size_of::<DemodUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            uniform_buf,
        }
    }

    /// Encode one demod dispatch. `net_out` = the net's pooled output STORAGE
    /// buffer (`[n,3]`), `aov` = native AOV (2 cells/px), `present` = the
    /// surface present accum (`[n]` vec4). All current-frame.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        net_out: &wgpu::Buffer,
        aov: &wgpu::Buffer,
        present: &wgpu::Buffer,
        n: u32,
        belief: bool,
    ) {
        let uniform = DemodUniform { n, mode: belief as u32, _pad: [0; 2] };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rdirect demod bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: net_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: aov.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: present.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rdirect demod pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
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
