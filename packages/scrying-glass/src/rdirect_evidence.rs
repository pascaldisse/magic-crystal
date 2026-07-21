//! V7-LIVE LANE STAGE 3 — evidence-clamp GPU driver (`rdirect_evidence.wgsl`).
//! Three compute passes, each a bind group over caller-owned pooled buffers
//! (no per-frame allocation beyond the bind group — same house pattern as
//! `FeatureGather`/`DemodPass`). Ported from the CPU reference in
//! `rdirect.rs` (`EvidenceAccum`, `local_max_3x3`, `clamp_evidence_lin`,
//! commit c8b9ba6) — see the WGSL file's module doc for the exact mapping.

use bytemuck::{Pod, Zeroable};

pub const EVIDENCE_SHADER: &str = include_str!("rdirect_evidence.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Dims4 {
    dims: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClampU {
    params: [f32; 4], // (tw, th, count, gamma)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackU {
    n: [u32; 4], // n.x used, rest padding
}

pub struct EvidenceClamp {
    accumulate_pipeline: wgpu::ComputePipeline,
    accumulate_layout: wgpu::BindGroupLayout,
    accumulate_uniform: wgpu::Buffer,

    clamp_pipeline: wgpu::ComputePipeline,
    clamp_layout: wgpu::BindGroupLayout,
    clamp_uniform: wgpu::Buffer,

    pack_pipeline: wgpu::ComputePipeline,
    pack_layout: wgpu::BindGroupLayout,
    pack_uniform: wgpu::Buffer,
}

impl EvidenceClamp {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rdirect evidence clamp"),
            source: wgpu::ShaderSource::Wgsl(EVIDENCE_SHADER.into()),
        });

        let accumulate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("evidence accumulate layout"),
            entries: &[
                uniform_entry(0, std::mem::size_of::<Dims4>() as u64),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let accumulate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("evidence accumulate pipeline layout"),
            bind_group_layouts: &[Some(&accumulate_layout)],
            immediate_size: 0,
        });
        let accumulate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evidence accumulate pipeline"),
            layout: Some(&accumulate_pl),
            module: &module,
            entry_point: Some("evidence_accumulate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let accumulate_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evidence accumulate uniform"),
            size: std::mem::size_of::<Dims4>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clamp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("evidence clamp layout"),
            entries: &[
                uniform_entry(3, std::mem::size_of::<ClampU>() as u64),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });
        let clamp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("evidence clamp pipeline layout"),
            bind_group_layouts: &[Some(&clamp_layout)],
            immediate_size: 0,
        });
        let clamp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evidence clamp pipeline"),
            layout: Some(&clamp_pl),
            module: &module,
            entry_point: Some("evidence_clamp_present"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let clamp_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evidence clamp uniform"),
            size: std::mem::size_of::<ClampU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("evidence pack layout"),
            entries: &[
                uniform_entry(6, std::mem::size_of::<PackU>() as u64),
                storage_entry(7, true),
                storage_entry(8, false),
            ],
        });
        let pack_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("evidence pack pipeline layout"),
            bind_group_layouts: &[Some(&pack_layout)],
            immediate_size: 0,
        });
        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evidence pack pipeline"),
            layout: Some(&pack_pl),
            module: &module,
            entry_point: Some("pack_out_dl3to4"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pack_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evidence pack uniform"),
            size: std::mem::size_of::<PackU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            accumulate_pipeline,
            accumulate_layout,
            accumulate_uniform,
            clamp_pipeline,
            clamp_layout,
            clamp_uniform,
            pack_pipeline,
            pack_layout,
            pack_uniform,
        }
    }

    /// Native-res persistent sum buffer bytes for `n` pixels (one vec4/px).
    pub fn sum_bytes(n: usize) -> u64 {
        (n as u64) * 16
    }

    /// Fold this frame's low-res `accum_ed` (E/D split radiance) into the
    /// persistent native-res `evidence_sum` buffer (add, not overwrite).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_accumulate(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        accum_ed: &wgpu::Buffer,
        evidence_sum: &wgpu::Buffer,
        low_w: u32,
        low_h: u32,
        target_w: u32,
        target_h: u32,
    ) {
        let dims = Dims4 { dims: [low_w, low_h, target_w, target_h] };
        queue.write_buffer(&self.accumulate_uniform, 0, bytemuck::bytes_of(&dims));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("evidence accumulate bind"),
            layout: &self.accumulate_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.accumulate_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum_ed.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: evidence_sum.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evidence accumulate pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.accumulate_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(target_w.div_ceil(8), target_h.div_ceil(8), 1);
    }

    /// Clamp `present` in place: `present = min(present, gamma/count *
    /// local_max_3x3(evidence_sum))`. `count` = frames folded into
    /// `evidence_sum` so far (>=1).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_clamp(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        evidence_sum: &wgpu::Buffer,
        present: &wgpu::Buffer,
        target_w: u32,
        target_h: u32,
        count: u32,
        gamma: f32,
    ) {
        let u = ClampU { params: [target_w as f32, target_h as f32, count as f32, gamma] };
        queue.write_buffer(&self.clamp_uniform, 0, bytemuck::bytes_of(&u));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("evidence clamp bind"),
            layout: &self.clamp_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 3, resource: self.clamp_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: evidence_sum.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: present.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evidence clamp pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.clamp_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(target_w.div_ceil(8), target_h.div_ceil(8), 1);
    }

    /// Repack `src3` (tight `[n,3]` demod-log net output) into `dst4`
    /// (vec4-per-pixel, `HistoryBuffers::prev_out_dl` layout).
    pub fn encode_pack(
        &self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src3: &wgpu::Buffer,
        dst4: &wgpu::Buffer,
        n: u32,
    ) {
        let u = PackU { n: [n, 0, 0, 0] };
        queue.write_buffer(&self.pack_uniform, 0, bytemuck::bytes_of(&u));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("evidence pack bind"),
            layout: &self.pack_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 6, resource: self.pack_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: src3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: dst4.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("evidence pack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
}

fn uniform_entry(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(min_size),
        },
        count: None,
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
