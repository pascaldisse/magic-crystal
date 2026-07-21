// NEURAL-LIVE N0 CUT 2 — GPU DEMOD. Replaces the per-frame GPU→CPU AOV
// readback + CPU undo-log-demod + re-upload round-trip (the N0.d present
// stage's 6.6ms scaffold stall) with one compute dispatch on the GPU.
//
// Reads the net's demod-log radiance output (row-major [N,3], the pooled
// MPSGraph output MTLBuffer wrapped for wgpu) plus the native AOV G-buffer
// (albedo lives in aov[2*px+0].xyz), undoes the log-demod, and writes the
// linear rgb straight into the present accum (vec4, w=1) the blit resolves.
//
// undo_log_demod_px must stay BIT-IDENTICAL to the CPU reference in main.rs
// (`undo_log_demod_px`) / rdirect.rs — same pixels as N0.d, no drift.

const ALBEDO_DEMOD_EPS: f32 = 1e-3;

struct DemodU {
    n: u32,          // pixel count (target_w * target_h)
    mode: u32,       // S12.5: 0 = presented (undo albedo demod), 1 = belief (raw net radiance)
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: DemodU;
// Net output: row-major [N, 3] f32 (demod-log radiance).
@group(0) @binding(1) var<storage, read> net_out: array<f32>;
// Native AOV: 2 vec4<f32> cells/px; cell 0 .xyz = albedo.
@group(0) @binding(2) var<storage, read> aov: array<vec4<f32>>;
// Present accum: 1 vec4<f32>/px (linear rgb, w = 1) — blit resolves it 1:1.
@group(0) @binding(3) var<storage, read_write> present: array<vec4<f32>>;

@compute @workgroup_size(64)
fn demod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }

    let albedo = aov[2u * i + 0u].xyz;
    let dl = vec3<f32>(net_out[3u * i + 0u], net_out[3u * i + 1u], net_out[3u * i + 2u]);

    var divisor: vec3<f32>;
    if (dot(albedo, albedo) > 1e-8) {
        divisor = albedo + vec3<f32>(ALBEDO_DEMOD_EPS);
    } else {
        divisor = vec3<f32>(1.0);
    }
    let e = max(exp(dl) - vec3<f32>(1.0), vec3<f32>(0.0));
    // S12.5 belief eye (mode 1): the net's RAW radiance, no albedo multiply.
    let lin = select(e * divisor, e, u.mode == 1u);
    present[i] = vec4<f32>(lin, 1.0);
}
