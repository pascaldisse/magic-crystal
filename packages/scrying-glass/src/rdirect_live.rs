//! NEURAL-LIVE — the ONE NET inside the live present path (STAGE N0 plumbing).
//!
//! THE DESIGN IS THE LAW (Architect, 07-18): the shipped present path is
//! trace → THE NET → screen and nothing else. This module is the net's live
//! embodiment: the R-Direct MLP (rdirect.rs, 23→5×64 ReLU→3, committed
//! weights) run as a BATCHED GEMM on the SAME Metal device wgpu drives,
//! reached through the wgpu-hal Metal backdoor (`Queue::as_hal` →
//! `MTLCommandQueue::device`). The batched-GEMM formulation is the measured
//! one (docs/perf/2026-07-18-rdirect-metal-tensor-spike.md: 4.47ms @ native
//! 960×640 on M1 Pro, ~94% fp32 roofline, ~63× over the per-thread WGSL).
//!
//! ── SCAFFOLD DISCIPLINE (dies at lane cutover) ────────────────────────────
//! During THIS lane the whole path is guarded by the `GAIA_NEURAL_LIVE` dev
//! flag (see `enabled()`), documented as construction scaffold. The merged,
//! cut-over state presents the net unconditionally — the flag and this guard
//! DIE with the lane. Until then the live window shows the one integrator's
//! young samples (truth, no stand-in) when the flag is off.
//!
//! ── STAGE STATUS (honest) ─────────────────────────────────────────────────
//! N0.a (DONE, this file): MPSGraph batched-GEMM forward built once from the
//!   weights blob; parity-gated vs the Rust CPU `Mlp::forward` on the exported
//!   fixed-pose feature buffer (GATE 1). Reaches the Metal device two ways:
//!   `from_system()` (own device+queue, for the offline parity ordeal) and
//!   `from_wgpu_queue()` (the wgpu device/queue — the live path).
//! N0.b (NEXT SHIFT, UNVERIFIED): zero-copy feature gather from the trace
//!   pass's wgpu textures/buffers into the graph's input, output texture wired
//!   into the present blit, buffer pooling (kill the 157MB churn), live frame
//!   budget measured on :8436. The `forward_cpu_roundtrip` path here is the
//!   CPU-staged bring-up; the shared-MTLBuffer path replaces it there.

#[cfg(target_os = "macos")]
pub use imp::RdirectLive;

/// The N0 construction scaffold flag. TRUE only when `GAIA_NEURAL_LIVE` is set
/// to a truthy value. Documented to DIE at lane cutover (the merged state
/// presents the net unconditionally — no flag).
pub fn enabled() -> bool {
    match std::env::var("GAIA_NEURAL_LIVE") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v.is_empty() || v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => false,
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use crate::rdirect::{deserialize_weights, Mlp, INPUT_FEATURES, OUTPUT_CHANNELS};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{AnyThread, Message};
    use objc2_foundation::{NSArray, NSData, NSNumber, NSString};
    use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
    use objc2_metal_performance_shaders::MPSDataType;
    use objc2_metal_performance_shaders_graph::{
        MPSGraph, MPSGraphDevice, MPSGraphTensor, MPSGraphTensorData,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    /// A single dense layer's Metal-resident weights, already transposed to the
    /// GEMM's second operand `[in, out]` (the CPU net stores `[out, in]`).
    struct GraphLayer {
        weight: Retained<MPSGraphTensor>,
        bias: Retained<MPSGraphTensor>,
        is_last: bool,
    }

    /// The live net: an MPSGraph batched-GEMM forward built once at construction
    /// from the committed weights, plus the Metal device/queue it runs on.
    pub struct RdirectLive {
        graph: Retained<MPSGraph>,
        mps_device: Retained<MPSGraphDevice>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        input: Retained<MPSGraphTensor>,
        output: Retained<MPSGraphTensor>,
        cpu_ref: Mlp,
        in_features: usize,
        out_channels: usize,
    }

    impl RdirectLive {
        /// Build the forward from a raw GAIARDR1 weights blob on an explicit
        /// Metal device + command queue.
        fn build(
            device: Retained<ProtocolObject<dyn MTLDevice>>,
            queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
            weights: &[u8],
        ) -> Result<Self, String> {
            let cpu_ref = deserialize_weights(weights)
                .ok_or_else(|| "rdirect_live: weights blob failed to parse".to_string())?;
            let dims = cpu_ref.layer_dims();
            if dims.first().map(|d| d.0 as usize) != Some(INPUT_FEATURES) {
                return Err(format!(
                    "rdirect_live: first layer in_dim {:?} != INPUT_FEATURES {}",
                    dims.first(),
                    INPUT_FEATURES
                ));
            }
            let flat = cpu_ref.flat_weights();

            // SAFETY: objc2 message sends to live Metal + MPSGraph objects.
            unsafe {
                let graph = MPSGraph::new();
                let mps_device = MPSGraphDevice::deviceWithMTLDevice(&device);

                // Layers, in blob order, transposing each weight to [in, out].
                let mut cursor = 0usize;
                let mut layers = Vec::with_capacity(dims.len());
                for (li, &(in_dim, out_dim)) in dims.iter().enumerate() {
                    let (in_dim, out_dim) = (in_dim as usize, out_dim as usize);
                    let w_flat = &flat[cursor..cursor + in_dim * out_dim];
                    cursor += in_dim * out_dim;
                    let b_flat = &flat[cursor..cursor + out_dim];
                    cursor += out_dim;

                    // Transpose [out, in] (row o*in + i) → [in, out] (row i*out + o).
                    let mut wt = vec![0f32; in_dim * out_dim];
                    for o in 0..out_dim {
                        for i in 0..in_dim {
                            wt[i * out_dim + o] = w_flat[o * in_dim + i];
                        }
                    }
                    let weight = constant_f32(&graph, &wt, &[in_dim, out_dim]);
                    let bias = constant_f32(&graph, b_flat, &[1, out_dim]);
                    layers.push(GraphLayer {
                        weight,
                        bias,
                        is_last: li == dims.len() - 1,
                    });
                }

                // input placeholder: [N, in_features]; N is dynamic (-1).
                let in_shape = shape(&[usize::MAX, INPUT_FEATURES]); // MAX → -1 sentinel
                let input = graph
                    .placeholderWithShape_dataType_name(
                        Some(&in_shape),
                        MPSDataType::Float32,
                        None,
                    );

                // Forward: Y = matmul(X, Wᵀ) + b, ReLU on hidden layers.
                let mut act = input.clone();
                for layer in &layers {
                    let mm = graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(
                        &act,
                        &layer.weight,
                        None,
                    );
                    let biased = graph.additionWithPrimaryTensor_secondaryTensor_name(
                        &mm,
                        &layer.bias,
                        None,
                    );
                    act = if layer.is_last {
                        biased
                    } else {
                        graph.reLUWithTensor_name(&biased, None)
                    };
                }
                let output = act;

                Ok(Self {
                    graph,
                    mps_device,
                    queue,
                    input,
                    output,
                    cpu_ref,
                    in_features: INPUT_FEATURES,
                    out_channels: OUTPUT_CHANNELS,
                })
            }
        }

        /// The offline ordeal path: create an own system Metal device + queue.
        /// Used by the N0 parity ordeal (no wgpu context needed).
        pub fn from_system(weights: &[u8]) -> Result<Self, String> {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "rdirect_live: no system Metal device".to_string())?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| "rdirect_live: newCommandQueue failed".to_string())?;
            Self::build(device, queue, weights)
        }

        /// THE LIVE PATH: reach the Metal device + queue wgpu itself drives,
        /// through the wgpu-hal Metal backdoor. Same device/queue as the trace
        /// — the net runs in the same command timeline.
        pub fn from_wgpu_queue(queue: &wgpu::Queue, weights: &[u8]) -> Result<Self, String> {
            // SAFETY: as_hal hands the live hal Queue; we retain the raw
            // MTLCommandQueue and derive its MTLDevice. Both outlive `self`.
            let (device, mtl_queue) = unsafe {
                queue
                    .as_hal::<wgpu::hal::api::Metal>()
                    .map(|hal_queue| {
                        let raw = hal_queue.as_raw();
                        let mtl_queue: Retained<ProtocolObject<dyn MTLCommandQueue>> =
                            raw.retain();
                        let device = mtl_queue.device();
                        (device, mtl_queue)
                    })
                    .ok_or_else(|| {
                        "rdirect_live: wgpu is not on the Metal backend".to_string()
                    })?
            };
            Self::build(device, mtl_queue, weights)
        }

        pub fn in_features(&self) -> usize {
            self.in_features
        }
        pub fn out_channels(&self) -> usize {
            self.out_channels
        }
        pub fn cpu_ref(&self) -> &Mlp {
            &self.cpu_ref
        }

        /// N0.a CPU-staged forward: features in (row-major `[N, in_features]`),
        /// demod-log radiance out (`[N, out_channels]`). This stages through
        /// host memory (NSData feed → run → readBytes) — the honest bring-up
        /// and the parity harness. N0.b replaces the staging with shared
        /// MTLBuffers gathered straight from the trace pass (zero-copy).
        pub fn forward_cpu_roundtrip(&self, features: &[f32]) -> Result<Vec<f32>, String> {
            let n = features.len() / self.in_features;
            if n * self.in_features != features.len() {
                return Err(format!(
                    "rdirect_live: feature len {} not a multiple of {}",
                    features.len(),
                    self.in_features
                ));
            }
            // SAFETY: objc2 message sends; buffers sized exactly to the shapes.
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    features.as_ptr() as *const u8,
                    std::mem::size_of_val(features),
                );
                let data = NSData::with_bytes(bytes);
                let in_shape = shape(&[n, self.in_features]);
                let input_data = MPSGraphTensorData::initWithDevice_data_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    &self.mps_device,
                    &data,
                    &in_shape,
                    MPSDataType::Float32,
                );

                let feeds = objc2_foundation::NSDictionary::<
                    MPSGraphTensor,
                    MPSGraphTensorData,
                >::from_slices(&[&*self.input], &[&*input_data]);
                let targets = NSArray::from_slice(&[&*self.output]);

                let results = self
                    .graph
                    .runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
                        &self.queue,
                        &feeds,
                        &targets,
                        None,
                    );
                let out_td = results
                    .objectForKey(&self.output)
                    .ok_or_else(|| "rdirect_live: no result for output tensor".to_string())?;

                let ndarray = out_td.mpsndarray();
                let mut out = vec![0f32; n * self.out_channels];
                ndarray.readBytes_strideBytes(
                    NonNull::new(out.as_mut_ptr() as *mut c_void).unwrap(),
                    std::ptr::null_mut(),
                );
                Ok(out)
            }
        }
    }

    /// Build an `MPSShape` (NSArray<NSNumber>). `usize::MAX` in a dim encodes
    /// the dynamic-batch sentinel (-1) MPSGraph expects for placeholders.
    fn shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
        let numbers: Vec<Retained<NSNumber>> = dims
            .iter()
            .map(|&d| {
                if d == usize::MAX {
                    NSNumber::new_isize(-1)
                } else {
                    NSNumber::new_isize(d as isize)
                }
            })
            .collect();
        let refs: Vec<&NSNumber> = numbers.iter().map(|n| n.as_ref()).collect();
        NSArray::from_slice(&refs)
    }

    /// A graph constant tensor from an f32 slice with an explicit shape.
    fn constant_f32(
        graph: &MPSGraph,
        values: &[f32],
        dims: &[usize],
    ) -> Retained<MPSGraphTensor> {
        // SAFETY: bytes sized to values; shape product == values.len().
        unsafe {
            let bytes = std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                std::mem::size_of_val(values),
            );
            let data = NSData::with_bytes(bytes);
            let sh = shape(dims);
            graph.constantWithData_shape_dataType(&data, &sh, MPSDataType::Float32)
        }
    }

    // Silence unused-import lint noise on NSString (kept for op naming later).
    #[allow(unused)]
    fn _touch(_: &NSString) {}
}
