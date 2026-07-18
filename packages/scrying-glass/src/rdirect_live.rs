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
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLResourceOptions,
    };
    use objc2_metal_performance_shaders::{MPSCommandBuffer, MPSDataType};
    use std::cell::Cell;
    use objc2_metal_performance_shaders_graph::{
        MPSGraph, MPSGraphDevice, MPSGraphExecutable,
        MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
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

    /// The zero-copy pool (N0.b): the feature input and net output as MTLBuffers
    /// allocated ONCE, sized to a fixed `max_pixels` ceiling. `feature_buf` is
    /// the SAME MTLBuffer wrapped as a wgpu STORAGE buffer, so the gather
    /// compute pass writes it and the MPSGraph forward reads it with no copy and
    /// no per-frame allocation (this is where the spike's 157 MB/frame churn
    /// dies). Only present on the live wgpu path (`from_wgpu_queue`).
    struct SharedPool {
        /// wgpu view of `feature_mtl` (STORAGE): the gather's destination.
        feature_buf: wgpu::Buffer,
        /// Same MTLBuffer as `feature_buf`, fed to the graph zero-copy.
        feature_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
        /// The graph's output MTLBuffer (Shared storage → CPU-readable AND the
        /// same buffer wrapped as `out_buf` for the GPU demod, CUT 2).
        out_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
        /// CUT 2: `out_mtl` wrapped as a wgpu STORAGE buffer so the demod
        /// compute pass reads the net output on the GPU (no CPU round-trip).
        out_buf: wgpu::Buffer,
        max_pixels: usize,
        // ── CUT 1: POOL THE NET — everything below is built ONCE here (the
        // metal4-probe pattern: a compiled MPSGraphExecutable + pre-allocated
        // MPSGraphTensorData on the persistent MTLBuffers). Per-frame
        // `forward` then just runs the executable — no graph rebuild, no
        // tensor-data / NSDictionary allocation, which is what took the net
        // stage from the probe's 4.47 ms down to 27 ms in the N0.d run.
        /// Compiled once for the FIXED shape `[max_pixels, in_features]`.
        executable: Retained<MPSGraphExecutable>,
        /// Input feed array `[in_td]` (in_td wraps `feature_mtl` zero-copy).
        inputs: Retained<NSArray<MPSGraphTensorData>>,
        /// Result array `[out_td]` (out_td wraps `out_mtl` zero-copy).
        results: Retained<NSArray<MPSGraphTensorData>>,
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
        pool: Option<SharedPool>,
        /// S3 instrument: GPU-only time of the last forward (MTLCommandBuffer
        /// GPUEndTime − GPUStartTime, ms). The `runWithMTLCommandQueue` API hid
        /// this; the encode path below owns the command buffer so it reads it.
        last_gpu_ms: Cell<f64>,
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
                    pool: None,
                    last_gpu_ms: Cell::new(0.0),
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
        ///
        /// `max_pixels` sizes the zero-copy pool (feature input + net output
        /// MTLBuffers, allocated once): the largest target-pixel count any frame
        /// will forward. `feature_buffer()` returns the gather's destination.
        pub fn from_wgpu_queue(
            wgpu_device: &wgpu::Device,
            queue: &wgpu::Queue,
            weights: &[u8],
            max_pixels: usize,
        ) -> Result<Self, String> {
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
            let mut me = Self::build(device.clone(), mtl_queue, weights)?;
            me.attach_pool(wgpu_device, &device, max_pixels)?;
            Ok(me)
        }

        /// Allocate the once-only zero-copy pool on `mtl_device` and bridge the
        /// feature MTLBuffer into a wgpu STORAGE buffer (so the gather writes it).
        fn attach_pool(
            &mut self,
            wgpu_device: &wgpu::Device,
            mtl_device: &Retained<ProtocolObject<dyn MTLDevice>>,
            max_pixels: usize,
        ) -> Result<(), String> {
            let in_bytes = max_pixels * self.in_features * std::mem::size_of::<f32>();
            let out_bytes = max_pixels * self.out_channels * std::mem::size_of::<f32>();
            // SAFETY: objc2 message sends + the wgpu-hal Metal buffer bridge; the
            // MTLBuffer we clone into wgpu outlives the wgpu buffer (both held in
            // `SharedPool`).
            unsafe {
                let feature_mtl = mtl_device
                    .newBufferWithLength_options(in_bytes, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| "rdirect_live: feature MTLBuffer alloc failed".to_string())?;
                let out_mtl = mtl_device
                    .newBufferWithLength_options(out_bytes, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| "rdirect_live: output MTLBuffer alloc failed".to_string())?;
                let hal_buf = wgpu::hal::metal::Device::buffer_from_raw(
                    feature_mtl.clone(),
                    in_bytes as u64,
                );
                let feature_buf = wgpu_device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                    hal_buf,
                    &wgpu::BufferDescriptor {
                        label: Some("rdirect feature (shared MTLBuffer)"),
                        size: in_bytes as u64,
                        // COPY_SRC so ordeals can read the gather output back for
                        // parity (the live path never copies it — the graph reads
                        // the same MTLBuffer in place).
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                );
                // CUT 2: wrap the SAME output MTLBuffer as a wgpu STORAGE buffer
                // so the GPU demod pass reads the net output in place (no CPU
                // readback). COPY_SRC lets `forward_shared` still map it for the
                // ordeal/CPU path.
                let out_hal = wgpu::hal::metal::Device::buffer_from_raw(
                    out_mtl.clone(),
                    out_bytes as u64,
                );
                let out_buf = wgpu_device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                    out_hal,
                    &wgpu::BufferDescriptor {
                        label: Some("rdirect net output (shared MTLBuffer)"),
                        size: out_bytes as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                );

                // ── CUT 1: compile the executable ONCE for the fixed shape and
                // pre-build the zero-copy feed/result tensor-data arrays. ──
                let in_shape = shape(&[max_pixels, self.in_features]);
                let out_shape = shape(&[max_pixels, self.out_channels]);
                let in_shaped = MPSGraphShapedType::initWithShape_dataType(
                    MPSGraphShapedType::alloc(),
                    Some(&in_shape),
                    MPSDataType::Float32,
                );
                let feeds = objc2_foundation::NSDictionary::<
                    MPSGraphTensor,
                    MPSGraphShapedType,
                >::from_slices(&[&*self.input], &[&*in_shaped]);
                let targets = NSArray::from_slice(&[&*self.output]);
                let executable = self
                    .graph
                    .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                        Some(&self.mps_device),
                        &feeds,
                        &targets,
                        None,
                        None,
                    );
                let in_td = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    &feature_mtl,
                    &in_shape,
                    MPSDataType::Float32,
                );
                let out_td = MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    &out_mtl,
                    &out_shape,
                    MPSDataType::Float32,
                );
                let inputs = NSArray::from_slice(&[&*in_td]);
                let results = NSArray::from_slice(&[&*out_td]);
                // NOTE: the forward now runs through `encodeToCommandBuffer` on a
                // command buffer WE own (see `run_executable`) — we commit + wait
                // manually so GPUStartTime/GPUEndTime are readable (S3). The old
                // `MPSGraphExecutableExecutionDescriptor` / GAIA_NATIVE_NET_ASYNC
                // block-toggle is gone; S2 one-frame pipelining, if adopted, drops
                // the `root.waitUntilCompleted()` in `run_executable` instead.

                self.pool = Some(SharedPool {
                    feature_buf,
                    feature_mtl,
                    out_mtl,
                    out_buf,
                    max_pixels,
                    executable,
                    inputs,
                    results,
                });
            }
            Ok(())
        }

        /// The gather's destination STORAGE buffer (the shared feature MTLBuffer
        /// wrapped for wgpu). `None` unless built via `from_wgpu_queue`.
        pub fn feature_buffer(&self) -> Option<&wgpu::Buffer> {
            self.pool.as_ref().map(|p| &p.feature_buf)
        }

        /// CUT 2: the net's OUTPUT MTLBuffer wrapped as a wgpu STORAGE buffer
        /// (`[N, out_channels]` row-major, demod-log radiance). The GPU demod
        /// pass reads this in place — no CPU readback. `None` unless built via
        /// `from_wgpu_queue`.
        pub fn output_buffer(&self) -> Option<&wgpu::Buffer> {
            self.pool.as_ref().map(|p| &p.out_buf)
        }

        /// CUT 1 CORE: run the pooled, compiled executable over the pooled
        /// buffers. Zero per-call allocation — the executable, feed/result
        /// tensor-data arrays and execution descriptor were all built once in
        /// `attach_pool`. Blocks until the GPU forward completes
        /// (`waitUntilCompleted`), so on return `out_mtl` / `out_buf` hold the
        /// frame's radiance. `n` must equal the pooled `max_pixels` (the
        /// executable is specialized to that fixed shape).
        fn run_executable(&self, n: usize) -> Result<&SharedPool, String> {
            let pool = self.pool.as_ref().ok_or_else(|| {
                "rdirect_live: forward needs the shared pool (from_wgpu_queue)".to_string()
            })?;
            if n != pool.max_pixels {
                return Err(format!(
                    "rdirect_live: n={n} != pooled max_pixels={} (the executable is \
                     compiled for the fixed boot shape)",
                    pool.max_pixels
                ));
            }
            // SAFETY: objc2 message send; the executable, inputs/results arrays
            // and descriptor are the pool's (fixed shapes over the persistent
            // MTLBuffers), and the run waits for completion before returning.
            // The run is wrapped in an autorelease pool so MPSGraph's per-call
            // intermediate NDArrays (the ~157MB hidden activations) are drained
            // this frame rather than piling into the thread's outer pool.
            //
            // S3: we OWN the command buffer (metal4-probe pattern) instead of
            // `runWithMTLCommandQueue` — that API allocs+commits+waits its own
            // buffer and gives back only wall time. Here a base MTLCommandBuffer
            // wrapped as MPSCommandBuffer lets us read GPUStartTime/GPUEndTime
            // (GPU-only ms) after the wait, separating GPU work from CPU encode.
            objc2::rc::autoreleasepool(|_| unsafe {
                let base = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| "rdirect_live: commandBuffer alloc failed".to_string())?;
                let mps_cmd = MPSCommandBuffer::commandBufferWithCommandBuffer(&base);
                let _ = pool
                    .executable
                    .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                        &mps_cmd,
                        &pool.inputs,
                        Some(&pool.results),
                        // descriptor None: we own commit + wait below, so the
                        // executable must not internally commit/wait (that hides
                        // GPU timestamps and races our root wait).
                        None,
                    );
                // encode may commitAndContinue; commit the live root and wait on
                // it. `base` is the first (and, for this small graph, only)
                // buffer — its GPU timestamps bound the forward's GPU work.
                let root = mps_cmd.rootCommandBuffer();
                mps_cmd.commit();
                root.waitUntilCompleted();
                let gpu_ms = (base.GPUEndTime() - base.GPUStartTime()) * 1000.0;
                self.last_gpu_ms.set(gpu_ms);
                Ok::<(), String>(())
            })?;
            Ok(pool)
        }

        /// CUT 1 + CUT 2 live path: run the pooled executable and leave the
        /// result ON the GPU in `output_buffer()`. No CPU readback, no output
        /// `Vec` allocation — the demod compute pass consumes `out_buf`
        /// directly.
        pub fn forward_shared_gpu(&self, n: usize) -> Result<(), String> {
            self.run_executable(n)?;
            Ok(())
        }

        /// S3 instrument: GPU-only ms of the last `forward_shared*` call
        /// (MTLCommandBuffer GPU timestamps). Compare against the wall time the
        /// caller measures around the same call to split GPU work from CPU
        /// encode/readback — the metal4-probe's 4.47ms is this number.
        pub fn last_gpu_ms(&self) -> f64 {
            self.last_gpu_ms.get()
        }

        /// Ceiling passed at construction (max target pixels per forward).
        pub fn max_pixels(&self) -> usize {
            self.pool.as_ref().map(|p| p.max_pixels).unwrap_or(0)
        }

        /// N0.b ZERO-COPY forward: read the `n`×23 features already written into
        /// the pooled feature MTLBuffer (by the gather pass — caller must submit
        /// & complete it first) and run the batched GEMM into the pooled output
        /// MTLBuffer, returning `[n, out_channels]` demod-log radiance. No NSData
        /// staging, no per-call allocation — both buffers are the pool's.
        pub fn forward_shared(&self, n: usize) -> Result<Vec<f32>, String> {
            // CUT 1: run the pooled compiled executable (blocks until done),
            // then read the pooled Shared-storage output back to a Vec. The
            // ordeal/example CPU path keeps this Vec return; the live present
            // uses `forward_shared_gpu` (no readback) + the GPU demod pass.
            let pool = self.run_executable(n)?;
            // SAFETY: `out_mtl` is Shared storage, sized to max_pixels ≥ n, and
            // `run_executable` waited for the GPU forward to complete.
            unsafe {
                let ptr = pool.out_mtl.contents().as_ptr() as *const f32;
                let out = std::slice::from_raw_parts(ptr, n * self.out_channels).to_vec();
                Ok(out)
            }
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
