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
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
        MTLDevice, MTLLibrary, MTLResourceOptions, MTLSharedEvent, MTLSize,
    };
    use objc2_metal_performance_shaders::{
        MPSCommandBuffer, MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
    };
    use std::cell::{Cell, RefCell};
    use objc2_metal_performance_shaders_graph::{
        MPSGraph, MPSGraphDevice, MPSGraphExecutable,
        MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorData,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::Arc;
    use std::thread::JoinHandle;

    /// S11 instrument gate: GAIA_NATIVE_NET_TRACE prints the gather→net fence
    /// value pair + command-buffer terminal status, to SEE the wedge.
    fn net_trace() -> bool {
        matches!(std::env::var("GAIA_NATIVE_NET_TRACE"), Ok(v) if !v.is_empty() && v != "0" && v != "false")
    }

    /// A single dense layer's Metal-resident weights, already transposed to the
    /// GEMM's second operand `[in, out]` (the CPU net stores `[out, in]`).
    struct GraphLayer {
        weight: Retained<MPSGraphTensor>,
        bias: Retained<MPSGraphTensor>,
        is_last: bool,
    }

    /// S5 — the RAW-KERNEL forward (kill the MPSGraph per-frame encode wall).
    ///
    /// n0e verdict: MPSGraph's per-frame `encodeToCommandBuffer` +
    /// `waitUntilCompleted` cost ~22 ms of CPU even though the GPU forward is
    /// ~4.5 ms. This is that same 23→5×64→3 net expressed as a chain of
    /// classic-MPS `MPSMatrixMultiplication` GEMMs (one encode per layer,
    /// µs-class CPU cost) interleaved with a hand-written `bias+ReLU` compute
    /// kernel — all on ONE MTLCommandBuffer we own. Same weights, same math as
    /// the CPU `Mlp::forward`; bit-parity gated by the ordeal.
    ///
    /// Each layer: `C[rows,out] = A[rows,in] · Wᵀ` via MPSMatrixMultiplication
    /// with `transposeRight=true` over the CPU-native `[out,in]` weight buffer
    /// (no CPU transpose), `beta=0`; then the compute kernel adds the bias and
    /// (hidden layers) applies ReLU in place. Ping-pong activation buffers.
    /// Built ONCE (kernels, MPSMatrix wrappers, weight/bias/act MTLBuffers, the
    /// pipeline) so per-frame `run` allocates nothing.
    struct MatmulChain {
        /// One MPSMatrixMultiplication per layer (fixed shapes).
        kernels: Vec<Retained<MPSMatrixMultiplication>>,
        /// Weight matrix per layer (wraps the `[out,in]` weight buffer).
        weight_mats: Vec<Retained<MPSMatrix>>,
        /// Left/input matrix per layer (layer 0 = feature buffer; else an act).
        in_mats: Vec<Retained<MPSMatrix>>,
        /// Result matrix per layer (last = out buffer; else an act).
        out_mats: Vec<Retained<MPSMatrix>>,
        /// Bias MTLBuffer per layer (`[out]` f32, Shared).
        bias_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        /// Output column count per layer (for the bias+ReLU grid).
        out_cols: Vec<u32>,
        /// ReLU applied to this layer's output (false only on the last).
        relu: Vec<bool>,
        /// The fused bias+ReLU compute pipeline (compiled once).
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        /// Kept alive: the weight buffers the weight matrices reference.
        _weight_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        /// Kept alive: the two ping-pong activation buffers.
        _act_bufs: [Retained<ProtocolObject<dyn MTLBuffer>>; 2],
        max_rows: usize,
    }

    /// The fused bias+ReLU kernel (one thread per output element, buffer is
    /// row-major `[rows, out_cols]` tightly packed = the MPSMatrix layout).
    const BIAS_RELU_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bias_relu(
    device float*        buf      [[buffer(0)]],
    const device float*  bias     [[buffer(1)]],
    constant uint&       out_cols [[buffer(2)]],
    constant uint&       do_relu  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint o = gid % out_cols;
  float v = buf[gid] + bias[o];
  if (do_relu != 0u) { v = fmax(v, 0.0f); }
  buf[gid] = v;
}
"#;

    impl MatmulChain {
        /// Build the whole chain once. `dims` are `(in, out)` per layer in blob
        /// order; `flat` is `w[out*in]` then `b[out]` per layer (Mlp layout).
        /// `input_mtl` is the feature buffer (`[max_rows, dims[0].0]`),
        /// `out_mtl` the result buffer (`[max_rows, dims.last().1]`).
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn new(
            device: &Retained<ProtocolObject<dyn MTLDevice>>,
            dims: &[(usize, usize)],
            flat: &[f32],
            max_rows: usize,
            input_mtl: &Retained<ProtocolObject<dyn MTLBuffer>>,
            out_mtl: &Retained<ProtocolObject<dyn MTLBuffer>>,
        ) -> Result<Self, String> {
            let f32sz = std::mem::size_of::<f32>();
            let max_width = dims.iter().map(|d| d.1).max().unwrap_or(1);

            // Two ping-pong activation buffers, sized to the widest layer.
            let act_bytes = max_rows * max_width * f32sz;
            let make_buf = |bytes: usize, what: &str| {
                device
                    .newBufferWithLength_options(bytes.max(f32sz), MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| format!("rdirect_live: MatmulChain {what} alloc failed"))
            };
            let act0 = make_buf(act_bytes, "act0")?;
            let act1 = make_buf(act_bytes, "act1")?;

            let mtl_matrix = |buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
                              rows: usize,
                              cols: usize|
             -> Retained<MPSMatrix> {
                let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    rows,
                    cols,
                    cols * f32sz,
                    MPSDataType::Float32,
                );
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buf, &desc)
            };

            let mut kernels = Vec::with_capacity(dims.len());
            let mut weight_mats = Vec::with_capacity(dims.len());
            let mut in_mats = Vec::with_capacity(dims.len());
            let mut out_mats = Vec::with_capacity(dims.len());
            let mut bias_bufs = Vec::with_capacity(dims.len());
            let mut weight_bufs = Vec::with_capacity(dims.len());
            let mut out_cols = Vec::with_capacity(dims.len());
            let mut relu = Vec::with_capacity(dims.len());

            let mut cursor = 0usize;
            for (li, &(in_dim, out_dim)) in dims.iter().enumerate() {
                let w_flat = &flat[cursor..cursor + in_dim * out_dim];
                cursor += in_dim * out_dim;
                let b_flat = &flat[cursor..cursor + out_dim];
                cursor += out_dim;
                let is_last = li == dims.len() - 1;

                // Weight buffer holds the CPU-native [out,in] row-major weights;
                // transposeRight in the GEMM makes B = Wᵀ = [in,out].
                let wbuf = make_buf(in_dim * out_dim * f32sz, "weight")?;
                std::ptr::copy_nonoverlapping(
                    w_flat.as_ptr(),
                    wbuf.contents().as_ptr() as *mut f32,
                    in_dim * out_dim,
                );
                let wmat = mtl_matrix(&wbuf, out_dim, in_dim);

                let bbuf = make_buf(out_dim * f32sz, "bias")?;
                std::ptr::copy_nonoverlapping(
                    b_flat.as_ptr(),
                    bbuf.contents().as_ptr() as *mut f32,
                    out_dim,
                );

                // Input matrix: layer 0 reads the feature buffer; later layers
                // read the previous activation (ping-pong act[(li-1)%2]).
                let in_mat = if li == 0 {
                    mtl_matrix(input_mtl, max_rows, in_dim)
                } else if (li - 1) % 2 == 0 {
                    mtl_matrix(&act0, max_rows, in_dim)
                } else {
                    mtl_matrix(&act1, max_rows, in_dim)
                };
                // Output matrix: last layer writes the result buffer; hidden
                // layers write act[li%2].
                let out_mat = if is_last {
                    mtl_matrix(out_mtl, max_rows, out_dim)
                } else if li % 2 == 0 {
                    mtl_matrix(&act0, max_rows, out_dim)
                } else {
                    mtl_matrix(&act1, max_rows, out_dim)
                };

                let kernel = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    device,
                    false, // A not transposed
                    true,  // B = Wᵀ (weights stored [out,in])
                    max_rows,
                    out_dim,
                    in_dim,
                    1.0,
                    0.0,
                );

                kernels.push(kernel);
                weight_mats.push(wmat);
                in_mats.push(in_mat);
                out_mats.push(out_mat);
                bias_bufs.push(bbuf);
                weight_bufs.push(wbuf);
                out_cols.push(out_dim as u32);
                relu.push(!is_last);
            }

            // Compile the fused bias+ReLU kernel once.
            let src = NSString::from_str(BIAS_RELU_MSL);
            let lib = device
                .newLibraryWithSource_options_error(&src, None)
                .map_err(|e| format!("rdirect_live: bias_relu compile failed: {e:?}"))?;
            let func = lib
                .newFunctionWithName(&NSString::from_str("bias_relu"))
                .ok_or_else(|| "rdirect_live: bias_relu function missing".to_string())?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| format!("rdirect_live: bias_relu pipeline failed: {e:?}"))?;

            Ok(Self {
                kernels,
                weight_mats,
                in_mats,
                out_mats,
                bias_bufs,
                out_cols,
                relu,
                pipeline,
                _weight_bufs: weight_bufs,
                _act_bufs: [act0, act1],
                max_rows,
            })
        }

        /// Encode the whole forward onto `cmd` (matmul + bias/ReLU per layer).
        /// No allocation — everything was built in `new`.
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn encode(&self, cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
            let tg = self.pipeline.maxTotalThreadsPerThreadgroup().min(256);
            for li in 0..self.kernels.len() {
                self.kernels[li].encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    cmd,
                    &self.in_mats[li],
                    &self.weight_mats[li],
                    &self.out_mats[li],
                );
                let enc = cmd
                    .computeCommandEncoder()
                    .expect("rdirect_live: computeCommandEncoder");
                enc.setComputePipelineState(&self.pipeline);
                let out_buf = self.out_mats[li].data();
                enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.bias_bufs[li]), 0, 1);
                let out_cols = self.out_cols[li];
                let do_relu: u32 = if self.relu[li] { 1 } else { 0 };
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&out_cols as *const u32 as *mut c_void).unwrap(),
                    std::mem::size_of::<u32>(),
                    2,
                );
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&do_relu as *const u32 as *mut c_void).unwrap(),
                    std::mem::size_of::<u32>(),
                    3,
                );
                let total = self.max_rows * out_cols as usize;
                enc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: total, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
                enc.endEncoding();
            }
        }
    }

    /// S9 double buffering: how many independent tensor-data/output sets the
    /// pipeline rotates through. Two = the render thread consumes set[frame%2]
    /// while the encode thread pre-encodes the OTHER set's next command buffer.
    const SET_COUNT: usize = 2;

    /// The render-thread-visible half of one buffer set: the wgpu views the
    /// gather (feature) and demod (out) passes touch, plus the MTLBuffers held
    /// alive. The objc encode half lives in `EncodeSet` (owned by the encode
    /// thread via the shared `EncodeCtx`).
    struct SetWgpu {
        /// wgpu view of this set's feature MTLBuffer (STORAGE): gather target.
        feature_buf: wgpu::Buffer,
        /// wgpu view of this set's output MTLBuffer (STORAGE): demod source.
        out_buf: wgpu::Buffer,
        /// Held alive: the MTLBuffers the wgpu views and the EncodeSet share.
        #[allow(dead_code)]
        feature_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
        #[allow(dead_code)]
        out_mtl: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    /// The objc encode half of one buffer set (feed/result tensor-data arrays
    /// over the set's zero-copy MTLBuffers, and the raw GEMM chain bound to
    /// them). Read-only after construction except during `encode`, which the
    /// pipeline guarantees only ONE thread touches per set at a time (S9).
    struct EncodeSet {
        /// Input feed array `[in_td]` (wraps this set's feature MTLBuffer).
        inputs: Retained<NSArray<MPSGraphTensorData>>,
        /// Result array `[out_td]` (wraps this set's output MTLBuffer).
        results: Retained<NSArray<MPSGraphTensorData>>,
        /// S5 raw-kernel forward bound to this set's feature/out MTLBuffers.
        chain: MatmulChain,
    }

    /// The encode context shared (Arc) between the render thread's synchronous
    /// set-0 path (ordeal/example) and the S9 encode thread. Holds the compiled
    /// executable (shared, MPSGraph encode is documented thread-safe across
    /// distinct command buffers) and the per-set encode halves.
    ///
    /// SAFETY (`Send`+`Sync`): the objc handles are thread-affine in general,
    /// but this context is used under a strict ownership discipline — either the
    /// synchronous path (pipeline NOT started, calling thread only) OR the
    /// pipeline (encode thread encodes, render thread commits), never both, and
    /// the double buffering means the encode thread and render thread never
    /// touch the SAME `EncodeSet` concurrently. `executable`/`queue` are the
    /// only shared-mutable objc state and both are Apple-documented thread-safe
    /// (executable encode; command-buffer creation off a queue).
    struct EncodeCtx {
        /// S12 — the net's OWN command queues (S11: ONE PER SET). The encode
        /// thread creates and the render thread commits net command buffers HERE,
        /// so the net's Metal work no longer shares the wgpu render queue with
        /// trace — the S9 contention that regressed trace +6 ms was CPU driver-
        /// lock on that shared queue.
        ///
        /// S11 net-wedge fix — WHY ONE QUEUE PER SET: MPSGraph's
        /// `encodeToCommandBuffer` internally `commitAndContinue`s, so `base`
        /// (carrying our `encodeWaitForEvent(V)`) is COMMITTED at ENCODE time,
        /// 1–2 frames AHEAD of `signal_gather_ready`. On ONE shared net queue
        /// that lands set-1's V2 wait on the FIFO ahead of set-0's continuation;
        /// the render thread signals one V per frame but blocks on set-0's
        /// completion queued BEHIND set-1's unsignaled wait → cross-buffer FIFO
        /// deadlock (base times out, GPU 0.00, black eyes). A DEDICATED queue
        /// per set removes the cross-set FIFO coupling: set-1's early wait can
        /// no longer stall set-0's buffers. Within one set's queue the waits are
        /// strictly increasing and signaled in frame order, so no self-block.
        /// Cross-queue hazards stay EXACTLY two: gather→net (the `event` fence)
        /// and net→demod (`commit_net`'s `waitUntilCompleted`).
        net_queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
        /// S12 — the gather→net fence. Protocol: each pipelined net command
        /// buffer `encodeWaitForEvent`s its own value V (claimed from
        /// `wait_counter` at encode time, on the net queue); after that frame's
        /// gather the render queue `encodeSignalEvent`s the SAME V
        /// (`signal_gather_ready`). FIFO consumption == encode order == strictly
        /// increasing V, so signals are monotonic (a fresh event is 0; V starts
        /// at 1). This is the ONLY event needed — net→demod is a CPU fence.
        event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        /// S12 — monotonic wait-value dispenser (see `event`). Starts at 1.
        wait_counter: AtomicU64,
        /// Compiled once for the FIXED shape `[max_pixels, in_features]`.
        executable: Retained<MPSGraphExecutable>,
        sets: Vec<EncodeSet>,
        /// S5/S8 A/B: true = MPSGraph executable (S8 default), false = chain.
        use_mpsgraph: AtomicBool,
    }
    // SAFETY: see the doc comment above — the ownership discipline, not raw
    // thread-safety of every field, is what makes this sound.
    unsafe impl Send for EncodeCtx {}
    unsafe impl Sync for EncodeCtx {}

    /// A net command buffer ENCODED but NOT yet committed — the S9 hand-off from
    /// the encode thread (which built it, ~14 ms CPU, off the critical path) to
    /// the render thread (which commits + waits, on the critical path only for
    /// the GPU work). `set` names which buffer set the gather must fill BEFORE
    /// this is committed, so the net reads the frame's own fresh evidence
    /// (0 latency — the pre-encode records references, not data).
    struct PreparedNet {
        base: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        /// Some on the MPSGraph path (the wrapper we commit), None on the chain.
        mps: Option<Retained<MPSCommandBuffer>>,
        set: usize,
        /// S12 — the gather→net fence value THIS buffer waits on (net queue),
        /// which the render thread signals after the gather. `None` on the
        /// sync/ordeal path (single queue+thread, gather already polled
        /// complete — no cross-queue wait encoded).
        wait_value: Option<u64>,
    }
    // SAFETY: the encode thread builds it, hands sole ownership across the
    // channel to the render thread, which is the only place it is committed.
    unsafe impl Send for PreparedNet {}

    impl EncodeCtx {
        /// Encode the net forward for `set` onto a FRESH command buffer off the
        /// shared queue — no commit, no wait. This is the ~14 ms CPU cost S9
        /// moves off the critical path. The path (MPSGraph vs chain) is read
        /// from `use_mpsgraph` at encode time.
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn encode(&self, set: usize, pipelined: bool) -> PreparedNet {
            // S12: net command buffers are created off the DEDICATED net queue
            // (not the shared wgpu render queue) — the whole point of the shift.
            // S11: ONE queue PER SET so MPSGraph's early (commitAndContinue)
            // commit of set-1's event wait cannot FIFO-block set-0's buffers.
            let base = self
                .net_queues[set]
                .commandBuffer()
                .expect("rdirect_live: commandBuffer alloc failed (encode)");
            // S12: the pipelined path fences on the gather across the queue split
            // — claim this buffer's monotonic value V and encode the wait FIRST
            // (before the forward), so the net cannot read a half-written feature
            // buffer. The sync/ordeal path runs single-queue+thread with the
            // gather already polled complete, so it needs no cross-queue wait.
            let wait_value = if pipelined {
                let v = self.wait_counter.fetch_add(1, Ordering::Relaxed);
                let ev = ProtocolObject::from_ref(&*self.event);
                base.encodeWaitForEvent_value(ev, v);
                if net_trace() {
                    eprintln!("[net-trace] encode set={set} awaits V={v} (event signaled={})", self.event.signaledValue());
                }
                Some(v)
            } else {
                None
            };
            let s = &self.sets[set];
            if self.use_mpsgraph.load(Ordering::Relaxed) {
                let mps = MPSCommandBuffer::commandBufferWithCommandBuffer(&base);
                let _ = self
                    .executable
                    .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                        &mps,
                        &s.inputs,
                        Some(&s.results),
                        // None: the render thread owns commit + wait, so the
                        // executable must not internally commit/wait.
                        None,
                    );
                PreparedNet { base, mps: Some(mps), set, wait_value }
            } else {
                s.chain.encode(&base);
                PreparedNet { base, mps: None, set, wait_value }
            }
        }
    }

    /// N0.i S13 — COMMIT ONLY (no wait). Submits the pre-encoded net buffer to
    /// its queue and returns immediately, so its GPU forward can OVERLAP the
    /// NEXT frame's trace+gather on the render queue. The wait moves one frame
    /// downstream (`wait_prepared`). Returns the CPU ms the commit itself cost
    /// (the S13 probe: is the ~8.5 ms net_wall−net_gpu gap the commit or the
    /// wait?). Render-thread only.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn commit_prepared_nowait(p: &PreparedNet) -> f64 {
        if net_trace() {
            eprintln!("[net-trace] pre-commit set={} base.status={:?}", p.set, p.base.status());
        }
        let t = std::time::Instant::now();
        if let Some(mps) = &p.mps {
            mps.commit();
        } else {
            p.base.commit();
        }
        t.elapsed().as_secs_f64() * 1000.0
    }

    /// N0.i S13 — WAIT ONLY. Blocks until an already-committed `PreparedNet`'s
    /// GPU forward completes and returns GPU-only ms (MTLCommandBuffer GPU
    /// timestamps). Called ONE frame after `commit_prepared_nowait`, so in
    /// steady state the buffer has been running during the intervening
    /// trace+gather and this wait is short (overlap dividend). Render-thread
    /// only.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn wait_prepared(p: &PreparedNet) -> f64 {
        if let Some(mps) = &p.mps {
            mps.rootCommandBuffer().waitUntilCompleted();
        } else {
            p.base.waitUntilCompleted();
        }
        // S11 INSTRUMENT: after the wait, read the base buffer's terminal status
        // + error to SEE the stuck value pair (net wedge diagnosis).
        if net_trace() {
            let status = p.base.status();
            let err = p.base.error();
            eprintln!(
                "[net-trace] commit set={} wait_value={:?} base.status={:?} root_is_base={:?} err={}",
                p.set,
                p.wait_value,
                status,
                p.mps.as_ref().map(|m| std::ptr::eq(
                    Retained::as_ptr(&m.rootCommandBuffer()) as *const (),
                    Retained::as_ptr(&p.base) as *const (),
                )),
                err.is_some(),
            );
        }
        (p.base.GPUEndTime() - p.base.GPUStartTime()) * 1000.0
    }

    /// The zero-copy pool (N0.b → S9): feature/output MTLBuffers per set,
    /// allocated ONCE, sized to a fixed `max_pixels` ceiling. Each set's
    /// `feature_buf` is the SAME MTLBuffer wrapped as a wgpu STORAGE buffer, so
    /// the gather writes it and the forward reads it with no copy (the spike's
    /// 157 MB/frame churn dies here). Only present on the live wgpu path
    /// (`from_wgpu_queue`). `ctx` holds the compiled executable + per-set encode
    /// halves, shared with the S9 encode thread.
    struct SharedPool {
        sets: Vec<SetWgpu>,
        max_pixels: usize,
        ctx: Arc<EncodeCtx>,
    }

    /// The S9 pipeline handle: the encode thread + the channels that hand
    /// pre-encoded net command buffers from it to the render thread, plus the
    /// buffer currently claimed by `begin_frame` awaiting `commit_net`.
    struct Pipeline {
        /// render → encode: "prepare the next command buffer for set i".
        tx_req: Sender<usize>,
        /// encode → render: a `PreparedNet` (FIFO in request order).
        rx_ready: Receiver<PreparedNet>,
        /// The buffer `begin_frame` claimed this frame, consumed by `commit_net`.
        current: Option<PreparedNet>,
        /// N0.i S13 FRAME OVERLAP — the buffer committed LAST frame, its GPU
        /// forward running (overlapping this frame's trace+gather) and awaiting
        /// its wait one frame downstream. `commit_net` commits `current` without
        /// blocking, moves it here, and waits/returns the PREVIOUS occupant (the
        /// frame whose net is now complete) for the demod+present. This is the
        /// double-buffered output that keeps output-or-nothing intact at the
        /// cost of one frame of DISPLAY latency (never evidence latency: each
        /// presented image is the complete image of its OWN frame's trace).
        pending: Option<PreparedNet>,
        /// N0.i S13 A/B: `true` (default) = frame overlap (commit now, wait the
        /// PREVIOUS buffer). `GAIA_NATIVE_NET_NOOVERLAP=1` forces the old S11
        /// blocking path (commit + wait THIS buffer) for a same-binary wall-fps
        /// A/B — proving whether overlap moves throughput on the single M1 GPU.
        overlap: bool,
        join: Option<JoinHandle<()>>,
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
        /// S12 — the gather→net fence, render-thread clone (created in `build`,
        /// shared with the encode thread via `EncodeCtx.event`). The render
        /// thread signals it (`signal_gather_ready`); the net queue waits on it.
        event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        /// S3 instrument: GPU-only time of the last forward (MTLCommandBuffer
        /// GPUEndTime − GPUStartTime, ms). The `runWithMTLCommandQueue` API hid
        /// this; the encode path below owns the command buffer so it reads it.
        last_gpu_ms: Cell<f64>,
        /// N0.i S13 probe: CPU ms the last net COMMIT cost (no wait) and the
        /// last downstream WAIT cost. Splits the ~8.5 ms net_wall−net_gpu gap
        /// into its commit half (on the critical path) vs its wait half (which
        /// the frame overlap hides). Read by the budget line.
        last_commit_ms: Cell<f64>,
        last_wait_ms: Cell<f64>,
        /// S8 A/B toggle: force the raw MPSMatrixMultiplication chain (default
        /// TRUE = MPSGraph, the fused-GPU winner, per S8). `GAIA_NATIVE_NET_CHAIN=1`
        /// opts into the chain; `set_use_mpsgraph` flips it (the parity ordeal).
        /// Mirrored into `pool.ctx.use_mpsgraph` once the pool exists.
        use_mpsgraph: Cell<bool>,
        /// S9 pipeline (encode thread). `None` until `start_pipeline`; the
        /// synchronous set-0 path (ordeal/example) requires it stay `None`.
        pipeline: RefCell<Option<Pipeline>>,
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
                // S12: the gather→net fence, made on the same device both queues
                // share (an MTLSharedEvent works cross-queue on one device).
                let event = device
                    .newSharedEvent()
                    .ok_or_else(|| "rdirect_live: newSharedEvent failed".to_string())?;

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
                    event,
                    last_gpu_ms: Cell::new(0.0),
                    last_commit_ms: Cell::new(0.0),
                    last_wait_ms: Cell::new(0.0),
                    // S8: MPSGraph (fused GPU, 6.65 ms) is the DEFAULT — it beat
                    // the un-fused chain (42.8 ms GPU) 1.8× in N0.f. The chain
                    // stays a lab A/B opt-in via `GAIA_NATIVE_NET_CHAIN`.
                    use_mpsgraph: Cell::new(!matches!(
                        std::env::var("GAIA_NATIVE_NET_CHAIN").ok().as_deref(),
                        Some("1") | Some("true") | Some("on")
                    )),
                    pipeline: RefCell::new(None),
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
            // MTLBuffers we clone into wgpu outlive the wgpu buffers (both held
            // in `SharedPool` / the shared `EncodeCtx`).
            unsafe {
                let in_shape = shape(&[max_pixels, self.in_features]);
                let out_shape = shape(&[max_pixels, self.out_channels]);

                // ── compile the executable ONCE for the fixed shape (shared by
                // both sets — MPSGraph encode is thread-safe across the distinct
                // command buffers the two sets feed). ──
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

                let dims: Vec<(usize, usize)> = self
                    .cpu_ref
                    .layer_dims()
                    .iter()
                    .map(|&(i, o)| (i as usize, o as usize))
                    .collect();
                let flat = self.cpu_ref.flat_weights();

                // ── S9: SET_COUNT independent feature/output buffer sets. ──
                let mut sets_wgpu: Vec<SetWgpu> = Vec::with_capacity(SET_COUNT);
                let mut encode_sets: Vec<EncodeSet> = Vec::with_capacity(SET_COUNT);
                for si in 0..SET_COUNT {
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
                            // COPY_SRC so ordeals can read the gather output back
                            // for parity (the live path never copies it).
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        },
                    );
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
                    let chain = MatmulChain::new(
                        mtl_device,
                        &dims,
                        &flat,
                        max_pixels,
                        &feature_mtl,
                        &out_mtl,
                    )?;
                    let _ = si;
                    sets_wgpu.push(SetWgpu { feature_buf, out_buf, feature_mtl, out_mtl });
                    encode_sets.push(EncodeSet { inputs, results, chain });
                }

                // S12: the net's dedicated command queues (separate from the
                // wgpu render queue) — S11: ONE PER SET (see EncodeCtx doc), so
                // set-1's early-committed event wait cannot FIFO-block set-0.
                let mut net_queues = Vec::with_capacity(SET_COUNT);
                for _ in 0..SET_COUNT {
                    net_queues.push(
                        mtl_device
                            .newCommandQueue()
                            .ok_or_else(|| "rdirect_live: net newCommandQueue failed".to_string())?,
                    );
                }
                let ctx = Arc::new(EncodeCtx {
                    net_queues,
                    event: self.event.clone(),
                    // Fresh event is 0; V=1 is the first real fence.
                    wait_counter: AtomicU64::new(1),
                    executable,
                    sets: encode_sets,
                    use_mpsgraph: AtomicBool::new(self.use_mpsgraph.get()),
                });

                self.pool = Some(SharedPool { sets: sets_wgpu, max_pixels, ctx });
            }
            Ok(())
        }

        /// The gather's destination STORAGE buffer for buffer set 0 (the
        /// synchronous ordeal/example path). `None` unless built via
        /// `from_wgpu_queue`. The S9 live path uses `feature_buffer_set`.
        pub fn feature_buffer(&self) -> Option<&wgpu::Buffer> {
            self.pool.as_ref().map(|p| &p.sets[0].feature_buf)
        }

        /// CUT 2: set 0's net OUTPUT MTLBuffer wrapped for wgpu (demod source,
        /// synchronous path). The S9 live path uses `output_buffer_set`.
        pub fn output_buffer(&self) -> Option<&wgpu::Buffer> {
            self.pool.as_ref().map(|p| &p.sets[0].out_buf)
        }

        /// S9: the gather's destination STORAGE buffer for buffer set `set`.
        pub fn feature_buffer_set(&self, set: usize) -> Option<&wgpu::Buffer> {
            self.pool.as_ref().map(|p| &p.sets[set].feature_buf)
        }

        /// S9: set `set`'s net OUTPUT MTLBuffer wrapped for wgpu (demod source).
        pub fn output_buffer_set(&self, set: usize) -> Option<&wgpu::Buffer> {
            self.pool.as_ref().map(|p| &p.sets[set].out_buf)
        }

        /// How many double-buffer sets the pipeline rotates through.
        pub fn set_count(&self) -> usize {
            SET_COUNT
        }

        /// SYNCHRONOUS forward over buffer set `set`: encode + commit + wait on
        /// the CALLING thread, leaving the result in `output_buffer_set(set)`.
        /// Used by the ordeal/example (pipeline NOT started). Records GPU-only
        /// ms. Panics if the S9 pipeline is running (that thread owns the
        /// encode objects).
        fn run_set_sync(&self, set: usize, n: usize) -> Result<(), String> {
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
            assert!(
                self.pipeline.borrow().is_none(),
                "rdirect_live: run_set_sync while the S9 pipeline owns the encode context"
            );
            // Mirror the live toggle into the shared ctx so the sync encode picks
            // the same path the ordeal selected via `set_use_mpsgraph`.
            pool.ctx
                .use_mpsgraph
                .store(self.use_mpsgraph.get(), Ordering::Relaxed);
            // SAFETY: encode + commit + wait on this thread; the pipeline is not
            // running (asserted), so no other thread touches `ctx.sets[set]`.
            // Autorelease pool drains MPSGraph's per-run intermediate NDArrays.
            objc2::rc::autoreleasepool(|_| unsafe {
                let prepared = pool.ctx.encode(set, false);
                // Sync/ordeal path: commit + wait back-to-back on this thread.
                let _ = commit_prepared_nowait(&prepared);
                let gpu_ms = wait_prepared(&prepared);
                self.last_gpu_ms.set(gpu_ms);
            });
            Ok(())
        }

        /// CUT 1 + CUT 2 SYNCHRONOUS live path (set 0): run the pooled forward
        /// and leave the result ON the GPU in `output_buffer()`. Kept for the
        /// non-pipelined bring-up / example. The S9 live loop uses
        /// `begin_frame` + `commit_net`.
        pub fn forward_shared_gpu(&self, n: usize) -> Result<(), String> {
            self.run_set_sync(0, n)
        }

        // ── S9 PIPELINE ─────────────────────────────────────────────────────

        /// Start the S9 encode thread. After this the synchronous set-0 path
        /// (`forward_shared*`) is invalid (the thread owns the encode context);
        /// the render loop drives the net via `begin_frame` + `commit_net`.
        /// Primes the pipeline `SET_COUNT` deep (one prepared buffer per set) so
        /// the first frame never stalls on the encode.
        pub fn start_pipeline(&self) -> Result<(), String> {
            if self.pipeline.borrow().is_some() {
                return Ok(());
            }
            let pool = self.pool.as_ref().ok_or_else(|| {
                "rdirect_live: start_pipeline needs the shared pool (from_wgpu_queue)".to_string()
            })?;
            // Sync the toggle into the ctx BEFORE the thread reads it.
            pool.ctx
                .use_mpsgraph
                .store(self.use_mpsgraph.get(), Ordering::Relaxed);
            let ctx = pool.ctx.clone();
            let (tx_req, rx_req) = std::sync::mpsc::channel::<usize>();
            let (tx_ready, rx_ready) = std::sync::mpsc::channel::<PreparedNet>();
            // The encode thread: for each requested set, build (encode, ~14 ms
            // CPU, NO commit) a fresh net command buffer and hand it back. This
            // is the CPU work S9 moves off the render thread's critical path.
            let join = std::thread::Builder::new()
                .name("rdirect-net-encode".into())
                .spawn(move || {
                    while let Ok(set) = rx_req.recv() {
                        let prepared =
                            objc2::rc::autoreleasepool(|_| unsafe { ctx.encode(set, true) });
                        if tx_ready.send(prepared).is_err() {
                            break; // render side gone
                        }
                    }
                })
                .map_err(|e| format!("rdirect_live: encode thread spawn failed: {e}"))?;
            // Prime: request one buffer per set (order 0,1,… = the render
            // thread's consumption order).
            for set in 0..SET_COUNT {
                let _ = tx_req.send(set);
            }
            let overlap = !matches!(
                std::env::var("GAIA_NATIVE_NET_NOOVERLAP").as_deref(),
                Ok("1") | Ok("true")
            );
            *self.pipeline.borrow_mut() = Some(Pipeline {
                tx_req,
                rx_ready,
                current: None,
                pending: None,
                overlap,
                join: Some(join),
            });
            Ok(())
        }

        /// S9: claim this frame's pre-encoded net command buffer. Blocks until
        /// the encode thread has one ready (immediate in steady state — the
        /// pipeline stays primed). Returns the buffer SET the caller must fill
        /// with the gather BEFORE `commit_net`, so the net reads THIS frame's
        /// own fresh evidence (0 latency). Panics if the pipeline is not
        /// running.
        pub fn begin_frame(&self) -> usize {
            let mut guard = self.pipeline.borrow_mut();
            let pipe = guard
                .as_mut()
                .expect("rdirect_live: begin_frame before start_pipeline");
            let prepared = pipe
                .rx_ready
                .recv()
                .expect("rdirect_live: encode thread hung up");
            let set = prepared.set;
            pipe.current = Some(prepared);
            set
        }

        /// N0.i S13 FRAME OVERLAP: commit this frame's claimed net buffer
        /// WITHOUT blocking (its GPU forward now overlaps the NEXT frame's
        /// trace+gather), stash it as `pending`, and WAIT/return the PREVIOUS
        /// frame's buffer — whose net has been running during THIS frame's
        /// trace+gather and is (near-)complete, so the wait is short. The
        /// returned `Some(set)` names the buffer set whose net output is ready
        /// for demod+present; `None` only on the very first frame (nothing
        /// finished yet — present nothing, output-or-nothing holds).
        ///
        /// Per-set net queues (N0.h) keep the FIFO waits monotonic PER QUEUE:
        /// each set's queue runs its buffers in commit order (frame N, N+2, …),
        /// each awaiting its own strictly-increasing V signaled that frame — the
        /// deferral moves only WHEN we `waitUntilCompleted`, never the
        /// signal/wait pairing, so the N0.h wedge cannot reopen.
        ///
        /// Refill cadence is unchanged: one refill request per frame for the
        /// CURRENT set (the buffer just committed), so the encode thread stays
        /// SET_COUNT deep and `begin_frame` never stalls.
        pub fn commit_net(&self) -> Result<Option<usize>, String> {
            let mut guard = self.pipeline.borrow_mut();
            let pipe = guard
                .as_mut()
                .expect("rdirect_live: commit_net before start_pipeline");
            let current = pipe
                .current
                .take()
                .expect("rdirect_live: commit_net without begin_frame");
            let cur_set = current.set;
            // Commit THIS frame's buffer without waiting (overlap starts now).
            // SAFETY: this thread solely owns `current`; commit here.
            let commit_ms =
                objc2::rc::autoreleasepool(|_| unsafe { commit_prepared_nowait(&current) });
            self.last_commit_ms.set(commit_ms);
            // Keep the encode a frame ahead: refill the set we just committed.
            let _ = pipe.tx_req.send(cur_set);
            // N0.i S13 A/B: blocking path — wait THIS buffer now, present it this
            // frame (no pending rotation, no display latency). The wall-fps A/B
            // against the overlap path on the same binary.
            if !pipe.overlap {
                let (gpu_ms, wait_ms) = objc2::rc::autoreleasepool(|_| unsafe {
                    let t = std::time::Instant::now();
                    let g = wait_prepared(&current);
                    (g, t.elapsed().as_secs_f64() * 1000.0)
                });
                self.last_gpu_ms.set(gpu_ms);
                self.last_wait_ms.set(wait_ms);
                // Buffer stays owned here until it drops (waited, safe).
                return Ok(Some(cur_set));
            }
            // Rotate: the buffer committed LAST frame becomes the one we wait on.
            let prev = pipe.pending.replace(current);
            match prev {
                Some(p) => {
                    let demod_set = p.set;
                    // SAFETY: `p` was committed last frame and is solely owned
                    // here now; wait for its GPU forward (short — overlapped).
                    let (gpu_ms, wait_ms) = objc2::rc::autoreleasepool(|_| unsafe {
                        let t = std::time::Instant::now();
                        let g = wait_prepared(&p);
                        (g, t.elapsed().as_secs_f64() * 1000.0)
                    });
                    self.last_gpu_ms.set(gpu_ms);
                    self.last_wait_ms.set(wait_ms);
                    Ok(Some(demod_set))
                }
                None => {
                    // First frame: nothing finished to present yet.
                    self.last_gpu_ms.set(0.0);
                    self.last_wait_ms.set(0.0);
                    Ok(None)
                }
            }
        }

        /// N0.i S13 probe: split of the net stage's wall cost — the commit CPU
        /// ms (critical path) and the downstream wait ms (overlap-hidden).
        pub fn last_commit_ms(&self) -> f64 {
            self.last_commit_ms.get()
        }
        pub fn last_wait_ms(&self) -> f64 {
            self.last_wait_ms.get()
        }

        /// S12: release the gather→net fence. Call AFTER the gather submit AND
        /// its `device.poll(wait)` (the render loop already CPU-waits the gather
        /// to full GPU completion before the net stage), BEFORE `commit_net`.
        ///
        /// The event stays the cross-queue primitive: the net command buffer
        /// `encodeWaitForEvent`s its value V on the net queue, and this signals
        /// the SAME V. The signal is CPU-side (`setSignaledValue`), NOT a GPU
        /// command buffer on the render queue — the first S12 cut tried the
        /// GPU-queue signal (charter's literal protocol) and it DEADLOCKED: wgpu
        /// owns the render queue's submission timeline, so a raw signal buffer
        /// injected onto it never scheduled and the net's wait timed out (black
        /// frame, GPU 0.00). The CPU signal is correct here BECAUSE the gather is
        /// already CPU-confirmed complete by the preceding `device.poll(wait)`,
        /// so V=1 truthfully means "the feature buffer for this frame is fully
        /// written" — the same guarantee the GPU signal would have carried, with
        /// no foreign buffer on wgpu's queue. `setSignaledValue` from the CPU
        /// unblocks the GPU waiter on the net queue (that is its purpose).
        /// No-op if the current buffer carries no fence. Render-thread only.
        pub fn signal_gather_ready(&self) {
            let guard = self.pipeline.borrow();
            let pipe = guard
                .as_ref()
                .expect("rdirect_live: signal_gather_ready before start_pipeline");
            let wait_value = pipe
                .current
                .as_ref()
                .expect("rdirect_live: signal_gather_ready without begin_frame")
                .wait_value;
            if let Some(v) = wait_value {
                if net_trace() {
                    eprintln!("[net-trace] signal set V={v} (was signaled={})", self.event.signaledValue());
                }
                self.event.setSignaledValue(v);
            }
        }

        /// S3 instrument: GPU-only ms of the last forward (sync or pipelined).
        pub fn last_gpu_ms(&self) -> f64 {
            self.last_gpu_ms.get()
        }

        /// S5/S8 A/B (ordeal-only): force this instance's forward path.
        /// `true` = MPSGraph executable (S8 default), `false` = raw GEMM chain.
        /// Mirrored into the shared ctx for both sync and pipelined encodes.
        pub fn set_use_mpsgraph(&self, on: bool) {
            self.use_mpsgraph.set(on);
            if let Some(pool) = self.pool.as_ref() {
                pool.ctx.use_mpsgraph.store(on, Ordering::Relaxed);
            }
        }

        /// S12.5: the live forward path (true = MPSGraph default, false = chain).
        pub fn use_mpsgraph_now(&self) -> bool {
            self.use_mpsgraph.get()
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
            // Run the pooled forward on set 0 SYNCHRONOUSLY (blocks until done),
            // then read the Shared-storage output back to a Vec. The
            // ordeal/example CPU path keeps this Vec return; the live present
            // uses the S9 pipeline (no readback) + the GPU demod pass.
            self.run_set_sync(0, n)?;
            let pool = self.pool.as_ref().ok_or_else(|| {
                "rdirect_live: forward_shared needs the shared pool".to_string()
            })?;
            // SAFETY: set 0's `out_mtl` is Shared storage, sized to max_pixels ≥
            // n, and `run_set_sync` waited for the GPU forward to complete.
            unsafe {
                let ptr = pool.sets[0].out_mtl.contents().as_ptr() as *const f32;
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

    impl Drop for RdirectLive {
        fn drop(&mut self) {
            // Stop the S9 encode thread: drop the request Sender (closes its
            // recv), which ends the loop; the thread's `send` also fails once
            // the ready Receiver drops. Join so the thread's Arc<EncodeCtx> is
            // released before ours (no objc use-after-free race).
            if let Some(mut pipe) = self.pipeline.borrow_mut().take() {
                let join = pipe.join.take();
                drop(pipe); // drops tx_req + rx_ready → thread loop ends
                if let Some(j) = join {
                    let _ = j.join();
                }
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
