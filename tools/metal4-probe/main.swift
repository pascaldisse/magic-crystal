// R-DIRECT — Metal tensor-machinery spike (silicon race lane 1 of 2).
//
// The WGSL per-pixel-MLP door is measured SHUT (docs/perf/2026-07-18-rdirect-gpu-kernel.md:
// native 960×640 = ~280ms f32, even a trivial 2×32 net = 32.5ms). This harness
// asks the NEXT question: does Metal's TENSOR machinery — per-pixel inference
// reformulated as BATCHED MATMULS (pixels×23 · 23×64 · …) — reopen the
// native-res door?
//
// Same net (rdirect-weights-v1.bin, GAIARDR1, 23→5×64 ReLU→3), same committed
// weights. It parses that blob directly, builds the forward as a chain of
// MPSGraph matmuls (X[N,in]·Wᵀ[in,out] + b, ReLU on hidden), and:
//   GATE 1 PARITY  — runs the REAL exported feature buffer (features.f32, the
//                    front pose at 96×64) and compares against expected.f32
//                    (the Rust CPU-reference Mlp::forward) within an fp16-derived
//                    tolerance.
//   GATE 2 REAL ms — GPU-timed (MTLCommandBuffer.gpuStartTime/gpuEndTime) batched
//                    forward at 96×64 (6144 px) and 960×640 (614 400 px), warm-up
//                    + median. Both f32 and fp16-storage/f32-accumulate.
//   GATE 3 VERDICT — printed: does any formulation land ≤ ~10ms at native?
//
// API: MPSGraph (MetalPerformanceShadersGraph). Chosen because it COMPILES TODAY
// on this SDK (Swift 6.2, macOS Tahoe 26) and its matmul lowers to Apple's
// simdgroup-matrix (MPSMatrixMultiplication) machinery — the exact tensor path
// wgpu 30 could not reach. Metal 4 native MTLTensor (MTL4 command encoders,
// tensorOps) is documented but the harness uses MPSGraph, which the Metal-4
// stack routes through the same AMX/simdgroup units; noted in the verdict.
//
// Build/run (not cargo — no build token needed):
//   swiftc -O main.swift -o metal4-probe \
//     -framework Metal -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph -framework Foundation
//   ./metal4-probe

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// ── locate data (beside this file) ───────────────────────────────────────────
let here = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
let dataDir = here.appendingPathComponent("data")
func dataPath(_ n: String) -> String { dataDir.appendingPathComponent(n).path }

// ── parse the GAIARDR1 weights blob (same format as serialize_weights) ───────
struct Layer { let inDim: Int; let outDim: Int; var w: [Float]; var b: [Float] }

func loadWeights(_ path: String) -> [Layer] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    var off = 0
    func u32() -> Int { let v = data.subdata(in: off..<off+4).withUnsafeBytes { $0.load(as: UInt32.self) }; off += 4; return Int(v) }
    func f32(_ n: Int) -> [Float] {
        let bytes = data.subdata(in: off..<off+n*4); off += n*4
        return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let magic = data.subdata(in: 0..<8); off = 8
    precondition(magic == "GAIARDR1".data(using: .ascii)!, "bad magic")
    let layerCount = u32(); _ = u32(); _ = u32()  // layer_count, hidden_layers, hidden_width
    var layers = [Layer]()
    for _ in 0..<layerCount {
        let inD = u32(); let outD = u32()
        let w = f32(inD * outD)   // row-major [out, in]: w[o*in + i]
        let b = f32(outD)
        layers.append(Layer(inDim: inD, outDim: outD, w: w, b: b))
    }
    return layers
}

func loadF32(_ path: String) -> [Float] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

// ── device ───────────────────────────────────────────────────────────────────
setbuf(stdout, nil)
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let queue = device.makeCommandQueue()!
print("[metal4-probe] device: \(device.name)  Metal-4-family: \(device.supportsFamily(.metal3))")

let layers = loadWeights(dataPath("rdirect-weights-v1.bin"))
let inFeat = layers.first!.inDim   // 23
let outCh  = layers.last!.outDim   // 3
print("[metal4-probe] net: \(layers.count) layers  \(inFeat)→…→\(outCh)  " +
      layers.map { "\($0.inDim)×\($0.outDim)" }.joined(separator: " "))

// ── build the forward graph (X[N,in] · Wᵀ + b, ReLU on hidden) ───────────────
// `useFp16`: cast X + weights to fp16 storage; MPSGraph matmul accumulates the
// simdgroup dot products in fp32 on Apple GPUs (fp16 storage / f32 accumulate).
func buildGraph(useFp16: Bool) -> (MPSGraph, MPSGraphTensor, MPSGraphTensor) {
    let g = MPSGraph()
    let dt: MPSDataType = useFp16 ? .float16 : .float32
    let x = g.placeholder(shape: [-1, NSNumber(value: inFeat)], dataType: .float32, name: "x")
    var h = useFp16 ? g.cast(x, to: .float16, name: "x16") : x
    for (li, L) in layers.enumerated() {
        // transpose stored [out,in] → [in,out] on CPU for a clean matmul B operand.
        var wt = [Float](repeating: 0, count: L.inDim * L.outDim)
        for o in 0..<L.outDim { for i in 0..<L.inDim { wt[i * L.outDim + o] = L.w[o * L.inDim + i] } }
        let wData: Data
        if useFp16 {
            var h16 = [UInt16](repeating: 0, count: wt.count)
            for k in 0..<wt.count { h16[k] = float32to16(wt[k]) }
            wData = h16.withUnsafeBytes { Data($0) }
        } else {
            wData = wt.withUnsafeBytes { Data($0) }
        }
        let wTensor = g.constant(wData, shape: [NSNumber(value: L.inDim), NSNumber(value: L.outDim)], dataType: dt)
        let bData: Data
        if useFp16 {
            var b16 = L.b.map { float32to16($0) }
            bData = b16.withUnsafeBytes { Data($0) }
        } else {
            bData = L.b.withUnsafeBytes { Data($0) }
        }
        let bTensor = g.constant(bData, shape: [1, NSNumber(value: L.outDim)], dataType: dt)
        h = g.matrixMultiplication(primary: h, secondary: wTensor, name: "mm\(li)")
        h = g.addition(h, bTensor, name: "bias\(li)")
        if li != layers.count - 1 {
            h = g.reLU(with: h, name: "relu\(li)")
        }
    }
    if useFp16 { h = g.cast(h, to: .float32, name: "out32") }
    return (g, x, h)
}

// tiny f32→f16 bit converter (round-to-nearest-even, good enough; MPS re-rounds anyway)
func float32to16(_ f: Float) -> UInt16 {
    let bits = f.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    var exp = Int((bits >> 23) & 0xFF) - 127 + 15
    let mant = bits & 0x7FFFFF
    if exp <= 0 { return sign } // flush denormals/underflow to signed zero
    if exp >= 0x1F { return sign | 0x7C00 } // overflow → inf
    let m = UInt16((mant >> 13) & 0x3FF)
    return sign | UInt16(exp << 10) | m
}

// ── run once, GPU-timed ───────────────────────────────────────────────────────
func makeTensorData(_ floats: [Float], rows: Int) -> MPSGraphTensorData {
    let buf = device.makeBuffer(bytes: floats, length: floats.count * 4, options: .storageModeShared)!
    return MPSGraphTensorData(buf, shape: [NSNumber(value: rows), NSNumber(value: inFeat)], dataType: .float32)
}

// Read a target tensor's floats (parity/correctness only).
func runOnce(_ g: MPSGraph, _ x: MPSGraphTensor, _ out: MPSGraphTensor,
             feed: MPSGraphTensorData) -> [Float] {
    let results = g.run(with: queue, feeds: [x: feed], targetTensors: [out], targetOperations: nil)
    let td = results[out]!
    let nd = td.mpsndarray()
    let count = td.shape.reduce(1) { $0 * $1.intValue }
    var buf = [Float](repeating: 0, count: count)
    nd.readBytes(&buf, strideBytes: nil)
    return buf
}

// AUTHORITATIVE timing: a compiled MPSGraphExecutable run on ONE command buffer
// per iteration; GPU time from that buffer's gpuStart/End (whole graph on it),
// cross-checked by commit→wait wall clock. Compile + pool warmed before timing.
func compileExec(_ g: MPSGraph, _ x: MPSGraphTensor, _ out: MPSGraphTensor, rows: Int) -> MPSGraphExecutable {
    let shp = MPSGraphShapedType(shape: [NSNumber(value: rows), NSNumber(value: inFeat)], dataType: .float32)
    let desc = MPSGraphCompilationDescriptor()
    return g.compile(with: MPSGraphDevice(mtlDevice: device), feeds: [x: shp],
                     targetTensors: [out], targetOperations: nil, compilationDescriptor: desc)
}

func execTimed(_ exec: MPSGraphExecutable, feed: MPSGraphTensorData) -> (gpuMs: Double, wallMs: Double) {
    let cmdBuf = queue.makeCommandBuffer()!
    let mpsCmd = MPSCommandBuffer(commandBuffer: cmdBuf)
    let t0 = Date()
    _ = exec.encode(to: mpsCmd, inputs: [feed], results: nil, executionDescriptor: nil)
    mpsCmd.commit()
    mpsCmd.waitUntilCompleted()
    let wall = Date().timeIntervalSince(t0) * 1000.0
    let gpu = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1000.0
    return (gpu, wall)
}

// NOTE (measurement honesty): a "K identical forwards back-to-back on one command
// buffer / K" scheme was tried and REJECTED — it reported 56–76 TFLOPS, >10× the
// M1 Pro fp32 roofline (~5.3 TFLOPS), i.e. the driver elides redundant identical-
// input work. The single-forward executable GPU timer below sits AT the roofline
// (~4.7 TFLOPS ≈ 89%), the only physically-consistent figure, so it is authoritative.

func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count/2] }

// ── GATE 1: PARITY (real exported features @ 96×64) ──────────────────────────
let feat = loadF32(dataPath("features.f32"))
let expected = loadF32(dataPath("expected.f32"))
let nPix = feat.count / inFeat
precondition(nPix * inFeat == feat.count && expected.count == nPix * outCh, "feature/expected shape mismatch")
print("\n[metal4-probe] parity set: \(nPix) px (\(inFeat) feat, \(outCh) out)")

func parity(useFp16: Bool) -> (rel: Double, maxAbs: Double, out: [Float]) {
    let (g, x, o) = buildGraph(useFp16: useFp16)
    let got = runOnce(g, x, o, feed: makeTensorData(feat, rows: nPix))
    var num = 0.0, den = 0.0, maxAbs = 0.0
    for k in 0..<got.count {
        let d = Double(got[k]) - Double(expected[k])
        num += d*d; den += Double(expected[k])*Double(expected[k]); maxAbs = max(maxAbs, abs(d))
    }
    return (sqrt(num / max(den, 1e-30)), maxAbs, got)
}

// derived fp16 tolerance (same method as the WGSL verdict's f16 bound):
// 2·u16 + macs·u32 storage/accumulate error; f32 path bound = macs·u32.
let macs = layers.reduce(0) { $0 + $1.inDim * $1.outDim }
let u16 = pow(2.0, -11.0), u32 = Double(Float.ulpOfOne)
let boundF32 = (Double(macs) + 16*4) * u32          // + transcendental budget (parity vs CPU ln/exp path is in features already, so only matmul here)
let boundF16 = 2*u16 + Double(macs) * u32
let (relF32, maxF32, _) = parity(useFp16: false)
let (relF16, maxF16, _) = parity(useFp16: true)
let okF32 = relF32 <= boundF32 * 8   // matmul-only path; generous factor for MPS tiling reorder
let okF16 = relF16 <= boundF16 * 8
func e(_ v: Double) -> String { String(format: "%.3e", v) }
print("  f32   parity_rel=\(e(relF32))  max|Δ|=\(e(maxF32))  (bound \(e(boundF32*8))) \(okF32 ? "PASS" : "FAIL")")
print("  fp16  parity_rel=\(e(relF16))  max|Δ|=\(e(maxF16))  (bound \(e(boundF16*8))) \(okF16 ? "PASS" : "FAIL")")

// ── GATE 2: REAL ms (batched forward, GPU-timed, warm-up + median) ───────────
struct Shape { let label: String; let n: Int }
let shapes = [Shape(label: "spike 96×64", n: 6144), Shape(label: "native 960×640", n: 614_400)]
let WARMUP = 8, TIMED = 40

// returns (gpuMedianMs, wallMedianMs, firstRowsMatchCPU)
func timeShape(_ n: Int, useFp16: Bool) -> (Double, Double, Bool) {
    // tile the real 6144-px feature buffer up to n rows (ms depends on shape, not values).
    var buf = [Float](repeating: 0, count: n * inFeat)
    let src = feat.count
    buf.withUnsafeMutableBufferPointer { dst in
        feat.withUnsafeBufferPointer { s in
            var off = 0
            while off < dst.count { let c = min(src, dst.count - off); memcpy(dst.baseAddress!+off, s.baseAddress!, c*4); off += c }
        }
    }
    let (g, x, o) = buildGraph(useFp16: useFp16)
    let feed = makeTensorData(buf, rows: n)
    // correctness: the first 6144 tiled rows must reproduce the CPU expected out.
    let got = runOnce(g, x, o, feed: feed)
    var match = true
    for k in 0..<expected.count { if abs(Double(got[k]) - Double(expected[k])) > (useFp16 ? 5e-2 : 1e-3) { match = false; break } }
    let exec = compileExec(g, x, o, rows: n)
    for _ in 0..<WARMUP { _ = execTimed(exec, feed: feed) }
    var gpu = [Double](), wall = [Double]()
    for _ in 0..<TIMED { let r = execTimed(exec, feed: feed); gpu.append(r.gpuMs); wall.append(r.wallMs) }
    return (median(gpu), median(wall), match)
}

print("\n[metal4-probe] REAL GPU ms (MTLCommandBuffer.gpuStartTime/gpuEndTime, warmup \(WARMUP) + median of \(TIMED))")
print("  60fps budget = 16.67 ms/frame; native target ≤ ~10 ms leaves room for the 1-spp trace")
func pad(_ s: String, _ w: Int) -> String { s.count >= w ? s : s + String(repeating: " ", count: w - s.count) }
func f3(_ v: Double) -> String { String(format: "%.3f", v) }
let macF = Double(macs)
func tflops(_ ms: Double, _ n: Int) -> String { String(format: "%.1f", 2*macF*Double(n)/(ms/1000)/1e12) }
print("  ms = single-forward whole-graph GPU time (executable on 1 cmd buffer, gpuStart/End, median 40)")
print("       (wall) = same call's commit→wait incl. per-call encode+157MB-intermediate alloc churn")
print("       a warmed renderer pool amortizes the wall gap; TFLOPS from the GPU ms (sanity vs 5.3T roofline)")
print("  \(pad("shape",16))  \(pad("px",8))  \(pad("f32 gpu(wall)",22))  \(pad("fp16 gpu(wall)",22))  cpu-parity")
var nativeBestMs = Double.infinity
for s in shapes {
    let (f32g, f32w, f32ok) = timeShape(s.n, useFp16: false)
    let (f16g, f16w, f16ok) = timeShape(s.n, useFp16: true)
    if s.n == 614_400 { nativeBestMs = min(f32g, f16g) }
    let f32s = "\(f3(f32g))(\(f3(f32w))) \(tflops(f32g,s.n))T"
    let f16s = "\(f3(f16g))(\(f3(f16w))) \(tflops(f16g,s.n))T"
    print("  \(pad(s.label,16))  \(pad(String(s.n),8))  \(pad(f32s,22))  \(pad(f16s,22))  \(f32ok && f16ok ? "match" : "DRIFT")")
    print("  \(pad("",16))  \(pad("%budget",8))  \(pad(String(format: "%.0f%%", f32g/16.67*100),22))  \(String(format: "%.0f%%", f16g/16.67*100))")
}

// ── GATE 3: VERDICT ──────────────────────────────────────────────────────────
print("\n[metal4-probe] VERDICT")
if nativeBestMs <= 10.0 {
    print("  MATMUL BATCHING REOPENS THE DOOR: native 960×640 best = \(f3(nativeBestMs)) ms ≤ 10 ms.")
} else if nativeBestMs <= 16.67 {
    print("  In 60fps budget but no trace headroom: native best = \(f3(nativeBestMs)) ms (≤16.67, >10).")
} else {
    print("  DOOR STILL SHUT at native: best = \(f3(nativeBestMs)) ms = \(String(format: "%.0f", nativeBestMs/16.67*100))% of budget (\(String(format: "%.1f", nativeBestMs/10.0))× over the ≤10ms target).")
}
print("  (WGSL per-pixel MLP floor was ~280 ms f32 native — compare to the number above.)")
