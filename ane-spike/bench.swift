// R-Direct ANE per-frame latency harness.
// Loads the compiled .mlmodelc, times end-to-end predict() (INCLUDING input
// MLMultiArray marshal) across compute-unit modes x pixel-batch shapes.
// Warm-up + median. Also parity-checks against golden.json (Rust f32).
import Foundation
import CoreML

func die(_ s: String) -> Never { FileHandle.standardError.write((s+"\n").data(using:.utf8)!); exit(1) }

let args = CommandLine.arguments
let modelURL = URL(fileURLWithPath: args[1])   // rdirect.mlmodelc
let goldenPath = args[2]                         // golden.json
let IN = 23, OUTC = 3

// ---- load golden (features + expected) ----
struct Golden: Decodable { let features: [[Float]]; let outputs: [[Float]] }
let golden = try! JSONDecoder().decode(Golden.self, from: try! Data(contentsOf: URL(fileURLWithPath: goldenPath)))
let gN = golden.features.count

func makeInput(_ n: Int, seedRows: [[Float]]? = nil) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [NSNumber(value:n), NSNumber(value:IN)], dataType: .float32)
    let p = arr.dataPointer.bindMemory(to: Float.self, capacity: n*IN)
    if let rows = seedRows {
        for i in 0..<n { for j in 0..<IN { p[i*IN+j] = rows[i % rows.count][j] } }
    } else {
        // deterministic filler in a plausible range
        var s: UInt64 = 0x1234
        for k in 0..<(n*IN) { s = s &* 6364136223846793005 &+ 1442695040888963407
            p[k] = Float((s >> 40) & 0xFFFF) / 65535.0 * 2.0 }
    }
    return arr
}

func loadModel(_ units: MLComputeUnits) -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units
    return try! MLModel(contentsOf: modelURL, configuration: cfg)
}

func outName(_ m: MLModel) -> String { m.modelDescription.outputDescriptionsByName.keys.first! }

func predictOnce(_ m: MLModel, _ x: MLMultiArray) -> MLFeatureProvider {
    let fp = try! MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: x)])
    return try! m.prediction(from: fp)
}

// ---- parity: run golden features through .all model, compare ----
func parity() {
    let m = loadModel(.all)
    let x = makeInput(gN, seedRows: golden.features)
    let out = predictOnce(m, x).featureValue(for: outName(m))!.multiArrayValue!
    let op = out.dataPointer.bindMemory(to: Float.self, capacity: gN*OUTC)
    var maxAbs: Float = 0, sumAbs: Float = 0, maxRel: Float = 0
    for i in 0..<gN { for c in 0..<OUTC {
        let e = abs(op[i*OUTC+c] - golden.outputs[i][c]); maxAbs = max(maxAbs,e); sumAbs += e
        maxRel = max(maxRel, e / (abs(golden.outputs[i][c]) + 1e-4)) } }
    let pass = maxRel < 2e-2 || maxAbs < 5e-3
    print(String(format:"PARITY(.all) N=%d maxAbs=%.6f meanAbs=%.6f maxRel=%.6f -> %@",
                 gN, maxAbs, sumAbs/Float(gN*OUTC), maxRel, pass ? "PASS":"FAIL"))
}

func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count/2] }

func bench(_ label: String, _ units: MLComputeUnits, _ n: Int, warm: Int, iters: Int) {
    let m = loadModel(units)
    let name = outName(m)
    // warm-up (includes first-run ANE/GPU pipeline compile & load)
    for _ in 0..<warm { _ = predictOnce(m, makeInput(n)) }
    var ts: [Double] = []
    for _ in 0..<iters {
        let x = makeInput(n)                 // fresh marshal each iter (honest per-frame cost)
        let t0 = DispatchTime.now().uptimeNanoseconds
        let out = predictOnce(m, x)
        _ = out.featureValue(for: name)!.multiArrayValue!  // force realize
        let t1 = DispatchTime.now().uptimeNanoseconds
        ts.append(Double(t1 - t0)/1e6)
    }
    let med = median(ts), mn = ts.min()!, mx = ts.max()!
    print(String(format:"%-22@ N=%7d  median=%8.3f ms  min=%8.3f  max=%8.3f  (warm=%d iters=%d)",
                 label, n, med, mn, mx, warm, iters))
}

parity()
print("--- per-frame latency (end-to-end predict incl. marshal) ---")
let shapes = [6144, 614400]     // 96x64 and 960x640
let modes: [(String, MLComputeUnits)] = [
    ("cpuOnly", .cpuOnly), ("cpuAndNeuralEngine", .cpuAndNeuralEngine), ("all", .all)
]
for n in shapes {
    let iters = n >= 100000 ? 30 : 100
    for (lbl, u) in modes { bench(lbl, u, n, warm: 8, iters: iters) }
}
