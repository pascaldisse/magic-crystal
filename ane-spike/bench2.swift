// R-Direct fixed-shape ANE bench: separates MARSHAL from pure PREDICT.
// usage: bench2 <model.mlmodelc> <N>
import Foundation
import CoreML
let args = CommandLine.arguments
let modelURL = URL(fileURLWithPath: args[1])
let N = Int(args[2])!
let IN = 23

func marshal(_ n: Int) -> MLMultiArray {
    let arr = try! MLMultiArray(shape: [NSNumber(value:n), NSNumber(value:IN)], dataType: .float32)
    let p = arr.dataPointer.bindMemory(to: Float.self, capacity: n*IN)
    var s: UInt64 = 0x1234
    for k in 0..<(n*IN) { s = s &* 6364136223846793005 &+ 1442695040888963407
        p[k] = Float((s >> 40) & 0xFFFF) / 65535.0 * 2.0 }
    return arr
}
func loadModel(_ u: MLComputeUnits) -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = u
    return try! MLModel(contentsOf: modelURL, configuration: cfg)
}
func median(_ v: [Double]) -> Double { let s=v.sorted(); return s[s.count/2] }

// marshal cost alone
do {
    var ts:[Double]=[]; for _ in 0..<20 { let t0=DispatchTime.now().uptimeNanoseconds; _=marshal(N)
        ts.append(Double(DispatchTime.now().uptimeNanoseconds-t0)/1e6) }
    print(String(format:"marshal(fill %d x %d)  median=%.3f ms", N, IN, median(ts)))
}

let modes:[(String,MLComputeUnits)]=[("cpuOnly",.cpuOnly),("cpuAndNeuralEngine",.cpuAndNeuralEngine),("all",.all)]
let x = marshal(N)                       // prebuilt, REUSED (pure predict cost)
let iters = N >= 100000 ? 40 : 200
for (lbl,u) in modes {
    let m = loadModel(u)
    let name = m.modelDescription.outputDescriptionsByName.keys.first!
    let fp = try! MLDictionaryFeatureProvider(dictionary:["x":MLFeatureValue(multiArray:x)])
    for _ in 0..<10 { _ = try! m.prediction(from: fp) }   // warm
    var ts:[Double]=[]
    for _ in 0..<iters {
        let t0=DispatchTime.now().uptimeNanoseconds
        let out = try! m.prediction(from: fp)
        _ = out.featureValue(for:name)!.multiArrayValue!
        ts.append(Double(DispatchTime.now().uptimeNanoseconds-t0)/1e6)
    }
    print(String(format:"%-20@ N=%7d predict-only median=%8.3f ms  min=%8.3f (iters=%d)",
                 lbl, N, median(ts), ts.min()!, iters))
}
