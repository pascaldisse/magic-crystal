import Foundation
import CoreML
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])
@available(macOS 14.4, *)
func dump() async {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
    guard let plan = try? await MLComputePlan.load(contentsOf: modelURL, configuration: cfg) else { print("unavail"); return }
    guard case let .program(p) = plan.modelStructure else { return }
    var counts:[String:Int]=[:]
    for (_,fn) in p.functions { for op in fn.block.operations {
        if let du = plan.deviceUsage(for: op) { counts[op.operatorName+" -> "+String(describing: du.preferred), default:0]+=1 } } }
    for (k,v) in counts.sorted(by:{$0.key<$1.key}) { print(String(format:"%4d  %@",v,k)) }
}
if #available(macOS 14.4, *) { let s=DispatchSemaphore(value:0); Task{ await dump(); s.signal() }; s.wait() }
