// Real silicon attribution via MLComputePlan (macOS 14.4+).
// Prints, per op, the device CoreML PLANS to run it on + supported devices.
import Foundation
import CoreML
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1])

@available(macOS 14.4, *)
func dump() async {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .all
    guard let plan = try? await MLComputePlan.load(contentsOf: modelURL, configuration: cfg) else {
        print("compute plan unavailable"); return
    }
    guard case let .program(program) = plan.modelStructure else { print("not a program"); return }
    var counts: [String:Int] = [:]
    for (_, fn) in program.functions {
        func walk(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                if let du = plan.deviceUsage(for: op) {
                    let d = String(describing: du.preferred)
                    counts[op.operatorName + " -> " + d, default: 0] += 1
                }
                for b in op.blocks { walk(b) }
            }
        }
        walk(fn.block)
    }
    for (k,v) in counts.sorted(by: {$0.key < $1.key}) { print(String(format:"%4d  %@", v, k)) }
}

if #available(macOS 14.4, *) {
    let sem = DispatchSemaphore(value: 0)
    Task { await dump(); sem.signal() }
    sem.wait()
} else { print("needs macOS 14.4+") }
