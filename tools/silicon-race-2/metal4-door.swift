// Silicon Race II · MTL4MachineLearningCommandEncoder reachability probe.
// Metal-only surface: queue/compiler/tensor/encoder runtime + command-timeline
// composition control. Network dispatch remains gated on a Metal ML package.

import Foundation
import Metal

setbuf(stdout, nil)

@inline(__always) func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}
func f6(_ value: Double) -> String { String(format: "%.6f", value) }

@available(macOS 26.0, *)
func extents(_ values: [Int]) -> MTLTensorExtents {
    values.withUnsafeBufferPointer { pointer in
        MTLTensorExtents(__rank: pointer.count, values: pointer.baseAddress!)!
    }
}

@available(macOS 26.0, *)
func makeTensor(device: MTLDevice, shape: [Int], label: String) throws -> MTLTensor {
    let descriptor = MTLTensorDescriptor()
    descriptor.dimensions = extents(shape)
    descriptor.dataType = .float16
    descriptor.usage = [.machineLearning, .compute]
    descriptor.storageMode = .shared
    let tensor = try device.makeTensor(descriptor: descriptor)
    tensor.label = label
    return tensor
}

let dummySource = """
#include <metal_stdlib>
using namespace metal;
kernel void dummy(uint tid [[thread_position_in_grid]]) {
    thread volatile uint value = tid;
    value = value * 1664525u + 1013904223u;
}
"""

@available(macOS 26.0, *)
func negativeOrdinaryFunctionPipeline() throws {
    let device = MTLCreateSystemDefaultDevice()!
    let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
    let library = try device.makeLibrary(source: dummySource, options: nil)
    let functionDescriptor = MTL4LibraryFunctionDescriptor()
    functionDescriptor.library = library
    functionDescriptor.name = "dummy"
    let descriptor = MTL4MachineLearningPipelineDescriptor()
    descriptor.label = "ordinary-function-negative-control"
    descriptor.machineLearningFunctionDescriptor = functionDescriptor
    print("negative_control=compile_ordinary_metal_function_as_ml_pipeline")
    _ = try compiler.makeMachineLearningPipelineState(descriptor: descriptor)
    print("negative_control=UNEXPECTED_SUCCESS")
}

@available(macOS 26.0, *)
func run() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal unavailable") }
    print("host=\(device.name) os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
    print("sdk_surface=MTL4MachineLearningCommandEncoder MTL4MachineLearningPipelineState MTLTensor")

    let queueDescriptor = MTL4CommandQueueDescriptor()
    queueDescriptor.label = "silicon-race-2"
    let queue = try device.makeMTL4CommandQueue(descriptor: queueDescriptor)
    let allocatorDescriptor = MTL4CommandAllocatorDescriptor()
    allocatorDescriptor.label = "silicon-race-2"
    let allocator = try device.makeCommandAllocator(descriptor: allocatorDescriptor)
    let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
    let tableDescriptor = MTL4ArgumentTableDescriptor()
    tableDescriptor.maxBufferBindCount = 2
    tableDescriptor.initializeBindings = true
    let table = try device.makeArgumentTable(descriptor: tableDescriptor)
    print("runtime_objects=queue:ok allocator:ok compiler:ok argument_table:ok")

    let tensorShapes: [(String, [Int], [Int])] = [
        ("tiny-64", [64, 14], [64, 3]),
        ("tiny-4096", [4096, 14], [4096, 3]),
        ("pleroma-307200", [640 * 480, 23], [640 * 480, 3]),
    ]
    var retainedTensors = [MTLTensor]()
    for (label, inputShape, outputShape) in tensorShapes {
        let input = try makeTensor(device: device, shape: inputShape, label: "\(label)-input")
        let output = try makeTensor(device: device, shape: outputShape, label: "\(label)-output")
        retainedTensors += [input, output]
        print("tensor=\(label) input_alloc=\(input.allocatedSize) output_alloc=\(output.allocatedSize) usage=ml+compute")
    }

    let library = try device.makeLibrary(source: dummySource, options: nil)
    guard let function = library.makeFunction(name: "dummy") else { fatalError("dummy function absent") }
    let computePipeline = try device.makeComputePipelineState(function: function)
    print("ordinary_function_pipeline=isolated_negative_control flag=--ordinary-function-negative")

    // Empty ML encoder + dummy compute on one MTL4 command buffer. Control only:
    // proves runtime encoder creation/composition; NOT a network dispatch timing.
    var emptyEncodeUs = [Double]()
    var controlGpuMs = [Double]()
    var controlWallMs = [Double]()
    for iteration in 0..<40 {
        guard let commandBuffer = device.makeCommandBuffer() else { fatalError("MTL4 command buffer unavailable") }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        guard let compute = commandBuffer.makeComputeCommandEncoder() else { fatalError("MTL4 compute encoder unavailable") }
        compute.setComputePipelineState(computePipeline)
        compute.dispatchThreads(
            threadsPerGrid: MTLSize(width: 1024, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(128, computePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        compute.endEncoding()

        let encodeStart = nowNs()
        guard let mlEncoder = commandBuffer.makeMachineLearningCommandEncoder() else {
            fatalError("MTL4 ML encoder unavailable")
        }
        mlEncoder.endEncoding()
        let encodeEnd = nowNs()
        commandBuffer.endCommandBuffer()

        let completion = DispatchSemaphore(value: 0)
        let commitOptions = MTL4CommitOptions()
        var gpuMs = Double.nan
        var commitError: Error?
        commitOptions.addFeedbackHandler { feedback in
            gpuMs = (feedback.gpuEndTime - feedback.gpuStartTime) * 1000
            commitError = feedback.error
            completion.signal()
        }
        let wallStart = nowNs()
        queue.commit([commandBuffer], options: commitOptions)
        guard completion.wait(timeout: .now() + 5) == .success else { fatalError("MTL4 feedback timeout") }
        let wallEnd = nowNs()
        if let commitError { throw commitError }
        if iteration >= 8 {
            emptyEncodeUs.append(Double(encodeEnd - encodeStart) / 1e3)
            controlGpuMs.append(gpuMs)
            controlWallMs.append(Double(wallEnd - wallStart) / 1e6)
        }
        allocator.reset()
    }
    print("empty_encoder_control=median32 encode_us=\(f6(median(emptyEncodeUs))) gpu_timeline_ms=\(f6(median(controlGpuMs))) commit_wait_ms=\(f6(median(controlWallMs)))")
    print("network_dispatch=UNVERIFIED reason=no_metal_ml_package")
    print("network_encode_cpu=UNVERIFIED")
    print("network_gpu_timeline_sync=UNVERIFIED")
    print("network_execution_locus=UNVERIFIED")
    print("retained_tensors=\(retainedTensors.count) table_device=\(table.device.name)")
}

if #available(macOS 26.0, *) {
    do {
        if CommandLine.arguments.contains("--ordinary-function-negative") {
            try negativeOrdinaryFunctionPipeline()
        } else {
            try run()
        }
    } catch {
        print("fatal=\(error)")
        exit(1)
    }
} else {
    print("fatal=requires_macos_26")
    exit(1)
}
