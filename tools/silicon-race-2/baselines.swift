// Silicon Race II · legal baselines: Accelerate CPU + MPSGraph GPU.
// Shapes: tiny 14→32→32→3 @64…4096; Pleroma 23→5×64→3 @640×480.
// Timing: warm state → single forward/call → median. GPU = command-buffer
// gpuStartTime/endTime; wall = encode→commit→wait; encode timed separately.

import Accelerate
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

setbuf(stdout, nil)

struct Layer {
    let inDim: Int
    let outDim: Int
    let weights: [Float] // [out,in]
    let bias: [Float]
}

struct Shape {
    let label: String
    let batch: Int
    let dims: [Int]
}

struct LCG {
    var state: UInt64
    mutating func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let u = Float((state >> 40) & 0xffff) / Float(0xffff)
        return (u * 2 - 1)
    }
}

func makeLayers(_ dims: [Int], seed: UInt64) -> [Layer] {
    var rng = LCG(state: seed)
    return (0..<(dims.count - 1)).map { i in
        let input = dims[i], output = dims[i + 1]
        let scale = Float(0.25 / sqrt(Double(input)))
        return Layer(
            inDim: input,
            outDim: output,
            weights: (0..<(input * output)).map { _ in rng.next() * scale },
            bias: (0..<output).map { _ in rng.next() * 0.01 }
        )
    }
}

func makeInput(batch: Int, width: Int, seed: UInt64) -> [Float] {
    var rng = LCG(state: seed)
    return (0..<(batch * width)).map { _ in rng.next() }
}

@inline(__always) func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}
func f6(_ value: Double) -> String { String(format: "%.6f", value) }

// Reused activations; SGEMM + fused-by-loop bias/ReLU reference.
final class CPUForward {
    let batch: Int
    let layers: [Layer]
    let input: [Float]
    var a: [Float]
    var b: [Float]

    init(batch: Int, layers: [Layer], input: [Float]) {
        self.batch = batch
        self.layers = layers
        self.input = input
        let width = max(layers.map(\.outDim).max()!, layers[0].inDim)
        self.a = [Float](repeating: 0, count: batch * width)
        self.b = [Float](repeating: 0, count: batch * width)
    }

    @inline(never) func run() -> Float {
        for (index, layer) in layers.enumerated() {
            let source = index == 0 ? input : a
            source.withUnsafeBufferPointer { x in
                b.withUnsafeMutableBufferPointer { y in
                    layer.weights.withUnsafeBufferPointer { w in
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int32(batch), Int32(layer.outDim), Int32(layer.inDim),
                            1, x.baseAddress!, Int32(layer.inDim),
                            w.baseAddress!, Int32(layer.inDim),
                            0, y.baseAddress!, Int32(layer.outDim)
                        )
                    }
                }
            }
            let hidden = index != layers.count - 1
            for row in 0..<batch {
                let base = row * layer.outDim
                for column in 0..<layer.outDim {
                    let value = b[base + column] + layer.bias[column]
                    b[base + column] = hidden ? max(0, value) : value
                }
            }
            swap(&a, &b)
        }
        return a[0] + a[(batch - 1) * layers.last!.outDim + layers.last!.outDim - 1]
    }
}

func float32to16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    var exponent = Int((bits >> 23) & 0xff) - 127 + 15
    var mantissa = bits & 0x7fffff
    if exponent <= 0 { return sign }
    if exponent >= 31 { return sign | 0x7c00 }
    mantissa += 0x1000
    if mantissa & 0x800000 != 0 {
        mantissa = 0
        exponent += 1
        if exponent >= 31 { return sign | 0x7c00 }
    }
    return sign | UInt16(exponent << 10) | UInt16((mantissa >> 13) & 0x3ff)
}

func graphConstant(_ graph: MPSGraph, values: [Float], shape: [NSNumber]) -> MPSGraphTensor {
    let fp16 = values.map(float32to16)
    let data = fp16.withUnsafeBytes { Data($0) }
    return graph.constant(data, shape: shape, dataType: .float16)
}

func buildGraph(layers: [Layer], inputWidth: Int) -> (MPSGraph, MPSGraphTensor, MPSGraphTensor) {
    let graph = MPSGraph()
    let input = graph.placeholder(shape: [-1, NSNumber(value: inputWidth)], dataType: .float32, name: "input")
    var value = graph.cast(input, to: .float16, name: "input_fp16")
    for (index, layer) in layers.enumerated() {
        var transposed = [Float](repeating: 0, count: layer.inDim * layer.outDim)
        for output in 0..<layer.outDim {
            for input in 0..<layer.inDim {
                transposed[input * layer.outDim + output] = layer.weights[output * layer.inDim + input]
            }
        }
        let weights = graphConstant(graph, values: transposed, shape: [NSNumber(value: layer.inDim), NSNumber(value: layer.outDim)])
        let bias = graphConstant(graph, values: layer.bias, shape: [1, NSNumber(value: layer.outDim)])
        value = graph.matrixMultiplication(primary: value, secondary: weights, name: "matmul_\(index)")
        value = graph.addition(value, bias, name: "bias_\(index)")
        if index != layers.count - 1 { value = graph.reLU(with: value, name: "relu_\(index)") }
    }
    return (graph, input, graph.cast(value, to: .float32, name: "output_fp32"))
}

func makeTensorData(device: MTLDevice, input: [Float], batch: Int, width: Int) -> MPSGraphTensorData {
    let buffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.size, options: .storageModeShared)!
    return MPSGraphTensorData(buffer, shape: [NSNumber(value: batch), NSNumber(value: width)], dataType: .float32)
}

struct GPUTiming {
    let encodeMs: Double
    let gpuMs: Double
    let wallMs: Double
}

func runGPU(
    queue: MTLCommandQueue,
    executable: MPSGraphExecutable,
    feed: MPSGraphTensorData
) -> GPUTiming {
    let commandBuffer = queue.makeCommandBuffer()!
    let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
    let wallStart = nowNs()
    let encodeStart = nowNs()
    _ = executable.encode(to: mpsCommandBuffer, inputs: [feed], results: nil, executionDescriptor: nil)
    let encodeEnd = nowNs()
    mpsCommandBuffer.commit()
    mpsCommandBuffer.waitUntilCompleted()
    let wallEnd = nowNs()
    return GPUTiming(
        encodeMs: Double(encodeEnd - encodeStart) / 1e6,
        gpuMs: (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000,
        wallMs: Double(wallEnd - wallStart) / 1e6
    )
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal unavailable")
}

let shapes: [Shape] = [
    Shape(label: "tiny", batch: 64, dims: [14, 32, 32, 3]),
    Shape(label: "tiny", batch: 128, dims: [14, 32, 32, 3]),
    Shape(label: "tiny", batch: 256, dims: [14, 32, 32, 3]),
    Shape(label: "tiny", batch: 512, dims: [14, 32, 32, 3]),
    Shape(label: "tiny", batch: 1024, dims: [14, 32, 32, 3]),
    Shape(label: "tiny", batch: 2048, dims: [14, 32, 32, 3]),
    Shape(label: "tiny", batch: 4096, dims: [14, 32, 32, 3]),
    Shape(label: "pleroma", batch: 640 * 480, dims: [23, 64, 64, 64, 64, 64, 3]),
]

print("host=\(device.name) os=\(ProcessInfo.processInfo.operatingSystemVersionString)")
print("method=warm_single_forward_median gpu=f16_storage_f32_io cpu=f32_accelerate")
print("CSV,label,batch,cpu_ms,cpu_bodies_s,gpu_encode_ms,gpu_timeline_ms,gpu_wall_ms,gpu_sync_submit_ms,gpu_wall_bodies_s,parity_max_abs")

var checksum: Float = 0
for shape in shapes {
    let seed = shape.label == "tiny" ? UInt64(0x51a7) : UInt64(0x91e4)
    let layers = makeLayers(shape.dims, seed: seed)
    let input = makeInput(batch: shape.batch, width: shape.dims[0], seed: seed ^ UInt64(shape.batch))

    let cpu = CPUForward(batch: shape.batch, layers: layers, input: input)
    let cpuWarm = shape.label == "tiny" ? 20 : 2
    let cpuTimed = shape.label == "tiny" ? 200 : 9
    for _ in 0..<cpuWarm { checksum += cpu.run() }
    var cpuSamples = [Double]()
    for _ in 0..<cpuTimed {
        let start = nowNs()
        checksum += cpu.run()
        cpuSamples.append(Double(nowNs() - start) / 1e6)
    }
    let cpuMs = median(cpuSamples)

    let (graph, inputTensor, outputTensor) = buildGraph(layers: layers, inputWidth: shape.dims[0])
    let feed = makeTensorData(device: device, input: input, batch: shape.batch, width: shape.dims[0])
    let shapedInput = MPSGraphShapedType(
        shape: [NSNumber(value: shape.batch), NSNumber(value: shape.dims[0])],
        dataType: .float32
    )
    let compilation = MPSGraphCompilationDescriptor()
    let executable = graph.compile(
        with: MPSGraphDevice(mtlDevice: device),
        feeds: [inputTensor: shapedInput],
        targetTensors: [outputTensor],
        targetOperations: nil,
        compilationDescriptor: compilation
    )

    // Correctness → compare full GPU result against current CPU output.
    let result = graph.run(with: queue, feeds: [inputTensor: feed], targetTensors: [outputTensor], targetOperations: nil)[outputTensor]!
    let count = shape.batch * shape.dims.last!
    var gpuOutput = [Float](repeating: 0, count: count)
    result.mpsndarray().readBytes(&gpuOutput, strideBytes: nil)
    var parityMaxAbs = 0.0
    for index in 0..<count { parityMaxAbs = max(parityMaxAbs, abs(Double(gpuOutput[index] - cpu.a[index]))) }

    let gpuWarm = shape.label == "tiny" ? 20 : 6
    let gpuTimed = shape.label == "tiny" ? 200 : 30
    for _ in 0..<gpuWarm { _ = runGPU(queue: queue, executable: executable, feed: feed) }
    var encodeSamples = [Double](), gpuSamples = [Double](), wallSamples = [Double]()
    for _ in 0..<gpuTimed {
        let timing = runGPU(queue: queue, executable: executable, feed: feed)
        encodeSamples.append(timing.encodeMs)
        gpuSamples.append(timing.gpuMs)
        wallSamples.append(timing.wallMs)
    }
    let encodeMs = median(encodeSamples)
    let gpuMs = median(gpuSamples)
    let wallMs = median(wallSamples)
    let syncSubmitMs = max(0, wallMs - encodeMs - gpuMs)

    print([
        "CSV", shape.label, String(shape.batch), f6(cpuMs), f6(Double(shape.batch) * 1000 / cpuMs),
        f6(encodeMs), f6(gpuMs), f6(wallMs), f6(syncSubmitMs),
        f6(Double(shape.batch) * 1000 / wallMs), f6(parityMaxAbs)
    ].joined(separator: ","))
}
print("checksum=\(checksum)")
