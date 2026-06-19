// MLX (Apple GPU) embedding benchmark — the "real GPU path".
//
// Runs a sentence-transformer (default MiniLM-L6, same model as the CPU
// baseline) on the Apple GPU via MLXEmbedders, with large fixed batches so the
// matmuls are big enough for the GPU to win. Reads the same corpus the Rust rig
// and afm_native use.
//
// Build:  swift build -c release   (in bench/swift-mlx)
// Run:    .build/release/mlx-embed-bench --corpus ../corpus/exe.bin --model minilm_l6 --batch 256 --limit 20000

import Foundation
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import HuggingFace
import Tokenizers

// ---- args ----------------------------------------------------------------
var corpusPath = "../corpus/exe.bin"
var modelName = "minilm_l6"
var batch = 256
var limit = 0
var budgetGB = 16.0
var sortByLen = true
var dumpPath: String? = nil

// Numerical sanity check — catches a mismatched metallib computing wrong math.
if CommandLine.arguments.contains("--selftest") {
    let a = MLXArray(converting: [1.0, 2, 3, 4], [2, 2])
    let id = MLXArray(converting: [1.0, 0, 0, 1], [2, 2])
    let c = matmul(a, id); c.eval()
    print("identity matmul (expect 1,2,3,4):", c.asArray(Float.self))
    let s = sum(MLXArray(converting: [1.0, 2, 3, 4])); s.eval()
    print("sum 1..4 (expect 10):", s.item(Float.self))
    let m = MLXArray(converting: [3.0, 4], [2])
    let nrm = sqrt(sum(m * m)); nrm.eval()
    print("norm([3,4]) (expect 5):", nrm.item(Float.self))
    exit(0)
}

var argi = CommandLine.arguments.dropFirst().makeIterator()
while let a = argi.next() {
    switch a {
    case "--corpus": corpusPath = argi.next() ?? corpusPath
    case "--model": modelName = argi.next() ?? modelName
    case "--batch": batch = Int(argi.next() ?? "256") ?? 256
    case "--limit": limit = Int(argi.next() ?? "0") ?? 0
    case "--budget-gb": budgetGB = Double(argi.next() ?? "16") ?? 16
    case "--no-sort": sortByLen = false
    case "--dump": dumpPath = argi.next()
    default: FileHandle.standardError.write("unknown arg \(a)\n".data(using: .utf8)!)
    }
}

// ---- RSS + watchdog ------------------------------------------------------
func rssBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.resident_size : 0
}
final class Atomic<T>: @unchecked Sendable { var v: T; let l = NSLock(); init(_ x: T){v=x}
    func get() -> T { l.lock(); defer{l.unlock()}; return v }
    func set(_ x: T){ l.lock(); v=x; l.unlock() }
    func bump(_ x: UInt64) where T == UInt64 { l.lock(); if x>v {v=x}; l.unlock() } }
let peakRSS = Atomic<UInt64>(0)
let stopWatch = Atomic<Bool>(false)
let budgetBytes = UInt64(budgetGB * 1e9)
Thread.detachNewThread {
    while !stopWatch.get() {
        let r = rssBytes(); peakRSS.bump(r)
        if r > budgetBytes {
            FileHandle.standardError.write("[watchdog] RSS \(Double(r)/1e9) GB > budget — aborting\n".data(using: .utf8)!)
            abort()
        }
        usleep(150_000)
    }
}

// ---- corpus --------------------------------------------------------------
func readCorpus(_ path: String) -> [String] {
    guard let data = FileManager.default.contents(atPath: path) else {
        FileHandle.standardError.write("cannot read corpus \(path)\n".data(using: .utf8)!); exit(1)
    }
    var texts: [String] = []
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        let base = raw.baseAddress!
        var off = 8 // skip magic
        let n = Int(base.load(fromByteOffset: off, as: UInt64.self)); off += 8
        texts.reserveCapacity(n)
        for _ in 0..<n {
            let len = Int(base.load(fromByteOffset: off, as: UInt32.self)); off += 4
            texts.append(String(decoding: UnsafeRawBufferPointer(start: base + off, count: len), as: UTF8.self))
            off += len
        }
    }
    return texts
}

func config(_ name: String) -> ModelConfiguration {
    switch name {
    case "minilm_l6", "minilm": return EmbedderRegistry.minilm_l6
    case "bge_small", "bge-small": return EmbedderRegistry.bge_small
    case "bge_base", "bge-base": return EmbedderRegistry.bge_base
    case "gte_tiny": return EmbedderRegistry.gte_tiny
    case "snowflake_xs": return EmbedderRegistry.snowflake_xs
    default: return EmbedderRegistry.minilm_l6
    }
}

var texts = readCorpus(corpusPath)
if limit > 0 && limit < texts.count { texts = Array(texts[0..<limit]) }
if sortByLen { texts.sort { $0.utf8.count < $1.utf8.count } }
let n = texts.count
FileHandle.standardError.write("corpus: \(n) texts, model=\(modelName), batch=\(batch)\n".data(using: .utf8)!)

// Bound GPU cache so we stay within budget.
MLX.GPU.set(cacheLimit: 2 * 1024 * 1024 * 1024)

// ---- load model ----------------------------------------------------------
let tInit = Date()
let container = try await EmbedderModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: config(modelName)
) { _ in }
let initS = Date().timeIntervalSince(tInit)
FileHandle.standardError.write(String(format: "init: %.2fs\n", initS).data(using: .utf8)!)

// Semantic sanity: similar sentences should score higher than dissimilar ones.
if CommandLine.arguments.contains("--semtest") {
    let probes = [
        "the cat sat on the mat",
        "a kitten is resting on a rug",
        "quantum chromodynamics and gluon fields",
    ]
    let vs: [[Float]] = await container.perform { (model, tokenizer, pooling) in
        let ids = probes.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
        let maxLen = ids.reduce(into: 1) { $0 = max($0, $1.count) }
        let pad = tokenizer.eosTokenId ?? 0
        let padded = stacked(ids.map { MLXArray($0 + Array(repeating: pad, count: maxLen - $0.count)) })
        let mask = padded .!= MLXArray(pad)
        let tt = MLXArray.zeros(like: padded)
        let p = pooling(model(padded, positionIds: nil, tokenTypeIds: tt, attentionMask: mask),
                        mask: mask, normalize: true, applyLayerNorm: false)
        p.eval()
        return p.map { $0.asArray(Float.self) }
    }
    func cos(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }
    print("eosTokenId-based pad. probe count:", vs.count, "dim:", vs.first?.count ?? 0)
    print(String(format: "cos(cat,kitten)=%.3f  cos(cat,quantum)=%.3f  cos(kitten,quantum)=%.3f",
                 cos(vs[0], vs[1]), cos(vs[0], vs[2]), cos(vs[1], vs[2])))
    exit(0)
}

// ---- embed in batches on the GPU -----------------------------------------
var dim = 0
let dumpN = 256
var dumpVecs: [[Float]] = []
let t0 = Date()
var done = 0
var batchStart = 0
while batchStart < n {
    let end = min(batchStart + batch, n)
    let slice = Array(texts[batchStart..<end])
    let vecs: [[Float]] = await container.perform { (model, tokenizer, pooling) in
        let ids = slice.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
        let maxLen = ids.reduce(into: 1) { acc, e in acc = max(acc, e.count) }
        let pad = tokenizer.eosTokenId ?? 0
        let padded = stacked(ids.map { e in
            MLXArray(e + Array(repeating: pad, count: maxLen - e.count))
        })
        let mask = padded .!= MLXArray(pad)
        let tokenTypes = MLXArray.zeros(like: padded)
        let pooled = pooling(
            model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
            mask: mask, normalize: true, applyLayerNorm: false
        )
        pooled.eval()
        return pooled.map { $0.asArray(Float.self) }
    }
    if dim == 0 { dim = vecs.first?.count ?? 0 }
    if dumpPath != nil && dumpVecs.count < dumpN {
        for v in vecs where dumpVecs.count < dumpN { dumpVecs.append(v) }
    }
    done += vecs.count
    batchStart = end
}
let secs = Date().timeIntervalSince(t0)
stopWatch.set(true)

// Optional dump for correctness comparison (Rust bench --compare format:
// dim u32, n u32, then n*dim f32 little-endian).
if let dp = dumpPath {
    var out = Data()
    var d32 = UInt32(dim).littleEndian; var n32 = UInt32(dumpVecs.count).littleEndian
    withUnsafeBytes(of: &d32) { out.append(contentsOf: $0) }
    withUnsafeBytes(of: &n32) { out.append(contentsOf: $0) }
    for v in dumpVecs {
        for f in v { var le = f.bitPattern.littleEndian; withUnsafeBytes(of: &le) { out.append(contentsOf: $0) } }
    }
    try? out.write(to: URL(fileURLWithPath: dp))
    FileHandle.standardError.write("dumped \(dumpVecs.count) vectors -> \(dp)\n".data(using: .utf8)!)
}

let rate = Double(n) / secs
let peakGB = Double(peakRSS.get()) / 1e9
print("──────────────────────────────────────────────")
print("backend     : mlx-metal \(modelName)")
print("dim         : \(dim)")
print("texts       : \(n)")
print(String(format: "embed time  : %.2f s", secs))
print(String(format: "throughput  : %.0f chunks/s", rate))
print(String(format: "init time   : %.2f s", initS))
print(String(format: "peak RSS    : %.2f GB", peakGB))
print("──────────────────────────────────────────────")
