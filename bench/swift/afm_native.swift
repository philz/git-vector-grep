// Native parallel benchmark for Apple's NLContextualEmbedding — the same
// library MacLocalAPI's `afm embed` server uses, but called directly with one
// instance per worker thread (no actor serialization, no HTTP).
//
// Mirrors NLContextualEmbeddingBackend: embeddingResult(for:) per text, mean
// pool over token vectors, L2 normalize. Reads the same corpus the Rust rig
// uses (bench/corpus/exe.bin).
//
// Build:  swiftc -O bench/swift/afm_native.swift -o target/afm_native
// Run:    target/afm_native --corpus bench/corpus/exe.bin --parallel 8 --limit 20000

import Foundation
import NaturalLanguage

// ---- args ----------------------------------------------------------------
var corpusPath = "bench/corpus/exe.bin"
var parallel = 8
var limit = 0
var budgetGB = 16.0
var sortByLen = true

var it = CommandLine.arguments.dropFirst().makeIterator()
while let a = it.next() {
    switch a {
    case "--corpus": corpusPath = it.next() ?? corpusPath
    case "--parallel": parallel = Int(it.next() ?? "8") ?? 8
    case "--limit": limit = Int(it.next() ?? "0") ?? 0
    case "--budget-gb": budgetGB = Double(it.next() ?? "16") ?? 16
    case "--no-sort": sortByLen = false
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

let peakRSS = ManagedAtomicU64(0)
let stopWatch = ManagedAtomicBool(false)
let budgetBytes = UInt64(budgetGB * 1e9)
final class ManagedAtomicU64 { var v: UInt64; let lock = NSLock(); init(_ x: UInt64){v=x}
    func max(_ x: UInt64){ lock.lock(); if x>v {v=x}; lock.unlock() }
    func fetchAdd(_ d: UInt64 = 1) -> UInt64 { lock.lock(); defer{lock.unlock()}; let old=v; v+=d; return old }
    func get() -> UInt64 { lock.lock(); defer{lock.unlock()}; return v } }
final class ManagedAtomicBool { var v: Bool; let lock = NSLock(); init(_ x: Bool){v=x}
    func set(_ x: Bool){ lock.lock(); v=x; lock.unlock() }
    func get() -> Bool { lock.lock(); defer{lock.unlock()}; return v } }

let watchdog = Thread {
    while !stopWatch.get() {
        let r = rssBytes()
        peakRSS.max(r)
        if r > budgetBytes {
            FileHandle.standardError.write("[watchdog] RSS \(Double(r)/1e9) GB > budget \(budgetGB) GB — aborting\n".data(using: .utf8)!)
            abort()
        }
        usleep(150_000)
    }
}
watchdog.start()

// ---- read corpus ---------------------------------------------------------
func readCorpus(_ path: String) -> [String] {
    guard let data = FileManager.default.contents(atPath: path) else {
        FileHandle.standardError.write("cannot read corpus \(path)\n".data(using: .utf8)!); exit(1)
    }
    var texts: [String] = []
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        let base = raw.baseAddress!
        var off = 0
        // magic[8]
        off += 8
        func readU32(_ o: Int) -> UInt32 { base.load(fromByteOffset: o, as: UInt32.self) }
        func readU64(_ o: Int) -> UInt64 { base.load(fromByteOffset: o, as: UInt64.self) }
        let n = Int(readU64(off)); off += 8
        texts.reserveCapacity(n)
        for _ in 0..<n {
            let len = Int(readU32(off)); off += 4
            let s = String(decoding: UnsafeRawBufferPointer(start: base + off, count: len), as: UTF8.self)
            texts.append(s)
            off += len
        }
    }
    return texts
}

var texts = readCorpus(corpusPath)
if limit > 0 && limit < texts.count { texts = Array(texts[0..<limit]) }
if sortByLen { texts.sort { $0.utf8.count < $1.utf8.count } }
let n = texts.count
FileHandle.standardError.write("corpus: \(n) texts, parallel=\(parallel)\n".data(using: .utf8)!)

// ---- build one embedding instance per worker -----------------------------
func makeEmbedding() -> NLContextualEmbedding {
    guard let e = NLContextualEmbedding(language: .english) else {
        FileHandle.standardError.write("failed to create NLContextualEmbedding\n".data(using: .utf8)!); exit(1)
    }
    do { try e.load() } catch {
        FileHandle.standardError.write("load failed: \(error)\n".data(using: .utf8)!); exit(1)
    }
    return e
}

let tInit = Date()
var instances: [NLContextualEmbedding] = []
for _ in 0..<parallel { instances.append(makeEmbedding()) }
let dim = Int(instances[0].dimension)
let initS = Date().timeIntervalSince(tInit)

func embedOne(_ e: NLContextualEmbedding, _ s: String, _ out: inout [Float]) {
    // empty/whitespace guard (framework rejects empties)
    if s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        for i in 0..<dim { out[i] = 0 }; return
    }
    guard let r = try? e.embeddingResult(for: s, language: .english) else {
        for i in 0..<dim { out[i] = 0 }; return
    }
    var sum = [Float](repeating: 0, count: dim)
    var cnt = 0
    let full = r.string.startIndex..<r.string.endIndex
    r.enumerateTokenVectors(in: full) { vec, _ in
        if vec.count == dim {
            for i in 0..<dim { sum[i] += Float(vec[i]) }
            cnt += 1
        }
        return true
    }
    if cnt > 0 {
        let scale = Float(cnt)
        var norm: Float = 0
        for i in 0..<dim { let v = sum[i] / scale; out[i] = v; norm += v*v }
        if norm > 0 { let inv = 1.0 / norm.squareRoot(); for i in 0..<dim { out[i] *= inv } }
    } else {
        for i in 0..<dim { out[i] = 0 }
    }
}

// ---- parallel embed ------------------------------------------------------
// Stable heap buffer shared across workers (disjoint per-row writes).
let flatPtr = UnsafeMutablePointer<Float>.allocate(capacity: n * dim)
flatPtr.initialize(repeating: 0, count: n * dim)
defer { flatPtr.deallocate() }

// Shared atomic work counter: P explicit threads each pull the next index.
// Balances load even when texts are length-sorted, and lets us oversubscribe
// past the core count (useful if the framework is latency-bound).
let counter = ManagedAtomicU64(0)
let group = DispatchGroup()
let t0 = Date()
for w in 0..<parallel {
    group.enter()
    let e = instances[w]
    Thread.detachNewThread {
        var scratch = [Float](repeating: 0, count: dim)
        while true {
            let idx = Int(counter.fetchAdd(1))
            if idx >= n { break }
            embedOne(e, texts[idx], &scratch)
            let base = idx * dim
            for d in 0..<dim { flatPtr[base + d] = scratch[d] }
        }
        group.leave()
    }
}
group.wait()
let secs = Date().timeIntervalSince(t0)

stopWatch.set(true)
let rate = Double(n) / secs
let peakGB = Double(peakRSS.get()) / 1e9
print("──────────────────────────────────────────────")
print("backend     : nl-contextual-native (parallel=\(parallel))")
print("dim         : \(dim)")
print("texts       : \(n)")
print(String(format: "embed time  : %.2f s", secs))
print(String(format: "throughput  : %.0f chunks/s", rate))
print(String(format: "init time   : %.2f s", initS))
print(String(format: "peak RSS    : %.2f GB", peakGB))
print("──────────────────────────────────────────────")
