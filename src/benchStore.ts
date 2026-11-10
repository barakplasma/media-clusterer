/**
 * Remembered speed measurements, so the app can pick the fastest accelerator over every run it has seen.
 * Written by npu-test.html (a benchmark run) and by the app itself (a real embedding run, from its
 * own numbers because the app's WebGPU path runs the q4 model where the benchmark runs fp16); read at start-up.
 *
 * Plain localStorage JSON, same origin as the app. Nothing here may throw: a blocked, full or hand-edited
 * store just means "no data", and the app falls back to its default order.
 */

import type { WebnnDevice } from './types'

export type Accelerator = 'webgpu' | 'webnn-npu' | 'webnn-gpu' | 'webnn-cpu' | 'wasm'
const ACCELERATORS: readonly string[] = ['webgpu', 'webnn-npu', 'webnn-gpu', 'webnn-cpu', 'wasm']

export interface BenchRun {
  at: number // Date.now()
  source: 'bench' | 'app'
  device: Accelerator
  dtype: string // model file variant, 'fp16' | 'q4': a speed only compares with the same one
  embedMs: number // median time to embed one photo, preparing its inputs included
  n: number // photos behind that median
}

/** Two devices embedding at the same time (npu-test.html hybrid run). */
export interface HybridRun {
  at: number
  devices: Accelerator[]
  seconds: number
  combinedPerSec: number
  bestAlonePerSec: number
  cosine: number // the two devices' vectors for the same picture: how close their embedding spaces are
}

interface Store {
  runs: BenchRun[]
  hybrid: HybridRun[]
}

export const BENCH_KEY = 'mc_bench'
const KEEP_RUNS = 30 // per device and dtype, newest
const KEEP_HYBRID = 20

const isRun = (r: unknown): r is BenchRun => {
  const x = r as BenchRun
  return (
    !!x &&
    ACCELERATORS.includes(x.device) &&
    typeof x.dtype === 'string' &&
    Number.isFinite(x.embedMs) &&
    x.embedMs > 0 &&
    (x.source === 'bench' || x.source === 'app')
  )
}

function read(): Store {
  try {
    const s = JSON.parse(localStorage.getItem(BENCH_KEY) ?? 'null')
    return {
      runs: Array.isArray(s?.runs) ? s.runs.filter(isRun) : [],
      hybrid: Array.isArray(s?.hybrid) ? s.hybrid : []
    }
  } catch {
    return { runs: [], hybrid: [] }
  }
}

function write(s: Store): void {
  try {
    localStorage.setItem(BENCH_KEY, JSON.stringify(s))
  } catch {
    /* full or blocked: the next run tries again */
  }
}

export function benchRuns(): BenchRun[] {
  return read().runs
}

export function hybridRuns(): HybridRun[] {
  return read().hybrid
}

export function recordBench(run: BenchRun): void {
  if (!isRun(run)) return
  const s = read()
  s.runs.push(run)
  // Keep the newest KEEP_RUNS of this device and dtype, everything else as it was.
  const same = (r: BenchRun) => r.device === run.device && r.dtype === run.dtype
  const drop = s.runs.filter(same).length - KEEP_RUNS
  if (drop > 0) {
    let n = drop
    s.runs = s.runs.filter((r) => !(same(r) && n-- > 0))
  }
  write(s)
}

export function recordHybrid(run: HybridRun): void {
  const s = read()
  s.hybrid = [...s.hybrid, run].slice(-KEEP_HYBRID)
  write(s)
}

export const median = (xs: number[]) => {
  const v = [...xs].sort((a, b) => a - b)
  return v.length % 2 ? v[v.length >> 1] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2
}

/** WebNN devices in the order to try: the fastest first when it is one of them, none at all when it is WebGPU. */
export function webnnOrder(webnn: WebnnDevice[], fastest?: Accelerator): WebnnDevice[] {
  if (fastest === 'webgpu') return []
  const first = webnn.find((t) => `webnn-${t}` === fastest)
  return first ? [first, ...webnn.filter((t) => t !== first)] : webnn
}

export interface Candidate {
  device: Accelerator
  dtype: string
}

/**
 * Time per photo for each candidate: the median over its runs (each run counts once, whatever its
 * photo count), from the benchmark page and the app alike. Candidates with no run are left out.
 */
export function measuredMs(candidates: Candidate[], runs: BenchRun[] = benchRuns()): Map<Accelerator, number> {
  const out = new Map<Accelerator, number>()
  for (const c of candidates) {
    const mine = runs.filter((r) => r.device === c.device && r.dtype === c.dtype)
    if (mine.length) out.set(c.device, median(mine.map((r) => r.embedMs)))
  }
  return out
}

/**
 * The fastest of `candidates` by what has been measured, or null while fewer than two of them have been:
 * one measured device against an unmeasured one proves nothing, so the app keeps its default order.
 */
export function fastestAccelerator(
  candidates: Candidate[],
  runs: BenchRun[] = benchRuns()
): { device: Accelerator; ms: number } | null {
  const ms = measuredMs(candidates, runs)
  if (ms.size < 2) return null
  const [device, best] = [...ms].reduce((a, b) => (b[1] < a[1] ? b : a))
  return { device, ms: best }
}
