/**
 * Overnight embedding: embed a big folder with one or more models, unattended.
 *
 * The visible page never loads a model. Each chunk of files runs in a fresh
 * hidden same-origin iframe (index.html?overnight-child) that loads the model,
 * embeds, writes IndexedDB and is then removed, which frees its JS heap, its
 * WebAssembly memory (that never shrinks in place) and its GPU device. A leak
 * can therefore only grow for one chunk. IndexedDB is the ground truth for what
 * is done, so a stop, crash or reload never redoes or loses finished work.
 */

import { makeCacheKey } from './embeddings'
import { isVideoName, keyHasPrefix } from './fileFilter'
import type { ModelVariant, PhotoFile, WebnnDevice } from './types'

export type ChunkEnd = 'ok' | 'gpu-lost' | 'low-memory' | 'error' | 'unsupported' | 'stall' | 'stopped'

/** Messages a worker frame sends its parent. */
export type ChildMsg =
  | { type: 'ready' }
  | { type: 'loading'; pct: number }
  | { type: 'loaded' }
  | { type: 'progress'; done: number; total: number }
  | { type: 'done'; reason: ChunkEnd; message?: string }

/** The one message the parent sends a worker frame. */
export interface RunMsg {
  type: 'run'
  model: ModelVariant
  files: PhotoFile[]
  /** Embed videos from DRAFT_VIDEO_FRAMES frames, under the draft namespace (see videoDraftPrefix). */
  draft?: boolean
  /** WebNN devices the page found. The frame skips the device check, and they decide its cache namespace. */
  webnn?: WebnnDevice[]
}

export interface OvernightModel {
  id: ModelVariant
  label: string
  prefix: string // cache namespace, see currentCachePrefix in app.ts
  /** Only photos or only videos; default both. */
  only?: 'images' | 'videos'
  /** The quick video pass: the frame embeds from fewer frames and writes `prefix`, a draft namespace. */
  draft?: boolean
  /** Other namespaces whose vectors already count as done for this pass (the full one, for the draft pass). */
  doneUnder?: string[]
  /** Improves vectors that exist: left out of "is this model finished", and still worth a layout when stopped. */
  bonus?: boolean
}

export interface ModelStats {
  id: ModelVariant
  label: string
  bonus: boolean
  state: 'waiting' | 'loading' | 'running' | 'done' | 'failed' | 'stopped'
  total: number // files in the folder selection
  cachedAtStart: number
  embedded: number // this run
  failed: number
  remaining: number
  loadPct: number
  rate: number // files/s over the last few minutes
  etaSec: number // NaN until known
  message: string
}

export interface OvernightStats {
  models: ModelStats[]
  startedAt: number
  elapsedSec: number
  work: number // files to embed across all models at start
  processed: number // embedded + failed + in-flight progress
  pct: number
  rate: number
  etaSec: number
  recycles: number // frames used and thrown away
  restarts: number // frames that crashed, stalled or lost the GPU
  chunkSize: number
  heapMB: number | null
  heapLimitMB: number | null
  finished: boolean
}

export interface OvernightDeps {
  files: PhotoFile[]
  models: OvernightModel[]
  cacheKeys: () => Promise<string[]>
  cacheHasBatch: (keys: string[]) => Promise<boolean[]>
  onUpdate: (s: OvernightStats) => void
  signal: AbortSignal
  childUrl?: string
  webnn?: WebnnDevice[]
  chunkSize?: number
  /** Silence after which a running frame counts as hung. */
  stallMs?: number
  /** Same, while it loads (first downloads and shader compiles are slow). */
  loadStallMs?: number
}

const RATE_WINDOW_MS = 3 * 60_000

/** Sliding-window throughput, files per second. */
export function makeRate(now: () => number = () => performance.now()) {
  const samples: [t: number, n: number][] = []
  return {
    add(n: number) {
      const t = now()
      samples.push([t, n])
      while (samples.length > 2 && samples[0][0] < t - RATE_WINDOW_MS) samples.shift()
    },
    reset() {
      samples.length = 0
    },
    get value() {
      if (samples.length < 2) return 0
      const [t0, n0] = samples[0]
      const [t1, n1] = samples[samples.length - 1]
      return t1 > t0 ? ((n1 - n0) * 1000) / (t1 - t0) : 0
    }
  }
}

function heap(): { used: number; limit: number } | null {
  const m = (performance as Performance & { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory
  return m ? { used: m.usedJSHeapSize / 1048576, limit: m.jsHeapSizeLimit / 1048576 } : null
}

export async function runOvernight(deps: OvernightDeps): Promise<OvernightStats> {
  const chunkSize = deps.chunkSize ?? 500
  const stallMs = deps.stallMs ?? 5 * 60_000
  const loadStallMs = deps.loadStallMs ?? 15 * 60_000
  const childUrl = deps.childUrl ?? `${location.pathname}?overnight-child`
  const keys = await deps.cacheKeys()

  // Work per model, decided once from the cache.
  const scopeByModel = deps.models.map((m) =>
    deps.files.filter((f) => !m.only || isVideoName(f.name) === (m.only === 'videos'))
  )
  const pendingByModel = deps.models.map((m, i) => {
    const done = new Set<string>()
    const prefixes = [m.prefix, ...(m.doneUnder ?? [])]
    for (const k of keys) if (prefixes.some((p) => keyHasPrefix(k, p))) done.add(k.slice(k.lastIndexOf('/') + 1))
    return scopeByModel[i].filter((f) => !done.has(makeCacheKey(f)))
  })
  const models: ModelStats[] = deps.models.map((m, i) => ({
    id: m.id,
    label: m.label,
    bonus: !!m.bonus,
    state: 'waiting',
    total: scopeByModel[i].length,
    cachedAtStart: scopeByModel[i].length - pendingByModel[i].length,
    embedded: 0,
    failed: 0,
    remaining: pendingByModel[i].length,
    loadPct: 0,
    rate: 0,
    etaSec: NaN,
    message: ''
  }))
  const stats: OvernightStats = {
    models,
    startedAt: Date.now(),
    elapsedSec: 0,
    work: pendingByModel.reduce((a, p) => a + p.length, 0),
    processed: 0,
    pct: 0,
    rate: 0,
    etaSec: NaN,
    recycles: 0,
    restarts: 0,
    chunkSize,
    heapMB: null,
    heapLimitMB: null,
    finished: false
  }
  const overallRate = makeRate()
  const modelRate = makeRate()
  let inFlight = 0 // files the current frame has processed, not yet confirmed in the cache

  const publish = () => {
    const settled = models.reduce((a, m) => a + m.embedded + m.failed, 0)
    stats.processed = settled + inFlight
    stats.pct = stats.work ? (100 * stats.processed) / stats.work : 100
    stats.elapsedSec = (Date.now() - stats.startedAt) / 1000
    overallRate.add(stats.processed)
    stats.rate = overallRate.value
    stats.etaSec = stats.rate > 0 ? (stats.work - stats.processed) / stats.rate : NaN
    const h = heap()
    stats.heapMB = h?.used ?? null
    stats.heapLimitMB = h?.limit ?? null
    const cur = models.find((m) => m.state === 'loading' || m.state === 'running')
    if (cur) {
      modelRate.add(cur.embedded + cur.failed + inFlight)
      cur.rate = modelRate.value
      const left = cur.remaining - inFlight
      cur.etaSec = cur.rate > 0 ? left / cur.rate : NaN
    }
    deps.onUpdate(stats)
  }
  // Keeps elapsed time and ETA moving while a model downloads or compiles.
  const ticker = setInterval(publish, 1000)

  const runChunk = (model: OvernightModel, chunk: PhotoFile[], st: ModelStats) =>
    new Promise<{ reason: ChunkEnd; loaded: boolean; message?: string }>((resolve) => {
      const frame = document.createElement('iframe')
      frame.src = childUrl
      frame.title = 'overnight embedding worker'
      frame.setAttribute('aria-hidden', 'true')
      frame.tabIndex = -1
      // In the viewport and 1 px: offscreen frames may be throttled.
      frame.style.cssText = 'position:fixed;left:0;top:0;width:1px;height:1px;border:0;opacity:0;pointer-events:none;z-index:-1'
      let loaded = false
      let timer: ReturnType<typeof setTimeout> | undefined
      const arm = () => {
        clearTimeout(timer)
        timer = setTimeout(() => finish('stall', 'no progress from the worker frame'), loaded ? stallMs : loadStallMs)
      }
      const onMsg = (e: MessageEvent) => {
        if (e.source !== frame.contentWindow || e.origin !== location.origin) return
        const m = e.data as ChildMsg
        arm()
        if (m.type === 'ready') {
          frame.contentWindow!.postMessage({ type: 'run', model: model.id, files: chunk, draft: model.draft, webnn: deps.webnn } satisfies RunMsg, location.origin)
        } else if (m.type === 'loading') {
          st.loadPct = m.pct
        } else if (m.type === 'loaded') {
          loaded = true
          st.state = 'running'
        } else if (m.type === 'progress') {
          inFlight = m.done
        } else if (m.type === 'done') {
          finish(m.reason, m.message)
        }
      }
      const onAbort = () => finish('stopped')
      const finish = (reason: ChunkEnd, message?: string) => {
        clearTimeout(timer)
        window.removeEventListener('message', onMsg)
        deps.signal.removeEventListener('abort', onAbort)
        frame.src = 'about:blank'
        frame.remove()
        resolve({ reason, loaded, message })
      }
      window.addEventListener('message', onMsg)
      deps.signal.addEventListener('abort', onAbort)
      document.body.append(frame)
      arm()
    })

  try {
    for (let mi = 0; mi < deps.models.length; mi++) {
      const model = deps.models[mi]
      const st = models[mi]
      let pending = pendingByModel[mi]
      let size = chunkSize
      let loadFails = 0
      modelRate.reset()
      if (pending.length === 0) {
        st.state = 'done'
        continue
      }
      while (pending.length > 0) {
        if (deps.signal.aborted) {
          st.state = 'stopped'
          break
        }
        st.state = 'loading'
        st.loadPct = 0
        inFlight = 0
        publish()
        const chunk = pending.slice(0, size)
        const res = await runChunk(model, chunk, st)
        const reached = inFlight // files it finished before it ended; the next one is where it stopped
        stats.recycles++
        const have = await deps.cacheHasBatch(chunk.map((f) => `${model.prefix}${makeCacheKey(f)}`))
        const rest = chunk.filter((_, i) => !have[i])
        st.embedded += chunk.length - rest.length
        pending = pending.slice(chunk.length)
        inFlight = 0

        if (res.reason === 'stopped') {
          pending.unshift(...rest)
          st.state = 'stopped'
        } else if (res.reason === 'unsupported') {
          st.state = 'failed'
          st.message = res.message ?? 'This model cannot run on this device.'
          pending = []
        } else if (res.reason === 'ok' || (res.reason === 'low-memory' && rest.length < chunk.length)) {
          // A file still missing after a clean run failed on its own (undecodable): it would fail again.
          if (res.reason === 'ok') {
            st.failed += rest.length
            if (res.message) st.message = res.message // why the files that failed did
          } else pending.unshift(...rest) // recycled early on purpose; carry on with a fresh frame
          size = chunkSize
          loadFails = 0
        } else if (!res.loaded) {
          // Never got as far as embedding: the model itself won't load.
          stats.restarts++
          pending.unshift(...rest)
          if (++loadFails >= 3) {
            st.state = 'failed'
            st.message = `Model failed to load: ${res.message ?? res.reason}`
            pending = []
          }
        } else {
          // Crash, hang or GPU loss mid-chunk. Narrow it down: a file that takes down a frame
          // on its own is skipped, everything else is retried in smaller chunks.
          stats.restarts++
          st.message = `Restarted after ${res.reason}${res.message ? `: ${res.message}` : ''}`
          if (chunk.length === 1 && rest.length === 1) st.failed++
          else {
            // A stall or crash happens on the file being worked on: try that one alone next, instead of halving
            // the chunk down to it, which cost a 5-minute stall per halving (9 for a chunk of 500). GPU loss is
            // not tied to a file, so that still narrows by halves.
            const stuck = rest.indexOf(chunk[reached])
            if (stuck > 0) rest.unshift(...rest.splice(stuck, 1))
            pending.unshift(...rest)
            const aFile = res.reason === 'stall' || res.reason === 'error'
            size = aFile ? 1 : Math.max(1, Math.floor(Math.min(size, rest.length) / 2))
          }
        }
        st.remaining = pending.length
        stats.chunkSize = size
        if (st.state === 'stopped' || st.state === 'failed') break
        // Let the removed frame's memory be collected before the next one loads: with 300 ms
        // the soak test showed both frames' memory at once (2.5 GB → 3.5 GB peak on nomic/CPU).
        await new Promise((r) => setTimeout(r, 2000))
      }
      if (st.state !== 'stopped' && st.state !== 'failed') st.state = 'done'
      st.remaining = pending.length
      st.etaSec = 0
      if (st.state === 'stopped') {
        for (const m of models) if (m.state === 'waiting') m.state = 'stopped'
        break
      }
    }
  } finally {
    clearInterval(ticker)
    stats.finished = true
    publish()
  }
  return stats
}

export function formatDuration(sec: number): string {
  if (!Number.isFinite(sec)) return '—'
  const s = Math.round(sec)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m > 0) return `${m}m ${String(s % 60).padStart(2, '0')}s`
  return `${s % 60}s`
}
