/**
 * npu-test.html: can EmbeddingGemma 2 run on WebNN (the only browser route to
 * an NPU), and how fast? Pixel 10's PowerVR GPU takes ~330 s per photo over
 * WebGPU, but Chrome's WebNN NPU ran Tiny YOLO v2 in ~59 ms there.
 *
 * Uses the fp16 export: the q4 files need MatMulNBits / GatherBlockQuantized,
 * which WebNN has no ops for. WebNN also wants static shapes, so every free
 * dimension is pinned from a real processor run before the model loads.
 *
 * Everything measured goes into one NpuReport, rendered as cards and copied as
 * JSON together with its JSON Schema. Log lines are also prefixed NPUTEST in
 * the console for logcat grepping.
 */

import { AutoProcessor, RawImage, env } from '@huggingface/transformers'
import * as ort from 'onnxruntime-web'

const REPO = 'onnx-community/embeddinggemma-2-ONNX'
const IMAGE_TOKENS = 70 // phone profile (src/vlmTiers.ts PHONE_OVERRIDES)
const SEQ_LEN = 296
const QUERY = 'a white circle on a red background' // matches picture A (hue 0), not B
// model.embed_tokens.weight inside model_fp16.onnx_data (printed by scripts/webnn_split_embed.py).
const EMBED_OFFSET = 1024
const EMBED_ROWS = 262144
const EMBED_WIDTH = 512
let embedTable: Uint16Array | null = null
const params = new URLSearchParams(location.search)
const device = params.get('device') ?? 'webnn-npu'
const dtype = params.get('dtype') ?? (device === 'webgpu' ? 'q4' : 'fp16')

// ── Report: the TypeScript shape and the JSON Schema shipped with the copy ──

interface WebnnContext {
  deviceType: 'npu' | 'gpu' | 'cpu'
  ok: boolean
  error?: string
  matmulTypes?: string[]
  maxTensorByteLength?: number
  missingOps?: string[]
}
interface ModelPart {
  name: 'vision_encoder' | 'model'
  downloadMs?: number
  megabytes?: number
  compileMs?: number
  /** True when the 268 MB embedding table was moved out of the graph (scripts/webnn_split_embed.py). */
  embedInJs?: boolean
}
interface VisionRun { rep: number; ms: number; tokens: number; nonFinite: number }
interface EmbedRun { image: string; rep: number; ms: number; visionMs: number; textMs: number; norm: number; nonFinite: number }
interface NpuReport {
  build: string
  startedAt: string
  device: string
  dtype: string
  status: 'running' | 'pass' | 'fail'
  phase: string
  error?: string
  totalMs?: number
  userAgent: string
  env: { cores: number; deviceMemoryGB: number | null; crossOriginIsolated: boolean }
  gpu: { vendor: string; architecture: string; isFallbackAdapter: boolean; maxStorageBufferBindingSize: number; shaderF16: boolean } | null
  webnn: { available: boolean; contexts: WebnnContext[] }
  inputShapes: Record<string, number[]>
  freeDimensionOverrides: Record<string, number>
  parts: ModelPart[]
  visionRuns: VisionRun[]
  embeds: EmbedRun[]
  /** Cosine similarities: a near-copy should score far above a different picture. */
  similarity: {
    aVsAShifted?: number
    aVsB?: number
    /** Same picture at 3:4 instead of 4:3: a different image-token count. */
    aVsPortrait?: number
    queryVsA?: number
    queryVsB?: number
    vsOtherRuns: Record<string, number>
  }
  warnings: string[]
}

const num = { type: 'number' }
const str = { type: 'string' }
const REPORT_SCHEMA = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  title: 'NpuReport',
  description: 'EmbeddingGemma 2 on WebNN: environment, model load and embedding timings (npu-test.html).',
  type: 'object',
  required: ['build', 'startedAt', 'device', 'dtype', 'status', 'phase', 'userAgent', 'env', 'gpu', 'webnn', 'inputShapes', 'freeDimensionOverrides', 'parts', 'visionRuns', 'embeds', 'similarity', 'warnings'],
  properties: {
    build: { ...str, description: 'git commit of the page' },
    startedAt: { ...str, format: 'date-time' },
    device: { ...str, description: 'ORT execution provider, e.g. webnn-npu, webgpu, wasm' },
    dtype: { ...str, description: 'model file variant, e.g. fp16' },
    status: { enum: ['running', 'pass', 'fail'] },
    phase: { ...str, description: 'last step started; where it failed when status=fail' },
    error: str,
    totalMs: num,
    userAgent: str,
    env: {
      type: 'object',
      properties: { cores: num, deviceMemoryGB: { type: ['number', 'null'] }, crossOriginIsolated: { type: 'boolean' } }
    },
    gpu: {
      type: ['object', 'null'],
      description: 'WebGPU adapter, null when none',
      properties: {
        vendor: str, architecture: str, isFallbackAdapter: { type: 'boolean' },
        maxStorageBufferBindingSize: num, shaderF16: { type: 'boolean' }
      }
    },
    webnn: {
      type: 'object',
      properties: {
        available: { type: 'boolean', description: 'navigator.ml exists (chrome://flags "WebNN API")' },
        contexts: {
          type: 'array',
          items: {
            type: 'object',
            required: ['deviceType', 'ok'],
            properties: {
              deviceType: { enum: ['npu', 'gpu', 'cpu'] }, ok: { type: 'boolean' }, error: str,
              matmulTypes: { type: 'array', items: str }, maxTensorByteLength: num,
              missingOps: { type: 'array', items: str, description: 'WebNN ops the model needs that this context lacks' }
            }
          }
        }
      }
    },
    inputShapes: { type: 'object', additionalProperties: { type: 'array', items: num }, description: 'processor output dims for picture A' },
    freeDimensionOverrides: { type: 'object', additionalProperties: num },
    parts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name'],
        properties: {
          name: { enum: ['vision_encoder', 'model'] }, downloadMs: num, megabytes: num, compileMs: num,
          embedInJs: { type: 'boolean', description: 'token-embedding table looked up in JS instead of the graph' }
        }
      }
    },
    visionRuns: {
      type: 'array',
      items: { type: 'object', properties: { rep: num, ms: num, tokens: num, nonFinite: num } }
    },
    embeds: {
      type: 'array',
      items: {
        type: 'object',
        properties: { image: str, rep: num, ms: num, visionMs: num, textMs: num, norm: num, nonFinite: num }
      }
    },
    similarity: {
      type: 'object',
      properties: {
        aVsAShifted: { ...num, description: 'cosine, near-copy; expect ≫ aVsB' },
        aVsB: { ...num, description: 'cosine, different picture' },
        aVsPortrait: { ...num, description: 'cosine, same picture at 3:4 (different image-token count)' },
        queryVsA: { ...num, description: 'cosine, text query describing A; expect > queryVsB' },
        queryVsB: num,
        vsOtherRuns: { type: 'object', additionalProperties: num, description: 'cosine of picture A vs earlier runs on other devices in this browser' }
      }
    },
    warnings: { type: 'array', items: str }
  }
} as const

const nav = navigator as Navigator & { deviceMemory?: number; ml?: ML }
const report: NpuReport = {
  build: __GIT_COMMIT__,
  startedAt: new Date().toISOString(),
  device,
  dtype,
  status: 'running',
  phase: 'diagnostics',
  userAgent: navigator.userAgent,
  env: { cores: navigator.hardwareConcurrency, deviceMemoryGB: nav.deviceMemory ?? null, crossOriginIsolated },
  gpu: null,
  webnn: { available: !!nav.ml, contexts: [] },
  inputShapes: {},
  freeDimensionOverrides: {},
  parts: [],
  visionRuns: [],
  embeds: [],
  similarity: { vsOtherRuns: {} },
  warnings: []
}

// ── Rendering ───────────────────────────────────────────────────────────────

const $ = (id: string) => document.getElementById(id)!
const logEl = $('log')
const esc = (s: unknown) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`)
const s = (ms?: number) => (ms === undefined ? '–' : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)}&nbsp;s`)
const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined)
const table = (head: string[], rows: unknown[][]) =>
  `<table><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</table>`
const card = (title: string, body: string) => `<section class="card"><h2>${title}</h2>${body}</section>`

function render(): void {
  const r = report
  const status = $('status')
  status.className = `badge ${r.status}`
  status.textContent = r.status === 'running' ? `running · ${r.phase}` : r.status === 'fail' ? `failed in ${r.phase}` : 'pass'

  // Steady state: skip the first run of picture A (it pays one-time warm-up).
  const steady = r.embeds.filter((e) => !(e.image === 'A' && e.rep === 1) && e.image !== 'query')
  const query = r.embeds.find((e) => e.image === 'query')
  const vision = r.visionRuns.slice(1)
  const tiles = [
    ['per photo', s(avg(steady.map((e) => e.ms)))],
    ['vision part', s(avg(steady.length ? steady.map((e) => e.visionMs) : vision.map((v) => v.ms)))],
    ['text part', s(avg(steady.map((e) => e.textMs)))],
    ['search query', s(query?.ms)],
    ['compile (both)', s(r.parts.reduce((a, p) => a + (p.compileMs ?? 0), 0) || undefined)]
  ]
  let html = card('Result', `<div class="tiles">${tiles.map(([k, v]) => `<div class="tile"><b>${v}</b><span>${k}</span></div>`).join('')}</div>` +
    (r.error ? `<p class="bad"><b>Error:</b> ${esc(r.error)}</p>` : ''))

  html += card('Device', table(['', ''], [
    ['Run on', `<code>${esc(r.device)}</code> · <code>${esc(r.dtype)}</code>`],
    ['Browser', `<span class="muted">${esc(r.userAgent)}</span>`],
    ['CPU / RAM', `${r.env.cores} cores · ${r.env.deviceMemoryGB ?? '?'} GB · ${r.env.crossOriginIsolated ? 'threads on' : 'threads off'}`],
    ['GPU', r.gpu ? `${esc(r.gpu.vendor)} ${esc(r.gpu.architecture)} · max binding ${(r.gpu.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MB · f16 ${r.gpu.shaderF16 ? 'yes' : 'no'}${r.gpu.isFallbackAdapter ? ' · <span class="bad">software</span>' : ''}` : '<span class="muted">none</span>']
  ]))

  html += card('WebNN', r.webnn.available
    ? table(['Device', 'Context', 'matmul types', 'Max tensor', 'Missing ops'], r.webnn.contexts.map((c) => [
      c.deviceType.toUpperCase(),
      c.ok ? '<span class="ok">ok</span>' : `<span class="bad">${esc(c.error)}</span>`,
      esc(c.matmulTypes?.join(', ') ?? '–'),
      c.maxTensorByteLength ? `${(c.maxTensorByteLength / 2 ** 30).toFixed(1)} GB` : '–',
      c.missingOps ? (c.missingOps.length ? `<span class="bad">${c.missingOps.join(', ')}</span>` : '<span class="ok">none</span>') : '–'
    ]))
    : '<p class="bad">navigator.ml is missing: turn on <code>chrome://flags</code> → "WebNN API" and restart Chrome.</p>')

  if (r.parts.length) {
    html += card('Model load', table(['Part', 'Download', 'Size', 'Compile'], r.parts.map((p) => [
      `${p.name === 'model' ? 'text' : 'vision'}${p.embedInJs ? ' <span class="muted">(token lookup in JS)</span>' : ''}`,
      s(p.downloadMs), p.megabytes ? `${p.megabytes}&nbsp;MB` : '–', s(p.compileMs)
    ])))
  }
  if (r.visionRuns.length) {
    html += card('Vision part alone', table(['Run', 'Time', 'Tokens', 'Broken values'], r.visionRuns.map((v) => [
      `#${v.rep}`, s(v.ms), v.tokens, v.nonFinite ? `<span class="bad">${v.nonFinite}</span>` : '<span class="ok">0</span>'
    ])))
  }
  if (r.embeds.length) {
    html += card('Full embeddings', table(['Picture', 'Total', 'Vision', 'Text', 'Norm', 'Broken'], r.embeds.map((e) => [
      `${esc(e.image)} #${e.rep}`, s(e.ms), s(e.visionMs), s(e.textMs), e.norm.toFixed(3),
      e.nonFinite ? `<span class="bad">${e.nonFinite}</span>` : '<span class="ok">0</span>'
    ])))
  }
  const sim = r.similarity
  if (sim.aVsB !== undefined) {
    const good = (sim.aVsAShifted ?? 0) > (sim.aVsB ?? 1) && (sim.queryVsA ?? 0) > (sim.queryVsB ?? 1)
    html += card('Sanity', table(['Comparison', 'Cosine'], [
      ['A vs near-copy of A', sim.aVsAShifted!.toFixed(4)],
      ['A vs different picture B', sim.aVsB.toFixed(4)],
      ['A vs same picture, portrait', sim.aVsPortrait?.toFixed(4) ?? '–'],
      [`query “${QUERY}” vs A`, sim.queryVsA?.toFixed(4) ?? '–'],
      ['same query vs B', sim.queryVsB?.toFixed(4) ?? '–'],
      ...Object.entries(sim.vsOtherRuns).map(([k, v]) => [`A vs earlier run on <code>${esc(k)}</code>`, `<span class="${v > 0.99 ? 'ok' : 'bad'}">${v.toFixed(4)}</span>`])
    ]) + `<p class="${good ? 'ok' : 'bad'}">${good ? 'Near-copy and matching query score higher, as expected.' : 'Near-copy or matching query does not score higher: the output looks broken.'}</p>`)
  }
  if (r.warnings.length) html += card('Warnings', `<pre>${esc(r.warnings.join('\n'))}</pre>`)
  $('report').innerHTML = html
}

for (const a of document.querySelectorAll<HTMLAnchorElement>('#devices a')) {
  const p = new URLSearchParams(a.search)
  if (params.has('diag') ? p.has('diag') : p.get('device') === device) a.classList.add('on')
}
const flash = (btn: HTMLElement, text: string) => {
  const old = btn.textContent
  btn.textContent = text
  setTimeout(() => (btn.textContent = old), 1500)
}
$('copy-json').onclick = (e) =>
  void navigator.clipboard?.writeText(JSON.stringify({ schema: REPORT_SCHEMA, report }, null, 2)).then(() => flash(e.target as HTMLElement, 'Copied'))
$('copy-log').onclick = (e) => void navigator.clipboard?.writeText(logEl.textContent ?? '').then(() => flash(e.target as HTMLElement, 'Copied'))

function log(msg: string): void {
  console.log(`NPUTEST ${msg}`)
  logEl.textContent += msg + '\n'
}
const errMsg = (e: unknown) => (e as Error)?.message ?? String(e)

// ORT reports which nodes fall back off the requested device as console
// warnings; keep them in the log and the report so a phone user can copy them.
for (const level of ['warn', 'error'] as const) {
  const orig = console[level].bind(console)
  console[level] = (...args: unknown[]) => {
    orig(...args)
    // eslint-disable-next-line no-control-regex
    const text = args.map(String).join(' ').replace(/\x1b\[[0-9;]*m/g, '').slice(0, 500)
    logEl.textContent += `[${level}] ${text}\n`
    report.warnings.push(text)
    render()
  }
}
window.addEventListener('error', (e) => log(`window.error ${e.message}`))

// The processor only needs the tokenizer/image code; inference uses onnxruntime-web
// directly. dist strips the ort-wasm binaries, so they come from the CDN.
env.allowLocalModels = false
env.useBrowserCache = true
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`
ort.env.logLevel = params.has('verbose') ? 'verbose' : 'warning'
const [epName, deviceType] = device.split('-') // webnn-npu → webnn + npu; webgpu → webgpu
const ep = deviceType ? { name: epName, deviceType } : epName

// WebNN build errors surface as unhandled rejections and the create never settles.
let rejectFailed: (e: unknown) => void = () => {}
const failed = new Promise<never>((_, reject) => (rejectFailed = reject))
window.addEventListener('unhandledrejection', (e) => {
  log(`unhandledrejection ${errMsg(e.reason)}`)
  rejectFailed(e.reason)
})

/** Fetch a model file once; later runs read it from the Cache API. */
async function cachedFetch(file: string): Promise<Uint8Array> {
  const url = `https://huggingface.co/${REPO}/resolve/main/onnx/${file}`
  const cache = await caches.open('npu-test').catch(() => null)
  let res = await cache?.match(url)
  if (!res) {
    res = await fetch(url)
    if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`)
    await cache?.put(url, res.clone()).catch((e) => log(`cache put ${file} failed: ${errMsg(e)}`))
  }
  return new Uint8Array(await res.arrayBuffer())
}

/** A graph this site serves itself; an HTML reply means the file is missing (SPA fallback). */
async function ownGraph(file: string): Promise<Uint8Array> {
  const res = await fetch(`${import.meta.env.BASE_URL}models/embeddinggemma2-webnn/${file}`)
  if (!res.ok || res.headers.get('content-type')?.includes('text/html')) {
    throw new Error(`${file} is not deployed (HTTP ${res.status}, ${res.headers.get('content-type')})`)
  }
  return new Uint8Array(await res.arrayBuffer())
}

// WebNN is not in the TS DOM lib yet; only what this page touches.
interface MLContext {
  opSupportLimits?: () => Record<string, unknown>
}
type ML = { createContext: (o: { deviceType: string }) => Promise<MLContext> }

// WebNN op names for every ONNX op in vision_encoder_fp16 + model_fp16
// (contrib ops listed by what ORT lowers them to).
const NEEDED_OPS = [
  'matmul', 'gemm', 'add', 'sub', 'mul', 'div', 'max', 'abs', 'neg', 'cos', 'sin', 'gelu', 'softmax',
  'cast', 'reshape', 'transpose', 'concat', 'slice', 'split', 'expand', 'pad', 'gather', 'gatherND',
  'scatterElements', 'where', 'equal', 'greater', 'lesserOrEqual', 'logicalAnd', 'logicalNot', 'clamp',
  'reduceSum', 'reduceMax', 'reduceL2', 'cumulativeSum', 'layerNormalization', 'sqrt', 'pow'
]

async function diagnostics(): Promise<void> {
  log(`build=${report.build} device=${device} dtype=${dtype} ua=${navigator.userAgent}`)
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null)
  if (adapter) {
    const i = adapter.info
    report.gpu = {
      vendor: i.vendor, architecture: i.architecture, isFallbackAdapter: i.isFallbackAdapter,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, shaderF16: adapter.features.has('shader-f16')
    }
  }
  log(`gpu ${JSON.stringify(report.gpu)}`)
  if (!nav.ml) return
  for (const deviceType of ['npu', 'gpu', 'cpu'] as const) {
    const c: WebnnContext = { deviceType, ok: false }
    try {
      const limits = (await nav.ml.createContext({ deviceType })).opSupportLimits?.()
      c.ok = true
      if (limits) {
        c.missingOps = NEEDED_OPS.filter((op) => !(op in limits))
        c.matmulTypes = (limits.matmul as { a?: { dataTypes?: string[] } } | undefined)?.a?.dataTypes
        c.maxTensorByteLength = Number(limits.maxTensorByteLength) || undefined
      }
    } catch (e) {
      c.error = errMsg(e)
    }
    report.webnn.contexts.push(c)
    log(`webnn ${JSON.stringify(c)}`)
    render()
  }
}

/** Synthetic picture; 640×480 unless given, a different size gives a different image-token count. */
async function picture(hue: number, label: string, shift = 0, w = 640, h = 480): Promise<RawImage> {
  const c = new OffscreenCanvas(w, h)
  const g = c.getContext('2d')!
  const grad = g.createLinearGradient(0, 0, w, h)
  grad.addColorStop(0, `hsl(${hue} 80% 55%)`)
  grad.addColorStop(1, `hsl(${(hue + 120) % 360} 70% 30%)`)
  g.fillStyle = grad
  g.fillRect(0, 0, w, h)
  g.fillStyle = 'white'
  g.beginPath()
  g.arc(w / 2 + shift, h / 2, 120, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = 'black'
  g.font = '60px sans-serif'
  g.fillText(label, 64, 96)
  const d = g.getImageData(0, 0, w, h)
  return new RawImage(new Uint8ClampedArray(d.data), w, h, 4)
}

const cos = (a: Float32Array, b: Float32Array) => {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) (dot += a[i] * b[i]), (na += a[i] * a[i]), (nb += b[i] * b[i])
  return dot / Math.sqrt(na * nb)
}
const nonFinite = (v: Float32Array) => v.filter((x) => !Number.isFinite(x)).length

type Proc = ((t: string[] | null, i: RawImage[][] | null, a: null, v: null, o: Record<string, unknown>) => Promise<Record<string, { dims: number[] }>>) & {
  image_processor: { max_soft_tokens: number }
}

async function run(): Promise<void> {
  const t0 = performance.now()
  const step = (phase: string) => {
    report.phase = phase
    render()
  }
  const beat = setInterval(() => log(`… still in ${report.phase} (${((performance.now() - t0) / 1000).toFixed(0)}s)`), 30_000)
  try {
    await diagnostics()
    if (params.has('diag')) {
      report.status = 'pass'
      return
    }

    step('processor')
    const proc = (await AutoProcessor.from_pretrained(REPO)) as unknown as Proc
    proc.image_processor.max_soft_tokens = IMAGE_TOKENS
    const opts = { padding: 'max_length', max_length: SEQ_LEN, truncation: true }
    const pics = {
      A: await picture(0, 'A'), 'A shifted': await picture(0, 'A', 12), B: await picture(200, 'B'),
      'A portrait': await picture(0, 'A', 0, 480, 640)
    }
    const inputs: Record<string, Record<string, { dims: number[] }>> = {}
    for (const [name, img] of Object.entries(pics)) inputs[name] = await proc(null, [[img]], null, null, opts)
    // A search query: text only, so zero image tokens.
    inputs.query = await proc(['task: search result | query: ' + QUERY], null, null, null, opts)
    for (const [k, v] of Object.entries(inputs.A)) if (v?.dims) report.inputShapes[k] = v.dims

    // Pin the free dimensions that stay fixed (names from the ONNX graphs).
    // Image/video token counts depend on the aspect ratio and are 0 for a
    // query, so they stay free: only the small merge step that consumes them
    // falls back to CPU, and the layers after it see a fixed [1, 296, 512].
    const overrides = report.freeDimensionOverrides
    Object.assign(overrides, {
      s11: 1, s35: inputs.A.pixel_values.dims[1], batch_size: 1, sequence_length: SEQ_LEN, num_audio_tokens: 0
    })
    log(`shapes ${JSON.stringify(report.inputShapes)}`)

    // Two ORT sessions, created one at a time, so a failure names its half.
    const sessions: Record<string, ort.InferenceSession> = {}
    // Rebuild as this ORT's tensors: transformers.js bundles its own ORT, and run() checks instanceof.
    const t = (inp: Record<string, unknown>, k: string) => {
      const x = inp[k] as { type: ort.Tensor.Type; data: ort.Tensor.DataType; dims: number[] }
      return new ort.Tensor(x.type, x.data, x.dims)
    }
    const vision = (inp: Record<string, unknown>) =>
      sessions.vision_encoder.run({ pixel_values: t(inp, 'pixel_values'), pixel_position_ids: t(inp, 'image_position_ids') })

    for (const name of ['vision_encoder', 'model'] as const) {
      const part: ModelPart = { name }
      report.parts.push(part)
      step(`download ${name}`)
      const tDl = performance.now()
      const suffix = dtype === 'fp32' ? '' : `_${dtype}`
      // fp16 text model on WebNN: our graph without the 268 MB embedding table
      // (scripts/webnn_split_embed.py), which WebNN refuses to build.
      part.embedInJs = name === 'model' && dtype === 'fp16' && device.startsWith('webnn')
      const graph = part.embedInJs ? await ownGraph('model_fp16_noembed.onnx') : await cachedFetch(`${name}${suffix}.onnx`)
      const data = await cachedFetch(`${name}${suffix}.onnx_data`)
      if (part.embedInJs) embedTable = new Uint16Array(data.buffer, data.byteOffset + EMBED_OFFSET, EMBED_ROWS * EMBED_WIDTH)
      part.downloadMs = performance.now() - tDl
      part.megabytes = Math.round((graph.byteLength + data.byteLength) / 1e6)

      step(`compile ${name}`)
      const tC = performance.now()
      sessions[name] = await Promise.race([
        ort.InferenceSession.create(graph, {
          executionProviders: [ep],
          freeDimensionOverrides: overrides,
          externalData: [{ path: `${name}${suffix}.onnx_data`, data }],
          logSeverityLevel: params.has('verbose') ? 0 : 2
        }),
        failed
      ])
      part.compileMs = performance.now() - tC
      log(`part ${JSON.stringify(part)}`)

      if (name === 'vision_encoder') {
        // Timed on its own, so a text-part failure still says whether the NPU is fast.
        for (let rep = 1; rep <= 3; rep++) {
          step(`vision run #${rep}`)
          const tV = performance.now()
          const out = await vision(inputs.A)
          const run: VisionRun = { rep, ms: performance.now() - tV, tokens: out.image_features.dims[0], nonFinite: nonFinite(out.image_features.data as Float32Array) }
          report.visionRuns.push(run)
          log(`vision ${JSON.stringify(run)}`)
        }
      }
    }

    const empty = new ort.Tensor('float32', new Float32Array(0), [0, 512])
    const vecs: Record<string, Float32Array> = {}
    for (const [image, inp] of Object.entries(inputs)) {
      for (let rep = 1; rep <= (image === 'A' ? 3 : 1); rep++) {
        step(`embed ${image} #${rep}`)
        const tE = performance.now()
        const imageFeatures = image === 'query' ? empty : (await vision(inp)).image_features
        const visionMs = performance.now() - tE
        const ids = t(inp, 'input_ids')
        const feeds: Record<string, ort.Tensor> = {}
        if (embedTable) {
          // The lookup the graph no longer does: one fp16 row per token.
          const idData = ids.data as BigInt64Array
          const rows = new Uint16Array(idData.length * EMBED_WIDTH)
          idData.forEach((id, i) => rows.set(embedTable!.subarray(Number(id) * EMBED_WIDTH, (Number(id) + 1) * EMBED_WIDTH), i * EMBED_WIDTH))
          feeds.inputs_embeds = new ort.Tensor('float16', rows, [...ids.dims, EMBED_WIDTH])
        }
        const out = await sessions.model.run({
          ...feeds,
          input_ids: ids, attention_mask: t(inp, 'attention_mask'),
          image_features: imageFeatures, video_features: empty, audio_features: empty
        })
        const ms = performance.now() - tE
        const v = Float32Array.from(out.sentence_embedding.data as Float32Array)
        const e: EmbedRun = { image, rep, ms, visionMs, textMs: ms - visionMs, norm: Math.sqrt(v.reduce((a, x) => a + x * x, 0)), nonFinite: nonFinite(v) }
        report.embeds.push(e)
        log(`embed ${JSON.stringify(e)}`)
        vecs[image] = v
      }
    }
    // fp16 overflow (why the app skips q4f16) shows up as NaN or collapsed scores.
    report.similarity.aVsAShifted = cos(vecs.A, vecs['A shifted'])
    report.similarity.aVsB = cos(vecs.A, vecs.B)
    report.similarity.aVsPortrait = cos(vecs.A, vecs['A portrait'])
    report.similarity.queryVsA = cos(vecs.query, vecs.A)
    report.similarity.queryVsB = cos(vecs.query, vecs.B)

    // Compare with earlier runs on other devices in this browser.
    const key = `npuTest:${device}:${dtype}`
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i)!
        if (k.startsWith('npuTest:') && k !== key) {
          report.similarity.vsOtherRuns[k.slice(8)] = cos(vecs.A, Float32Array.from(JSON.parse(localStorage.getItem(k)!)))
        }
      }
      localStorage.setItem(key, JSON.stringify(Array.from(vecs.A)))
    } catch {}
    log(`similarity ${JSON.stringify(report.similarity)}`)
    report.status = 'pass'
  } catch (e) {
    report.status = 'fail'
    report.error = errMsg(e)
  } finally {
    clearInterval(beat)
    report.totalMs = performance.now() - t0
    log(`DONE ${report.status}${report.error ? ` in ${report.phase}: ${report.error}` : ''} total=${(report.totalMs / 1000).toFixed(1)}s`)
    render()
  }
}

// Self-check: the schema shipped with the copy must cover every report field.
const unschemed = Object.keys(report).filter((k) => !(k in REPORT_SCHEMA.properties))
if (unschemed.length) report.warnings.push(`schema is missing report fields: ${unschemed.join(', ')}`)

render()
void run()
