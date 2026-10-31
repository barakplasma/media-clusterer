/**
 * npu-test.html: can EmbeddingGemma 2 run on WebNN (the only browser route to
 * an NPU), and how fast? Pixel 10's PowerVR GPU takes ~330 s per photo over
 * WebGPU, but Chrome's WebNN NPU ran Tiny YOLO v2 in ~59 ms there.
 *
 * Uses the fp16 export: the q4 files need MatMulNBits / GatherBlockQuantized,
 * which WebNN has no ops for. WebNN also wants static shapes, so every free
 * dimension is pinned from a real processor run before the model loads.
 *
 * Every line is prefixed NPUTEST in the console for logcat grepping.
 */

import { AutoProcessor, RawImage, env } from '@huggingface/transformers'
import * as ort from 'onnxruntime-web'

const REPO = 'onnx-community/embeddinggemma-2-ONNX'
const IMAGE_TOKENS = 70 // phone profile (src/vlmTiers.ts PHONE_OVERRIDES)
const SEQ_LEN = 296
// model.embed_tokens.weight inside model_fp16.onnx_data (printed by scripts/webnn_split_embed.py).
const EMBED_OFFSET = 1024
const EMBED_ROWS = 262144
const EMBED_WIDTH = 512
let embedTable: Uint16Array | null = null
const params = new URLSearchParams(location.search)
const device = params.get('device') ?? 'webnn-npu'
const dtype = params.get('dtype') ?? (device === 'webgpu' ? 'q4' : 'fp16')

const logEl = document.getElementById('log')!
function log(msg: string): void {
  console.log(`NPUTEST ${msg}`)
  logEl.textContent += msg + '\n'
}
const errMsg = (e: unknown) => (e as Error)?.message ?? String(e)
document.getElementById('copy')!.onclick = () => void navigator.clipboard?.writeText(logEl.textContent ?? '')

// ORT reports which nodes fall back off the requested device as console
// warnings; mirror them onto the page so a phone user can copy them.
for (const level of ['warn', 'error'] as const) {
  const orig = console[level].bind(console)
  console[level] = (...args: unknown[]) => {
    orig(...args)
    logEl.textContent += `[${level}] ${args.map(String).join(' ').slice(0, 500)}\n`
  }
}
window.addEventListener('error', (e) => log(`window.error ${e.message}`))
window.addEventListener('unhandledrejection', (e) => log(`unhandledrejection ${errMsg(e.reason)}`))

// The processor only needs the tokenizer/image code; inference uses onnxruntime-web
// directly. dist strips the ort-wasm binaries, so they come from the CDN.
env.allowLocalModels = false
env.useBrowserCache = true
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`
ort.env.logLevel = params.has('verbose') ? 'verbose' : 'warning'
const [epName, deviceType] = device.split('-') // webnn-npu → webnn + npu; webgpu → webgpu
const ep = deviceType ? { name: epName, deviceType } : epName

let rejectFailed: (e: unknown) => void = () => {}
const failed = new Promise<never>((_, reject) => (rejectFailed = reject))
window.addEventListener('unhandledrejection', (e) => rejectFailed(e.reason))

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

// WebNN is not in the TS DOM lib yet; only what this page touches.
interface MLContext {
  opSupportLimits?: () => Record<string, { input?: { dataTypes?: string[] }; a?: { dataTypes?: string[] } } | unknown>
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
  const nav = navigator as Navigator & { deviceMemory?: number; ml?: ML }
  log(`build=${__GIT_COMMIT__} device=${device} dtype=${dtype}`)
  log(`ua=${navigator.userAgent}`)
  log(`cores=${navigator.hardwareConcurrency} deviceMemory=${nav.deviceMemory ?? '?'}GB crossOriginIsolated=${crossOriginIsolated}`)

  const adapter = await navigator.gpu?.requestAdapter().catch(() => null)
  if (adapter) {
    const i = adapter.info
    log(`gpu vendor=${i.vendor} arch=${i.architecture} fallback=${i.isFallbackAdapter} maxStorageBinding=${adapter.limits.maxStorageBufferBindingSize} f16=${adapter.features.has('shader-f16')}`)
  } else log('gpu adapter none')

  if (!nav.ml) {
    log('webnn navigator.ml missing: turn on chrome://flags "WebNN API" and restart Chrome')
    return
  }
  for (const deviceType of ['npu', 'gpu', 'cpu']) {
    try {
      const ctx = await nav.ml.createContext({ deviceType })
      const limits = ctx.opSupportLimits?.() as Record<string, unknown> | undefined
      if (!limits) {
        log(`webnn ${deviceType}: context ok, no opSupportLimits()`)
        continue
      }
      const missing = NEEDED_OPS.filter((op) => !(op in limits))
      const mm = limits.matmul as { a?: { dataTypes?: string[] } } | undefined
      const maxBytes = Number(limits.maxTensorByteLength ?? 0)
      // Biggest constant: the text model's token-embedding table, 262144 × 768 fp16.
      const embedBytes = 262144 * 768 * (dtype === 'fp32' ? 4 : 2)
      log(
        `webnn ${deviceType}: context ok, matmul types=${mm?.a?.dataTypes?.join('/') ?? '?'}, ` +
          `maxTensorByteLength=${maxBytes || '?'}${maxBytes && maxBytes < embedBytes ? ` (< ${embedBytes} embed table!)` : ''}, ` +
          `missing ops=${missing.join(',') || 'none'}`
      )
    } catch (e) {
      log(`webnn ${deviceType}: createContext failed: ${errMsg(e)}`)
    }
  }
}

/** Synthetic 640×480 picture; same size every time so the shapes stay static. */
async function picture(hue: number, label: string, shift = 0): Promise<RawImage> {
  const c = new OffscreenCanvas(640, 480)
  const g = c.getContext('2d')!
  const grad = g.createLinearGradient(0, 0, 640, 480)
  grad.addColorStop(0, `hsl(${hue} 80% 55%)`)
  grad.addColorStop(1, `hsl(${(hue + 120) % 360} 70% 30%)`)
  g.fillStyle = grad
  g.fillRect(0, 0, 640, 480)
  g.fillStyle = 'white'
  g.beginPath()
  g.arc(320 + shift, 240, 120, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = 'black'
  g.font = '60px sans-serif'
  g.fillText(label, 64, 96)
  const d = g.getImageData(0, 0, 640, 480)
  return new RawImage(new Uint8ClampedArray(d.data), 640, 480, 4)
}

const cos = (a: Float32Array, b: Float32Array) => {
  let s = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) (s += a[i] * b[i]), (na += a[i] * a[i]), (nb += b[i] * b[i])
  return s / Math.sqrt(na * nb)
}

type Proc = ((t: null, i: RawImage[][], a: null, v: null, o: Record<string, unknown>) => Promise<Record<string, { dims: number[] }>>) & {
  image_processor: { max_soft_tokens: number }
}
type Model = (i: Record<string, unknown>) => Promise<{ sentence_embedding: { data: Float32Array } }>

async function run(): Promise<void> {
  const t0 = performance.now()
  const secs = (t: number) => `${((performance.now() - t) / 1000).toFixed(1)}s`
  let phase = 'diagnostics'
  const beat = setInterval(() => log(`… still in ${phase} (${secs(t0)})`), 30_000)
  try {
    await diagnostics()
    if (params.has('diag')) return

    phase = 'processor'
    const proc = (await AutoProcessor.from_pretrained(REPO)) as unknown as Proc
    proc.image_processor.max_soft_tokens = IMAGE_TOKENS
    const opts = { padding: 'max_length', max_length: SEQ_LEN, truncation: true }
    const pics = { A: await picture(0, 'A'), 'A shifted': await picture(0, 'A', 12), B: await picture(200, 'B') }
    const inputs: Record<string, Record<string, { dims: number[] }>> = {}
    for (const [name, img] of Object.entries(pics)) inputs[name] = await proc(null, [[img]], null, null, opts)
    const shapes = Object.entries(inputs.A).map(([k, v]) => `${k}=[${v.dims}]`).join(' ')
    log(`processor ok ${secs(t0)}: ${shapes}`)

    // Pin every free dimension (names from the ONNX graphs) to this input.
    const patches = inputs.A.pixel_values.dims[1]
    const freeDimensionOverrides: Record<string, number> = {
      s11: 1, s35: patches, batch_size: 1, sequence_length: SEQ_LEN,
      num_image_tokens: 0, num_video_tokens: 0, num_audio_tokens: 0
    }
    log(`overrides ${JSON.stringify(freeDimensionOverrides)} (num_image_tokens set from the vision output)`)

    // Two ORT sessions, created one at a time, so a failure names its half.
    const sessions: Record<string, ort.InferenceSession> = {}
    // Rebuild as this ORT's tensors: transformers.js bundles its own ORT, and run() checks instanceof.
    const t = (inp: Record<string, unknown>, k: string) => {
      const x = inp[k] as { type: ort.Tensor.Type; data: ort.Tensor.DataType; dims: number[] }
      return new ort.Tensor(x.type, x.data, x.dims)
    }
    const vision = (inp: Record<string, unknown>) =>
      sessions.vision_encoder.run({ pixel_values: t(inp, 'pixel_values'), pixel_position_ids: t(inp, 'image_position_ids') })
    for (const part of ['vision_encoder', 'model']) {
      phase = `download ${part}`
      const tDl = performance.now()
      const suffix = dtype === 'fp32' ? '' : `_${dtype}`
      // fp16 text model: our graph without the 268 MB embedding table
      // (scripts/webnn_split_embed.py), which WebNN refuses to build.
      const split = part === 'model' && dtype === 'fp16' && device.startsWith('webnn')
      const graph = split
        ? new Uint8Array(await (await fetch(`${import.meta.env.BASE_URL}models/embeddinggemma2-webnn/model_fp16_noembed.onnx`)).arrayBuffer())
        : await cachedFetch(`${part}${suffix}.onnx`)
      const data = await cachedFetch(`${part}${suffix}.onnx_data`)
      if (split) embedTable = new Uint16Array(data.buffer, data.byteOffset + EMBED_OFFSET, EMBED_ROWS * EMBED_WIDTH)
      log(`download ${part} ${secs(tDl)} (${((graph.byteLength + data.byteLength) / 1e6).toFixed(0)} MB)`)
      phase = `compile ${part}`
      const tC = performance.now()
      sessions[part] = await Promise.race([
        ort.InferenceSession.create(graph, {
          executionProviders: [ep],
          freeDimensionOverrides,
          externalData: [{ path: `${part}${suffix}.onnx_data`, data }],
          logSeverityLevel: params.has('verbose') ? 0 : 2
        }),
        failed // WebNN build errors surface as unhandled rejections and the create never settles
      ])
      log(`COMPILE ${part} ok ${secs(tC)}`)
      if (part === 'vision_encoder') {
        // Timed on its own, so a text-half failure still says whether the NPU is fast.
        for (let rep = 1; rep <= 3; rep++) {
          phase = `vision run #${rep}`
          const tV = performance.now()
          const out = await vision(inputs.A)
          const d = out.image_features.data as Float32Array
          log(`VISION #${rep} ${secs(tV)} dims=[${out.image_features.dims}] nonFinite=${d.filter((x) => !Number.isFinite(x)).length}`)
          // Soft tokens depend on the aspect ratio (63 for 4:3, not the 70 budget).
          freeDimensionOverrides.num_image_tokens = out.image_features.dims[0]
        }
      }
    }
    const model: Model = async (inp) => {
      const tV = performance.now()
      const vis = await vision(inp)
      const vMs = performance.now() - tV
      const empty = new ort.Tensor('float32', new Float32Array(0), [0, 512])
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
        image_features: vis.image_features, video_features: empty, audio_features: empty
      })
      log(`  vision ${(vMs / 1000).toFixed(2)}s, text ${((performance.now() - tV - vMs) / 1000).toFixed(2)}s`)
      return { sentence_embedding: { data: out.sentence_embedding.data as Float32Array } }
    }

    const vecs: Record<string, Float32Array> = {}
    for (const [name, inp] of Object.entries(inputs)) {
      for (let rep = 0; rep < (name === 'A' ? 3 : 1); rep++) {
        phase = `embed ${name} #${rep + 1}`
        const t = performance.now()
        const out = await model(inp)
        const v = Float32Array.from(out.sentence_embedding.data)
        const nan = v.filter((x) => !Number.isFinite(x)).length
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0))
        log(`EMBED ${name} #${rep + 1} ${secs(t)} dim=${v.length} norm=${norm.toFixed(3)} nonFinite=${nan}`)
        vecs[name] = v
      }
    }
    // Sanity: a near-copy should score far above a different picture. fp16
    // overflow (why the app skips q4f16) shows up as NaN or collapsed scores.
    log(`COS A~A-shifted=${cos(vecs.A, vecs['A shifted']).toFixed(4)} A~B=${cos(vecs.A, vecs.B).toFixed(4)} (expect first ≫ second)`)

    // Compare with earlier runs on other devices in this browser.
    const key = 'npuTest:' + device + ':' + dtype
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i)!
        if (k.startsWith('npuTest:') && k !== key) {
          log(`COS vs ${k.slice(8)} (picture A) = ${cos(vecs.A, Float32Array.from(JSON.parse(localStorage.getItem(k)!))).toFixed(4)}`)
        }
      }
      localStorage.setItem(key, JSON.stringify(Array.from(vecs.A)))
    } catch {}
    log(`DONE pass total=${secs(t0)}`)
  } catch (e) {
    log(`FAIL in ${phase}: ${errMsg(e)}`)
    log(`DONE fail total=${secs(t0)}`)
  } finally {
    clearInterval(beat)
  }
}

void run()
