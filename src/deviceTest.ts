/**
 * Headless smoke test for the EmbeddingGemma 2 path on a real phone
 * (device-test.html, driven by Firebase Test Lab). Every step logs one line
 * prefixed EG2TEST so it can be grepped out of logcat (`chromium` tag).
 *
 * Inputs are drawn on a canvas, so the page needs no bundled media.
 */

import { embedTextWithVlm, embedWithVlm, loadVlm } from './vlm'

const PREFIX = 'EG2TEST'
const logEl = document.getElementById('log')!

function log(msg: string): void {
  console.log(`${PREFIX} ${msg}`)
  logEl.textContent += msg + '\n'
}

function fail(step: string, err: unknown): void {
  const e = err as { name?: string; message?: string }
  console.error(`${PREFIX} FAIL ${step}: ${e?.name ?? 'Error'}: ${e?.message ?? String(err)}`)
  logEl.textContent += `FAIL ${step}: ${e?.message ?? String(err)}\n`
}

/** A distinguishable synthetic picture: gradient, shapes and a label. */
async function drawImage(w: number, h: number, hue: number, label: string): Promise<ImageBitmap> {
  const c = new OffscreenCanvas(w, h)
  const g = c.getContext('2d')!
  const grad = g.createLinearGradient(0, 0, w, h)
  grad.addColorStop(0, `hsl(${hue} 80% 55%)`)
  grad.addColorStop(1, `hsl(${(hue + 120) % 360} 70% 30%)`)
  g.fillStyle = grad
  g.fillRect(0, 0, w, h)
  g.fillStyle = 'white'
  g.beginPath()
  g.arc(w / 2, h / 2, Math.min(w, h) / 4, 0, Math.PI * 2)
  g.fill()
  g.fillStyle = 'black'
  g.font = `${Math.round(Math.min(w, h) / 8)}px sans-serif`
  g.fillText(label, w / 10, h / 5)
  return createImageBitmap(c)
}

function norm(v: Float32Array): number {
  let s = 0
  for (const x of v) s += x * x
  return Math.sqrt(s)
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

async function gpuInfo(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
  if (!gpu) {
    log('gpu none (navigator.gpu missing)')
    return
  }
  const adapter = await gpu.requestAdapter()
  if (!adapter) {
    log('gpu adapter null')
    return
  }
  const i = adapter.info
  const l = adapter.limits
  log(
    `gpu vendor=${i.vendor} arch=${i.architecture} device=${i.device} desc=${i.description} ` +
      `maxBuffer=${l.maxBufferSize} maxStorageBinding=${l.maxStorageBufferBindingSize} ` +
      `maxWorkgroupStorage=${l.maxComputeWorkgroupStorageSize} f16=${adapter.features.has('shader-f16')}`
  )
}

async function run(): Promise<void> {
  const t0 = performance.now()
  const since = (t: number) => `${((performance.now() - t) / 1000).toFixed(1)}s`
  log(`START ua=${navigator.userAgent} build=${__GIT_COMMIT__}`)
  await gpuInfo()

  let t = performance.now()
  let lastPct = -10
  try {
    const tier = await loadVlm('embeddinggemma2', 'https://huggingface.co', (pct) => {
      if (pct - lastPct >= 10) {
        lastPct = pct
        log(`download ${pct.toFixed(0)}%`)
      }
    })
    log(`LOAD ok ${since(t)} imageTokens=${tier.imageTokens} frames=${tier.framesPerVideo} seq=${tier.fixedSeqLen}`)
  } catch (err) {
    fail('load', err)
    log(`DONE fail total=${since(t0)}`)
    return
  }

  const items: { name: string; vec: Float32Array }[] = []
  const photos: [string, number, number, number][] = [
    ['landscape-4x3', 640, 480, 0],
    ['portrait-3x4', 480, 640, 200],
    ['square', 512, 512, 100],
    ['panorama-21x9', 1050, 450, 300]
  ]
  let failures = 0
  for (const [name, w, h, hue] of photos) {
    t = performance.now()
    try {
      const [vec] = await embedWithVlm([[await drawImage(w, h, hue, name)]])
      items.push({ name, vec })
      log(`PHOTO ${name} ok ${since(t)} dim=${vec.length} norm=${norm(vec).toFixed(3)}`)
    } catch (err) {
      failures++
      fail(`photo ${name}`, err)
    }
  }

  t = performance.now()
  try {
    const frames = await Promise.all([0, 1, 2, 3].map((k) => drawImage(480, 270, k * 60, `frame ${k}`)))
    const [vec] = await embedWithVlm([frames])
    items.push({ name: 'video-4frames', vec })
    log(`VIDEO 4 frames ok ${since(t)} dim=${vec.length} norm=${norm(vec).toFixed(3)}`)
  } catch (err) {
    failures++
    fail('video', err)
  }

  const queries = ['a white circle on a red background', 'a tall portrait photo', 'a wide panorama', 'a video of colourful frames']
  for (const q of queries) {
    t = performance.now()
    try {
      const [qv] = await embedTextWithVlm(['task: search result | query: ' + q])
      const ranked = items.map((it) => ({ n: it.name, s: dot(qv, it.vec) })).sort((a, b) => b.s - a.s)
      log(`SEARCH "${q}" ok ${since(t)} top=${ranked.slice(0, 3).map((r) => `${r.n}:${r.s.toFixed(3)}`).join(',')}`)
    } catch (err) {
      failures++
      fail(`search "${q}"`, err)
    }
  }

  log(`DONE ${failures === 0 ? 'pass' : `fail failures=${failures}`} total=${since(t0)}`)
}

window.addEventListener('error', (e) => console.error(`${PREFIX} window.error ${e.message}`))
window.addEventListener('unhandledrejection', (e) => console.error(`${PREFIX} unhandledrejection ${String(e.reason)}`))
void run()
