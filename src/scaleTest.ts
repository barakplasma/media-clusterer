/**
 * Device-test branch only: a scale test of the real app pipeline on a phone.
 *
 * Active only with ?eg2scale in the URL. It clicks Load model like a user,
 * generates ~20 media files on the page (JPEG/PNG/WebP stills of several
 * aspect ratios incl. 4000x3000, MediaRecorder WebMs, bundled H.264 MP4s, a
 * truncated MP4 and a corrupt JPEG), feeds them to window.__debug.processFiles,
 * then runs text searches. Every line is prefixed EG2TEST and mirrored into an
 * on-page <pre> that the Test Lab harness reads back through accessibility.
 */

import type { PhotoFile } from './types'

interface Debug {
  state: {
    phase: string
    files: PhotoFile[]
    searchResults: ArrayLike<number> | null
    searchScores: Float32Array | number[] | null
  }
  embedFailed: Set<number>
  processFiles(files: PhotoFile[]): Promise<void>
  searchImages(query: string): Promise<void>
}

const PREFIX = 'EG2TEST'
const BROKEN = ['broken-truncated.mp4', 'broken-corrupt.jpg']

let logEl: HTMLPreElement
const origError = console.error.bind(console)
const errors: string[] = []

function log(msg: string): void {
  console.log(`${PREFIX} ${msg}`)
  logEl.textContent += msg + '\n'
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const now = () => performance.now()
const secs = (t: number) => `${((now() - t) / 1000).toFixed(1)}s`

function memory(): string {
  const m = (performance as Performance & { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory
  return m ? `heap=${(m.usedJSHeapSize / 2 ** 20).toFixed(0)}MB/${(m.jsHeapSizeLimit / 2 ** 20).toFixed(0)}MB` : 'heap=n/a'
}

function text(id: string): string {
  return (document.getElementById(id)?.textContent ?? '').trim().replace(/\s+/g, ' ')
}

// ── Scenes: distinct, searchable content ────────────────────────────────────
type Scene = 'sunset' | 'night' | 'forest' | 'checker' | 'hello' | 'ocean' | 'stripes' | 'circles' | 'snow' | 'desert' | 'flowers' | 'city'

function drawScene(g: OffscreenCanvasRenderingContext2D, w: number, h: number, scene: Scene, seed: number): void {
  const rnd = mulberry32(seed)
  const s = Math.min(w, h)
  const sky = (top: string, bottom: string, until = h) => {
    const grad = g.createLinearGradient(0, 0, 0, until)
    grad.addColorStop(0, top)
    grad.addColorStop(1, bottom)
    g.fillStyle = grad
    g.fillRect(0, 0, w, until)
  }
  switch (scene) {
    case 'sunset':
      sky('#2b1055', '#ff7e29', h * 0.65)
      g.fillStyle = '#ffd23f'
      g.beginPath()
      g.arc(w / 2, h * 0.62, s * 0.18, 0, Math.PI * 2)
      g.fill()
      g.fillStyle = '#1b3a5c'
      g.fillRect(0, h * 0.65, w, h * 0.35)
      for (let i = 0; i < 12; i++) {
        g.fillStyle = `rgba(255,170,60,${0.2 + rnd() * 0.4})`
        g.fillRect(w * (0.3 + rnd() * 0.4), h * (0.67 + i * 0.025), w * 0.15 * rnd(), h * 0.006)
      }
      break
    case 'night':
      sky('#02030f', '#0b1a3d')
      for (let i = 0; i < 400; i++) {
        g.fillStyle = `rgba(255,255,255,${0.4 + rnd() * 0.6})`
        g.beginPath()
        g.arc(rnd() * w, rnd() * h, s * 0.002 * (1 + rnd() * 2), 0, Math.PI * 2)
        g.fill()
      }
      g.fillStyle = '#f4f1c9'
      g.beginPath()
      g.arc(w * 0.75, h * 0.25, s * 0.08, 0, Math.PI * 2)
      g.fill()
      break
    case 'forest':
      sky('#9fd3f5', '#e6f6ff', h * 0.5)
      g.fillStyle = '#3d7a2e'
      g.fillRect(0, h * 0.5, w, h * 0.5)
      for (let i = 0; i < 18; i++) {
        const x = rnd() * w
        const base = h * (0.55 + rnd() * 0.4)
        const th = s * (0.2 + rnd() * 0.2)
        g.fillStyle = `hsl(${110 + rnd() * 30} 60% ${18 + rnd() * 15}%)`
        g.beginPath()
        g.moveTo(x, base - th)
        g.lineTo(x - th * 0.35, base)
        g.lineTo(x + th * 0.35, base)
        g.closePath()
        g.fill()
        g.fillStyle = '#5a3a1a'
        g.fillRect(x - th * 0.04, base, th * 0.08, th * 0.12)
      }
      break
    case 'checker': {
      const n = 8
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
          g.fillStyle = (x + y) % 2 ? '#111' : '#f5f5f5'
          g.fillRect((x * w) / n, (y * h) / n, w / n + 1, h / n + 1)
        }
      break
    }
    case 'hello':
      g.fillStyle = '#fffbe6'
      g.fillRect(0, 0, w, h)
      g.fillStyle = '#c0182b'
      g.font = `bold ${Math.round(s * 0.3)}px sans-serif`
      g.textAlign = 'center'
      g.textBaseline = 'middle'
      g.fillText('HELLO', w / 2, h / 2)
      break
    case 'ocean':
      sky('#87ceeb', '#d8f1ff', h * 0.4)
      g.fillStyle = '#0d5c91'
      g.fillRect(0, h * 0.4, w, h * 0.6)
      g.strokeStyle = 'rgba(255,255,255,0.7)'
      g.lineWidth = s * 0.006
      for (let row = 0; row < 14; row++) {
        const y0 = h * (0.43 + row * 0.04)
        g.beginPath()
        for (let x = 0; x <= w; x += w / 60) g.lineTo(x, y0 + Math.sin(x / (w / 12) + row) * h * 0.008)
        g.stroke()
      }
      break
    case 'stripes':
      for (let i = 0; i < 14; i++) {
        g.fillStyle = `hsl(${i * 26} 80% 55%)`
        g.fillRect((i * w) / 14, 0, w / 14 + 1, h)
      }
      break
    case 'circles':
      g.fillStyle = '#101820'
      g.fillRect(0, 0, w, h)
      for (let i = 10; i > 0; i--) {
        g.fillStyle = i % 2 ? '#f2aa4c' : '#101820'
        g.beginPath()
        g.arc(w / 2, h / 2, (s * 0.48 * i) / 10, 0, Math.PI * 2)
        g.fill()
      }
      break
    case 'snow':
      sky('#cfd8e3', '#ffffff')
      g.fillStyle = '#ffffff'
      g.beginPath()
      g.moveTo(0, h * 0.7)
      g.quadraticCurveTo(w * 0.3, h * 0.55, w * 0.6, h * 0.68)
      g.quadraticCurveTo(w * 0.8, h * 0.75, w, h * 0.62)
      g.lineTo(w, h)
      g.lineTo(0, h)
      g.fill()
      for (let i = 0; i < 300; i++) {
        g.fillStyle = 'rgba(255,255,255,0.9)'
        g.beginPath()
        g.arc(rnd() * w, rnd() * h * 0.7, s * 0.004, 0, Math.PI * 2)
        g.fill()
      }
      g.fillStyle = '#7a8796'
      g.beginPath()
      g.moveTo(w * 0.2, h * 0.65)
      g.lineTo(w * 0.42, h * 0.25)
      g.lineTo(w * 0.64, h * 0.65)
      g.fill()
      break
    case 'desert':
      sky('#f9d77e', '#fff3c4', h * 0.55)
      g.fillStyle = '#d9a35b'
      g.fillRect(0, h * 0.55, w, h * 0.45)
      g.fillStyle = '#2f6b2f'
      for (let i = 0; i < 3; i++) {
        const x = w * (0.2 + i * 0.3)
        g.fillRect(x, h * 0.45, s * 0.04, h * 0.25)
        g.fillRect(x - s * 0.06, h * 0.52, s * 0.06, s * 0.025)
        g.fillRect(x - s * 0.06, h * 0.46, s * 0.025, h * 0.07)
      }
      break
    case 'flowers':
      g.fillStyle = '#4c9a2a'
      g.fillRect(0, 0, w, h)
      for (let i = 0; i < 70; i++) {
        const x = rnd() * w
        const y = rnd() * h
        const r = s * (0.02 + rnd() * 0.025)
        g.fillStyle = `hsl(${[0, 50, 300, 330, 20][i % 5]} 90% 60%)`
        for (let p = 0; p < 6; p++) {
          g.beginPath()
          g.arc(x + Math.cos((p * Math.PI) / 3) * r, y + Math.sin((p * Math.PI) / 3) * r, r * 0.7, 0, Math.PI * 2)
          g.fill()
        }
        g.fillStyle = '#ffd400'
        g.beginPath()
        g.arc(x, y, r * 0.6, 0, Math.PI * 2)
        g.fill()
      }
      break
    case 'city':
      sky('#1e2a4a', '#f08a4b')
      for (let i = 0; i < 24; i++) {
        const bw = w / 24
        const bh = h * (0.25 + rnd() * 0.55)
        g.fillStyle = '#141414'
        g.fillRect(i * bw, h - bh, bw * 0.92, bh)
        g.fillStyle = '#ffe28a'
        for (let wy = h - bh + bw * 0.3; wy < h - bw * 0.3; wy += bw * 0.5)
          if (rnd() > 0.4) g.fillRect(i * bw + bw * 0.2, wy, bw * 0.15, bw * 0.2)
      }
      break
  }
}

function mulberry32(a: number): () => number {
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

async function makeStill(name: string, w: number, h: number, type: string, scene: Scene, seed: number): Promise<File> {
  const c = new OffscreenCanvas(w, h)
  drawScene(c.getContext('2d')!, w, h, scene, seed)
  const blob = await c.convertToBlob({ type, quality: 0.9 })
  if (blob.type !== type) throw new Error(`${type} encode unsupported (got ${blob.type})`)
  return new File([blob], name, { type, lastModified: Date.now() - seed * 60_000 })
}

/** Record an animated canvas with MediaRecorder: duration comes out Infinity. */
async function recordWebm(name: string, seconds: number, variant: number): Promise<File> {
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m))
  if (!mime) throw new Error('no WebM MediaRecorder')
  const c = document.createElement('canvas')
  c.width = 640
  c.height = 360
  const g = c.getContext('2d')!
  const rec = new MediaRecorder(c.captureStream(30), { mimeType: mime, videoBitsPerSecond: 1_500_000 })
  const chunks: Blob[] = []
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
  const stopped = new Promise((r) => (rec.onstop = r))
  const t0 = now()
  const draw = () => {
    const t = (now() - t0) / 1000
    if (variant === 0) {
      // A red ball bouncing across a blue sky, the background darkening.
      g.fillStyle = `hsl(210 70% ${70 - t * 8}%)`
      g.fillRect(0, 0, 640, 360)
      g.fillStyle = '#e53935'
      g.beginPath()
      g.arc(40 + ((t * 140) % 560), 300 - Math.abs(Math.sin(t * 3)) * 220, 34, 0, Math.PI * 2)
      g.fill()
    } else {
      // A rotating yellow square over a background cycling through hues.
      g.fillStyle = `hsl(${(t * 70) % 360} 70% 45%)`
      g.fillRect(0, 0, 640, 360)
      g.save()
      g.translate(320, 180)
      g.rotate(t * 1.5)
      g.fillStyle = '#ffeb3b'
      g.fillRect(-70, -70, 140, 140)
      g.restore()
    }
  }
  draw()
  rec.start(250)
  const timer = setInterval(draw, 33)
  await sleep(seconds * 1000)
  clearInterval(timer)
  rec.stop()
  await stopped
  const blob = new Blob(chunks, { type: 'video/webm' })
  return new File([blob], name, { type: 'video/webm', lastModified: Date.now() - (100 + variant) * 60_000 })
}

async function fetchFile(path: string, name: string, type: string, truncate?: number): Promise<File> {
  const buf = await (await fetch(path)).arrayBuffer()
  const bytes = truncate ? buf.slice(0, Math.floor(buf.byteLength * truncate)) : buf
  return new File([bytes], name, { type, lastModified: Date.now() - 200 * 60_000 })
}

function corruptJpeg(): File {
  // A valid SOI/APP0 header followed by noise: sniffs as JPEG, cannot decode.
  const head = [0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]
  const body = new Uint8Array(40_000)
  const rnd = mulberry32(7)
  for (let i = 0; i < body.length; i++) body[i] = Math.floor(rnd() * 256)
  return new File([new Uint8Array(head), body], 'broken-corrupt.jpg', { type: 'image/jpeg', lastModified: Date.now() - 300 * 60_000 })
}

async function gpuInfo(): Promise<void> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
  const adapter = gpu ? await gpu.requestAdapter() : null
  if (!adapter) return log('gpu none')
  const i = adapter.info
  const l = adapter.limits
  log(
    `gpu vendor=${i.vendor} arch=${i.architecture} maxBuffer=${l.maxBufferSize} ` +
      `maxStorageBinding=${l.maxStorageBufferBindingSize} f16=${adapter.features.has('shader-f16')}`
  )
}

async function waitFor<T>(what: string, fn: () => T | undefined | null | false, timeoutMs: number): Promise<T> {
  const t0 = now()
  for (;;) {
    const v = fn()
    if (v) return v
    if (now() - t0 > timeoutMs) throw new Error(`timed out waiting for ${what}`)
    await sleep(250)
  }
}

async function run(): Promise<void> {
  const t0 = now()
  const params = new URLSearchParams(location.search)
  log(`START ua=${navigator.userAgent} build=${__GIT_COMMIT__} cache=${params.get('cache') ?? 'lazyRelease'}`)
  await gpuInfo()
  log(
    `mediarecorder webm-vp9=${MediaRecorder.isTypeSupported('video/webm;codecs=vp9')} ` +
      `webm-vp8=${MediaRecorder.isTypeSupported('video/webm;codecs=vp8')} ` +
      `mp4=${MediaRecorder.isTypeSupported('video/mp4')} mp4-avc1=${MediaRecorder.isTypeSupported('video/mp4;codecs=avc1')}`
  )

  // Status and toast as the user sees them.
  let lastStatus = ''
  const pcts: string[] = []
  new MutationObserver(() => {
    const s = text('status')
    if (s === lastStatus) return
    lastStatus = s
    const m = s.match(/Downloading .*?(\d+)%/)
    if (m) {
      if (pcts[pcts.length - 1] !== m[1]) pcts.push(m[1])
    } else {
      log(`status ${s.slice(0, 160)}`)
    }
  }).observe(document.getElementById('status')!, { childList: true, characterData: true, subtree: true })
  let lastToast = ''
  const toastEl = document.getElementById('toast')
  if (toastEl)
    new MutationObserver(() => {
      const s = text('toast')
      if (s && s !== lastToast) log(`toast ${s.slice(0, 200)}`)
      lastToast = s
    }).observe(toastEl, { childList: true, characterData: true, subtree: true, attributes: true })

  const debug = await waitFor('window.__debug', () => (window as Window & { __debug?: Debug }).__debug, 60_000)
  // ── Load, as the app does: it starts loading on page open. The button is
  // only clickable when that failed (its retry state), so click it then.
  let t = now()
  let beatLabel = 'load'
  const beat = setInterval(() => log(`wait ${beatLabel} ${secs(t)} ${memory()}`), 30_000)
  const btn = document.getElementById('load-model-btn') as HTMLButtonElement
  log(`load-button hidden=${btn.hidden} disabled=${btn.disabled} phase=${debug.state.phase} status="${text('status')}"`)
  if (!btn.hidden && !btn.disabled) btn.click()
  try {
    await waitFor('model_ready', () => debug.state.phase === 'model_ready' || /fail|error/i.test(lastStatus), 30 * 60_000)
  } finally {
    log(`progress ${pcts.join(',')}`)
  }
  if (debug.state.phase !== 'model_ready') {
    clearInterval(beat)
    log(`FAIL load: ${lastStatus}`)
    log(`DONE fail total=${secs(t0)}`)
    return
  }
  log(`LOAD ok ${secs(t)} ${memory()}`)

  // ── Generate media on the device ──
  t = now()
  beatLabel = 'generate'
  const stills: [string, number, number, string, Scene][] = [
    ['p01-sunset-large.jpg', 4000, 3000, 'image/jpeg', 'sunset'],
    ['p02-night-large.jpg', 3000, 4000, 'image/jpeg', 'night'],
    ['p03-forest-large.jpg', 4000, 3000, 'image/jpeg', 'forest'],
    ['p04-checker.jpg', 1200, 1200, 'image/jpeg', 'checker'],
    ['p05-hello.jpg', 1920, 1080, 'image/jpeg', 'hello'],
    ['p06-ocean.jpg', 1080, 1920, 'image/jpeg', 'ocean'],
    ['p07-stripes.jpg', 2100, 900, 'image/jpeg', 'stripes'],
    ['p08-circles.jpg', 1600, 1200, 'image/jpeg', 'circles'],
    ['p09-sunset.png', 1200, 1600, 'image/png', 'sunset'],
    ['p10-snow.png', 1024, 1024, 'image/png', 'snow'],
    ['p11-desert.png', 1920, 1080, 'image/png', 'desert'],
    ['p12-night.webp', 1200, 900, 'image/webp', 'night'],
    ['p13-flowers.webp', 900, 1200, 'image/webp', 'flowers'],
    ['p14-city.webp', 2100, 900, 'image/webp', 'city']
  ]
  const files: File[] = []
  for (const [i, [name, w, h, type, scene]] of stills.entries()) {
    try {
      files.push(await makeStill(name, w, h, type, scene, i + 1))
    } catch (err) {
      log(`gen-fail ${name}: ${(err as Error).message}`)
    }
  }
  for (const [i, secsLen] of [4, 5].entries()) {
    try {
      files.push(await recordWebm(`v0${i + 1}-recorded.webm`, secsLen, i))
    } catch (err) {
      log(`gen-fail webm ${i}: ${(err as Error).message}`)
    }
  }
  files.push(await fetchFile('/eg2/clip-testpattern.mp4', 'v03-testpattern.mp4', 'video/mp4'))
  files.push(await fetchFile('/eg2/clip-fractal.mp4', 'v04-fractal.mp4', 'video/mp4'))
  files.push(await fetchFile('/eg2/clip-testpattern.mp4', 'broken-truncated.mp4', 'video/mp4', 0.4))
  files.push(corruptJpeg())
  // ?small: a quick harness check (one still, one WebM, both broken files).
  if (params.has('small')) {
    const keep = new Set(['p05-hello.jpg', 'v01-recorded.webm', ...BROKEN])
    files.splice(0, files.length, ...files.filter((f) => keep.has(f.name)))
  }
  log(`generated ${files.length} files in ${secs(t)}: ${files.map((f) => `${f.name}(${(f.size / 1024).toFixed(0)}KB)`).join(' ')}`)

  // ── The real pipeline ──
  t = now()
  beatLabel = 'processFiles'
  const photoFiles: PhotoFile[] = files.map((file) => ({
    name: file.name,
    size: file.size,
    lastModified: file.lastModified,
    file,
    objectURL: null
  }))
  try {
    await debug.processFiles(photoFiles)
  } catch (err) {
    log(`FAIL processFiles threw: ${(err as Error).message}`)
  }
  const embedMs = now() - t
  const names = debug.state.files.map((f) => f.name)
  const failed = [...debug.embedFailed].map((i) => names[i]).sort()
  log(
    `PROCESS done ${(embedMs / 1000).toFixed(1)}s items=${names.length} avg=${(embedMs / 1000 / Math.max(1, names.length)).toFixed(1)}s/item ${memory()}`
  )
  log(`failed [${failed.join(', ')}] chip="${text('failed-chip')}"`)
  log(`status-final "${lastStatus}"`)

  // ── Search ──
  beatLabel = 'search'
  const queries: [string, string][] = [
    ['a red sunset over the sea', 'sunset'],
    ['stars in a night sky', 'night'],
    ['a black and white checkerboard', 'checker'],
    ['green pine trees', 'forest'],
    ['the word HELLO', 'hello']
  ]
  let searchFails = 0
  for (const [q, expect] of queries) {
    t = now()
    try {
      await debug.searchImages(q)
      // searchResults is a typed array; map() on it would coerce labels to numbers.
      const idx = Array.from(debug.state.searchResults ?? [])
      const scores = debug.state.searchScores
      const top = idx.slice(0, 3).map((i) => `${names[i]}:${scores ? Number(scores[i]).toFixed(3) : '?'}`)
      const hit = top.some((s) => s.includes(expect)) ? 'hit' : 'miss'
      if (!idx.length) searchFails++
      log(`SEARCH "${q}" ${secs(t)} ${hit} top=${top.join(',')}`)
    } catch (err) {
      searchFails++
      log(`SEARCH "${q}" threw ${(err as Error).message}`)
    }
  }
  clearInterval(beat)

  const lost = errors.some((e) => /device.?lost|VK_ERROR|OUT_OF_DEVICE/i.test(e)) || /lost/i.test(lastStatus)
  const isolationOk = failed.length === BROKEN.length && BROKEN.every((b) => failed.includes(b))
  log(`errors ${errors.length}${errors.length ? ': ' + errors.slice(0, 5).join(' | ') : ''}`)
  const pass = isolationOk && searchFails === 0 && !lost && names.length === files.length
  log(
    `DONE ${pass ? 'pass' : 'fail'} isolation=${isolationOk} searchFails=${searchFails} deviceLost=${lost} total=${secs(t0)}`
  )
}

if (new URLSearchParams(location.search).has('eg2scale')) {
  logEl = document.createElement('pre')
  logEl.id = 'eg2log'
  logEl.style.cssText =
    'position:fixed;left:0;right:0;bottom:0;max-height:40vh;overflow:auto;margin:0;z-index:99999;' +
    'background:rgba(0,0,0,.8);color:#0f0;font:10px monospace;pointer-events:none;white-space:pre-wrap'
  document.body.appendChild(logEl)
  console.error = (...args: unknown[]) => {
    const msg = args.map((a) => (a instanceof Error ? `${a.name}: ${a.message}` : String(a))).join(' ')
    errors.push(msg.slice(0, 300))
    logEl.textContent += `console.error ${msg.slice(0, 300)}\n`
    origError(...args)
  }
  window.addEventListener('unhandledrejection', (e) => log(`unhandledrejection ${String(e.reason)}`))
  // A modal <dialog> (the first-visit About box on a fresh device, or the
  // model-fallback prompt after a failed load) makes this log inert, which
  // hides it from the Test Lab harness. Close them and log what they said.
  setInterval(() => {
    for (const d of document.querySelectorAll<HTMLDialogElement>('dialog[open]')) {
      const said = (d.textContent ?? '').trim().replace(/\s+/g, ' ').slice(0, 160)
      log(`closed dialog #${d.id}: ${said}`)
      d.close()
    }
  }, 1000)
  run().catch((err) => {
    log(`FAIL harness: ${(err as Error).message}`)
    log('DONE fail')
  })
}
