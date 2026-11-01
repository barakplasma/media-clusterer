// Same slice (status + progress + toast) driven through the old vanilla code and the Solid code.
import { mountChrome, setStatus, setProgress, showToast } from '../../src/ui/chrome'

// Verbatim copy of the pre-Solid implementation from src/app.ts.
const $ = (id: string) => document.getElementById(id)!
const v = {
  setStatus(msg: string) { $('status').textContent = msg; $('welcome-status').textContent = msg },
  setProgress(pct: number) {
    const w = `${Math.min(100, Math.max(0, pct))}%`
    $('progress-bar').style.width = w
    $('welcome-bar').style.width = w
  },
  toastTimer: null as ReturnType<typeof setTimeout> | null,
  showToast(message: string, level = 'info', durationMs = 5000) {
    const el = $('toast')
    el.textContent = message
    el.className = `visible ${level === 'info' ? '' : level}`.trim()
    if (v.toastTimer) clearTimeout(v.toastTimer)
    v.toastTimer = setTimeout(() => { el.classList.remove('visible'); v.toastTimer = null }, durationMs)
  }
}
const s = { setStatus, setProgress, showToast }
const impls = { vanilla: v, solid: s }
type Impl = typeof v | typeof s
mountChrome()

const pct = (a: number[], p: number) => a.slice().sort((x, y) => x - y)[Math.floor((a.length - 1) * p)]

// Sync: tight loop, no yielding. Pure cost of the update path including style invalidation.
function sync(i: Impl, n: number) {
  const t = performance.now()
  for (let k = 0; k < n; k++) { i.setStatus(`Embedding ${k}/${n}`); i.setProgress((k / n) * 100) }
  return performance.now() - t
}
function toastSync(i: Impl, n: number) {
  const t = performance.now()
  for (let k = 0; k < n; k++) i.showToast(`msg ${k}`, k % 3 ? 'warn' : 'info', 60000)
  return performance.now() - t
}
// Realistic: an embed loop updating every ~1ms for `ms`, frames measured via rAF.
function frames(i: Impl, ms: number) {
  return new Promise<{ p50: number; p95: number; max: number; frames: number; updates: number }>((res) => {
    const d: number[] = []; let last = performance.now(); let k = 0; const end = last + ms
    const raf = () => { const n = performance.now(); d.push(n - last); last = n; if (n < end) requestAnimationFrame(raf) }
    requestAnimationFrame(raf)
    const tick = () => {
      if (performance.now() >= end) return res({ p50: pct(d, .5), p95: pct(d, .95), max: Math.max(...d), frames: d.length, updates: k })
      i.setStatus(`Embedding ${k}`); i.setProgress((k++ % 1000) / 10); setTimeout(tick, 0)
    }
    tick()
  })
}
;(window as any).bench = async (name: 'vanilla' | 'solid') => {
  const i = impls[name]
  sync(i, 2000) // warm up JIT
  const syncRuns = [0, 1, 2, 3, 4].map(() => sync(i, 50000))
  const toastRuns = [0, 1, 2, 3, 4].map(() => toastSync(i, 5000))
  const f = await frames(i, 3000)
  return { name, syncMs50k: pct(syncRuns, .5), toastMs5k: pct(toastRuns, .5), ...f }
}
