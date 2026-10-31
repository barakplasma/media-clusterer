import { makeScene, cameraAt, pct, VIEW_W, VIEW_H, FRAMES, Stats } from './scene'
import { createBaseline } from './baseline'
import { createOpt2d } from './opt2d'
import { createPixi } from './pixi'
import { createCuller } from './cull'
import { cullAndPrioritize } from '../../src/compute'

const mk = (): HTMLCanvasElement => { const c = document.createElement('canvas'); c.width = VIEW_W; c.height = VIEW_H; document.body.appendChild(c); return c }

/** Main-thread jank probe: a 4ms timer; its lateness shows how long the main thread was blocked. */
function probe() {
  const late: number[] = []; let last = performance.now()
  const id = setInterval(() => { const n = performance.now(); late.push(n - last - 4); last = n }, 4)
  return () => { clearInterval(id); return { mainLateP95: pct(late, .95), mainLateMax: Math.max(...late) } }
}

async function runMain(name: string, n: number, flush: boolean) {
  const scene = makeScene(n)
  const r = name === 'baseline-main' ? await createBaseline(mk(), scene)
    : name === 'opt2d-main' ? await createOpt2d(mk(), scene)
    : name === 'opt2d-all' ? await createOpt2d(mk(), scene, 8000)
    : name === 'logic-baseline' ? logicBaseline(scene)
    : name === 'logic-grid' ? logicGrid(scene)
    : await createPixi(mk(), scene, name === 'pixi-all')
  // warm up zoomed-out so sprite pools / caches reach steady state before measuring
  for (let f = 0; f < 12; f++) r.frame(cameraAt(120 + (f % 5), scene.side)); r.flush()
  const stop = probe()
  const d: number[] = []; let drawn = 0, f = 0, drawnSum = 0
  while (f < FRAMES) {
    const t = performance.now()
    drawn = r.frame(cameraAt(f++, scene.side)); drawnSum += drawn
    if (flush) r.flush()
    d.push(performance.now() - t)
    await new Promise((res) => setTimeout(res, 0))
  }
  return { renderer: name, n, frames: d.length, p50: pct(d, .5), p95: pct(d, .95), max: Math.max(...d), drawn: drawnSum / d.length, ...stop() }
}

// Culling/selection only, no drawing: the hardware-independent part of the per-frame cost.
function logicBaseline(scene: any) {
  const pts: [number, number][] = Array.from({ length: scene.n }, (_, i) => [scene.xy[2 * i], scene.xy[2 * i + 1]])
  return {
    frame(cam: any) {
      new Int32Array(pts.length).fill(pts.length)
      let minD = Infinity
      for (let i = 0; i < pts.length; i++) { const dx = pts[i][0] - cam.x, dy = pts[i][1] - cam.y, d = dx * dx + dy * dy; if (d < minD) minD = d }
      return cullAndPrioritize(pts, cam, VIEW_W, VIEW_H, 24 * cam.scale, null, 400).length
    },
    flush() {},
  }
}
function logicGrid(scene: any) { const q = createCuller(scene); return { frame: (cam: any) => q(cam, 400).n, flush() {} } }

async function runWorker(n: number, flush: boolean) {
  const canvas = mk().transferControlToOffscreen()
  const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  const stop = probe()
  const res = await new Promise<Stats>((ok) => { w.onmessage = (e) => ok(e.data); w.postMessage({ n, flush, canvas }, [canvas]) })
  w.terminate()
  return { ...res, ...stop() }
}

;(window as any).bench = (name: string, n: number, flush: boolean) => (name === 'opt2d-worker' ? runWorker(n, flush) : runMain(name, n, flush))
