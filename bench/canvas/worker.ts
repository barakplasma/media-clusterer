// Worker host for the optimized Canvas2D renderer on an OffscreenCanvas; drives its own frame loop.
import { createOpt2d } from './opt2d'
import { makeScene, cameraAt, pct, VIEW_W, VIEW_H, FRAMES } from './scene'

self.onmessage = async (e: MessageEvent) => {
  const { n, flush, canvas } = e.data as { n: number; flush: boolean; canvas: OffscreenCanvas }
  const scene = makeScene(n)
  const r = await createOpt2d(canvas, scene)
  for (let f = 0; f < 12; f++) r.frame(cameraAt(120 + (f % 5), scene.side)); r.flush() // warm up zoomed-out
  const d: number[] = []; let drawn = 0, f = 0
  while (f < FRAMES) {
    const t = performance.now()
    drawn += r.frame(cameraAt(f++, scene.side)); if (flush) r.flush()
    d.push(performance.now() - t)
    await new Promise((res) => setTimeout(res, 0)) // yield like the main-thread variants do
  }
  ;(self as any).postMessage({ renderer: 'opt2d-worker', n, frames: d.length, p50: pct(d, .5), p95: pct(d, .95), max: Math.max(...d), drawn: drawn / d.length })
}
void VIEW_W; void VIEW_H
