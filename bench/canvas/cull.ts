// Uniform-grid culler shared by the optimized 2D and Pixi renderers. Buffers are reused across frames.
import { selectSmallest } from '../../src/compute'
import { Scene, THUMB_WORLD, VIEW_W, VIEW_H } from './scene'

const CELL = 128

export function createCuller(scene: Scene) {
  const { n, xy } = scene
  const gw = Math.ceil((scene.side * 1.3) / CELL) + 2, gh = gw
  const cellOf = (x: number, y: number) =>
    Math.min(gh - 1, Math.max(0, Math.floor(y / CELL))) * gw + Math.min(gw - 1, Math.max(0, Math.floor(x / CELL)))
  const cellStart = new Int32Array(gw * gh + 1), items = new Int32Array(n)
  for (let i = 0; i < n; i++) cellStart[cellOf(xy[2 * i], xy[2 * i + 1]) + 1]++
  for (let c = 0; c < gw * gh; c++) cellStart[c + 1] += cellStart[c]
  const fill = cellStart.slice(0, -1)
  for (let i = 0; i < n; i++) items[fill[cellOf(xy[2 * i], xy[2 * i + 1])]++] = i
  let cand = new Int32Array(4096), keys = new Float64Array(4096), order = new Int32Array(4096)
  const out = { idx: new Int32Array(0), n: 0, active: -1 }
  let sel = new Int32Array(4096)
  return function query(cam: { x: number; y: number; scale: number }, budget: number) {
    const s = cam.scale, half = (THUMB_WORLD * s) / 2
    const wx0 = cam.x - (VIEW_W / 2 + half) / s, wx1 = cam.x + (VIEW_W / 2 + half) / s
    const wy0 = cam.y - (VIEW_H / 2 + half) / s, wy1 = cam.y + (VIEW_H / 2 + half) / s
    const cx0 = Math.max(0, Math.floor(wx0 / CELL)), cx1 = Math.min(gw - 1, Math.floor(wx1 / CELL))
    const cy0 = Math.max(0, Math.floor(wy0 / CELL)), cy1 = Math.min(gh - 1, Math.floor(wy1 / CELL))
    let m = 0
    for (let cy = cy0; cy <= cy1; cy++) for (let cx = cx0; cx <= cx1; cx++) {
      const c = cy * gw + cx
      for (let p = cellStart[c]; p < cellStart[c + 1]; p++) {
        const i = items[p], x = xy[2 * i], y = xy[2 * i + 1]
        if (x < wx0 || x > wx1 || y < wy0 || y > wy1) continue
        if (m === cand.length) {
          const grow = (a: any, T: any) => { const b = new T(a.length * 2); b.set(a); return b }
          cand = grow(cand, Int32Array); keys = grow(keys, Float64Array); order = grow(order, Int32Array)
        }
        cand[m] = i
        const dx = x - cam.x, dy = y - cam.y
        keys[m] = dx * dx + dy * dy
        order[m] = m
        m++
      }
    }
    const take = Math.min(m, budget)
    if (m > budget) selectSmallest(order.subarray(0, m), keys, budget)
    if (sel.length < take) sel = new Int32Array(take * 2)
    let active = -1, minD = Infinity
    for (let p = 0; p < take; p++) {
      const q = m > budget ? order[p] : p
      sel[p] = cand[q]
      if (keys[q] < minD) { minD = keys[q]; active = cand[q] }
    }
    out.idx = sel; out.n = take; out.active = active
    return out
  }
}
