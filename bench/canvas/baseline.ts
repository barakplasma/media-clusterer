// Faithful replica of the hot path of render() in src/app.ts: per-frame rank alloc, O(n) center scan,
// O(n) cull + quickselect (the app's own cullAndPrioritize), thumbnail drawImage + strokeRect.
import { cullAndPrioritize } from '../../src/compute'
import { Scene, THUMB_WORLD, VIEW_W, VIEW_H, BUDGET, COLORS, makeAtlas, K, TH } from './scene'

export async function createBaseline(canvas: OffscreenCanvas | HTMLCanvasElement, scene: Scene) {
  const ctx = canvas.getContext('2d')! as CanvasRenderingContext2D
  const atlas = makeAtlas()
  const thumbs = await Promise.all(Array.from({ length: K }, (_, k) => createImageBitmap(atlas, (k % 8) * TH, Math.floor(k / 8) * TH, TH, TH)))
  const pts: [number, number][] = Array.from({ length: scene.n }, (_, i) => [scene.xy[2 * i], scene.xy[2 * i + 1]])
  return {
    frame(cam: { x: number; y: number; scale: number }) {
      ctx.clearRect(0, 0, VIEW_W, VIEW_H)
      new Int32Array(pts.length).fill(pts.length) // app allocates this every frame even with no search
      const s = cam.scale, half = THUMB_WORLD * s / 2, ds = THUMB_WORLD * s
      let minD = Infinity, active = -1
      for (let i = 0; i < pts.length; i++) {
        const dx = pts[i][0] - cam.x, dy = pts[i][1] - cam.y, d = dx * dx + dy * dy
        if (d < minD) { minD = d; active = i }
      }
      const vis = cullAndPrioritize(pts, cam, VIEW_W, VIEW_H, half, null, BUDGET)
      for (const i of vis) {
        const sx = (pts[i][0] - cam.x) * s + VIEW_W / 2, sy = (pts[i][1] - cam.y) * s + VIEW_H / 2
        ctx.globalAlpha = 0.9
        ctx.drawImage(thumbs[scene.thumbIdx[i]], sx - ds / 2, sy - ds / 2, ds, ds)
        ctx.globalAlpha = 1
        ctx.strokeStyle = i === active ? '#fff' : COLORS[scene.cluster[i]]
        ctx.lineWidth = Math.max(1.5, 2 * Math.min(s, 1))
        ctx.strokeRect(sx - ds / 2, sy - ds / 2, ds, ds)
      }
      return vis.length
    },
    flush() { ctx.getImageData(0, 0, 1, 1) }, // force raster so frame time includes the drawing
  }
}
