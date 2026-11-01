// Optimized Canvas2D: flat points, grid-indexed culling, reused buffers, no O(n) center scan
// (highlight = closest of the drawn candidates). Runs on the main thread or inside a worker.
import { createCuller } from './cull'
import { Scene, THUMB_WORLD, VIEW_W, VIEW_H, BUDGET, COLORS, makeAtlas, K, TH } from './scene'

export async function createOpt2d(canvas: OffscreenCanvas | HTMLCanvasElement, scene: Scene, budget = BUDGET) {
  const ctx = canvas.getContext('2d')! as CanvasRenderingContext2D
  const atlas = makeAtlas()
  const thumbs = await Promise.all(Array.from({ length: K }, (_, k) => createImageBitmap(atlas, (k % 8) * TH, Math.floor(k / 8) * TH, TH, TH)))
  const query = createCuller(scene)
  const xy = scene.xy
  return {
    frame(cam: { x: number; y: number; scale: number }) {
      ctx.clearRect(0, 0, VIEW_W, VIEW_H)
      const s = cam.scale, ds = THUMB_WORLD * s, half = ds / 2
      const r = query(cam, budget)
      for (let p = 0; p < r.n; p++) {
        const i = r.idx[p]
        const sx = (xy[2 * i] - cam.x) * s + VIEW_W / 2, sy = (xy[2 * i + 1] - cam.y) * s + VIEW_H / 2
        ctx.globalAlpha = 0.9
        ctx.drawImage(thumbs[scene.thumbIdx[i]], sx - half, sy - half, ds, ds)
        ctx.globalAlpha = 1
        ctx.strokeStyle = i === r.active ? '#fff' : COLORS[scene.cluster[i]]
        ctx.lineWidth = Math.max(1.5, 2 * Math.min(s, 1))
        ctx.strokeRect(sx - half, sy - half, ds, ds)
      }
      return r.n
    },
    flush() { ctx.getImageData(0, 0, 1, 1) },
  }
}
