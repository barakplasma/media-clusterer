// WebGL via PixiJS v8: one atlas texture, two sprites per photo (tinted white quad = border + thumbnail).
// `budget` = BUDGET for apples-to-apples, Infinity for "what WebGL allows" (capped by CAP sprites).
import { Application, Container, Rectangle, Sprite, Texture } from 'pixi.js'
import { createCuller } from './cull'
import { Scene, THUMB_WORLD, VIEW_W, VIEW_H, BUDGET, COLORS, makeAtlas, K, TH } from './scene'

const CAP = 8000

export async function createPixi(canvas: HTMLCanvasElement, scene: Scene, all: boolean) {
  const app = new Application()
  await app.init({ canvas, width: VIEW_W, height: VIEW_H, antialias: false, resolution: 1, autoStart: false, preference: 'webgl', backgroundAlpha: 0 })
  const atlasTex = Texture.from(makeAtlas() as unknown as HTMLCanvasElement)
  const frames = Array.from({ length: K }, (_, k) => new Texture({ source: atlasTex.source, frame: new Rectangle((k % 8) * TH, Math.floor(k / 8) * TH, TH, TH) }))
  const layer = new Container()
  app.stage.addChild(layer)
  const borders: Sprite[] = [], pics: Sprite[] = []
  const query = createCuller(scene)
  const xy = scene.xy
  const budget = all ? CAP : BUDGET
  const gl = (app.renderer as any).gl as WebGL2RenderingContext
  const px = new Uint8Array(4)
  return {
    frame(cam: { x: number; y: number; scale: number }) {
      const s = cam.scale, ds = THUMB_WORLD * s, half = ds / 2, b = Math.max(1.5, 2 * Math.min(s, 1))
      const r = query(cam, budget)
      while (pics.length < r.n) {
        const bd = new Sprite(Texture.WHITE), pc = new Sprite(); borders.push(bd); pics.push(pc)
      }
      while (layer.children.length < r.n * 2) { const k = layer.children.length / 2; layer.addChild(borders[k], pics[k]) }
      while (layer.children.length > r.n * 2) layer.removeChildAt(layer.children.length - 1)
      for (let p = 0; p < r.n; p++) {
        const i = r.idx[p]
        const sx = (xy[2 * i] - cam.x) * s + VIEW_W / 2, sy = (xy[2 * i + 1] - cam.y) * s + VIEW_H / 2
        const bd = borders[p], pc = pics[p]
        bd.tint = i === r.active ? 0xffffff : parseInt(COLORS[scene.cluster[i]].slice(1), 16)
        bd.x = sx - half - b; bd.y = sy - half - b; bd.width = ds + 2 * b; bd.height = ds + 2 * b
        pc.texture = frames[scene.thumbIdx[i]]; pc.alpha = 0.9
        pc.x = sx - half; pc.y = sy - half; pc.width = ds; pc.height = ds
      }
      app.renderer.render(app.stage)
      return r.n
    },
    flush() { gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px) },
  }
}
