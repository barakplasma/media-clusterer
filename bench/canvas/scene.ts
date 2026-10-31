// Shared synthetic scene: N clustered photos, K reusable 64px thumbnails, a scripted pan+zoom path.
export const THUMB_WORLD = 48
export const VIEW_W = 1280
export const VIEW_H = 720
export const BUDGET = 400 // desktop default drawBudget in src/app.ts
export const K = 64
export const TH = 64
export const COLORS = ['#ef4444', '#f59e0b', '#22c55e', '#06b6d4', '#6366f1', '#d946ef', '#6b7280', '#facc15']

export interface Scene { n: number; xy: Float32Array; cluster: Uint8Array; thumbIdx: Uint8Array; side: number }

export function makeScene(n: number): Scene {
  let seed = 12345
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296)
  const side = Math.sqrt(n) * 60 // ~constant density: one photo per 60x60 world units
  const xy = new Float32Array(n * 2), cluster = new Uint8Array(n), thumbIdx = new Uint8Array(n)
  const centers = Array.from({ length: 24 }, () => [rnd() * side, rnd() * side])
  for (let i = 0; i < n; i++) {
    const c = i % 24
    const a = rnd() * 6.283, r = Math.sqrt(rnd()) * side * 0.12
    xy[2 * i] = centers[c][0] + Math.cos(a) * r
    xy[2 * i + 1] = centers[c][1] + Math.sin(a) * r
    cluster[i] = c % COLORS.length
    thumbIdx[i] = i % K
  }
  return { n, xy, cluster, thumbIdx, side }
}

export const FRAMES = 480 // two full camera cycles, so every renderer sees the identical workload
const PERIOD = 240
/** Deterministic, periodic camera: zooms from 1.5 (close) out to 0.08 (thousands visible) and back, while panning. */
export function cameraAt(f: number, side: number) {
  const t = (2 * Math.PI * f) / PERIOD
  const z = (1 - Math.cos(t)) / 2 // 0 = close, 1 = far out
  const scale = Math.exp(Math.log(1.5) + z * (Math.log(0.08) - Math.log(1.5)))
  return { x: side / 2 + Math.sin(t) * side * 0.2, y: side / 2 + Math.cos(t * 0.5) * side * 0.2, scale }
}

/** 8x8 atlas of K thumbnails; every renderer draws from this same pixel data. */
export function makeAtlas(): OffscreenCanvas {
  const c = new OffscreenCanvas(TH * 8, TH * 8)
  const g = c.getContext('2d')!
  for (let k = 0; k < K; k++) {
    const x = (k % 8) * TH, y = Math.floor(k / 8) * TH
    const gr = g.createLinearGradient(x, y, x + TH, y + TH)
    gr.addColorStop(0, `hsl(${k * 5.6},70%,55%)`); gr.addColorStop(1, `hsl(${k * 5.6 + 60},70%,30%)`)
    g.fillStyle = gr; g.fillRect(x, y, TH, TH)
    g.fillStyle = '#fff'; g.font = '20px sans-serif'; g.fillText(String(k), x + 8, y + 36)
  }
  return c
}

export interface Stats { renderer: string; n: number; frames: number; p50: number; p95: number; max: number; drawn: number }
export const pct = (a: number[], p: number) => a.slice().sort((x, y) => x - y)[Math.floor((a.length - 1) * p)]
