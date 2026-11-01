import { test, expect } from '@playwright/test'
import { cullAndPrioritize, type ViewCamera } from '../src/compute'
import type { Point } from '../src/types'

// Pure function check, no browser: the grid-indexed cull must equal the linear scan, order included.
function scene(n: number, spread: number, seed: number): Point[] {
  let s = seed
  const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296
  return Array.from({ length: n }, (): Point => [(rnd() - 0.3) * spread, (rnd() - 0.6) * spread])
}

for (const [n, spread] of [[3000, 4000], [20000, 20000], [500, 1e6], [1, 10]] as const) {
  test(`indexed cull matches linear scan (${n} pts, spread ${spread})`, () => {
    const pts = scene(n, spread, n)
    const rank = new Int32Array(n).fill(n)
    for (let i = 0; i < Math.min(60, n); i++) rank[(i * 7) % n] = i
    for (const scale of [0.05, 0.3, 1, 3]) {
      for (const [cx, cy] of [[0, 0], [spread * 0.2, -spread * 0.1], [-spread, spread]]) {
        const cam: ViewCamera = { x: cx, y: cy, scale }
        for (const budget of [400, 5000]) for (const r of [null, rank]) {
          const a = cullAndPrioritize(pts, cam, 1280, 720, 24 * scale, r, budget, false)
          const b = cullAndPrioritize(pts, cam, 1280, 720, 24 * scale, r, budget, true)
          // Over budget the winners' order depends on quickselect ties, so compare as sets; under budget, exactly.
          if (a.length < budget && !r) expect(b).toEqual(a)
          else expect(new Set(b)).toEqual(new Set(a))
        }
      }
    }
  })
}

test('indexed cull falls back to the linear scan for non-finite points', () => {
  const pts: Point[] = [[0, 0], [NaN, 5], [10, 10]]
  const cam: ViewCamera = { x: 0, y: 0, scale: 1 }
  expect(cullAndPrioritize(pts, cam, 100, 100, 24, null, 400, true)).toEqual(cullAndPrioritize(pts, cam, 100, 100, 24, null, 400, false))
})
