/**
 * Projection worker: runs the DruidJS dimensionality reduction off the main
 * thread. `.transform()` is one synchronous call that takes tens of seconds on
 * a few thousand 768-d vectors. Main-thread facade: `src/projection.ts`.
 */

import * as druid from '@saehrimnir/druidjs'
import type { ProjectionRequest, ProjectionResponse } from './types'

/**
 * t-SNE (and the other DruidJS methods) hold n×n double matrices: ~800 MB per matrix at 10,000 photos (and 1,500 points already take ~25 s), and the
 * tab is killed. Above this many points only an evenly spaced sample is laid out, and the rest are placed
 * from it (placeByNeighbours). shortcut: a sample, not Barnes-Hut t-SNE; swap in a tree-based t-SNE if layouts of 50k+ matter.
 */
const MAX_LAYOUT_POINTS = 1500

function layout(method: ProjectionRequest['method'], nNeighbors: number, rows: Float32Array[]) {
  const n = rows.length
  const matrix = druid.Matrix.from(rows.map((r) => Float64Array.from(r)))
  let result: druid.Matrix

  switch (method) {
    case 'TSNE':
      result = new druid.TSNE(matrix, {
        d: 2,
        perplexity: Math.min(30, Math.floor(n / 3))
      }).transform()
      break
    case 'PCA':
      result = new druid.PCA(matrix, { d: 2 }).transform()
      break
    case 'ISOMAP':
      result = new druid.ISOMAP(matrix, {
        d: 2,
        neighbors: nNeighbors
      }).transform()
      break
    case 'LLE':
      result = new druid.LLE(matrix, {
        d: 2,
        neighbors: nNeighbors
      }).transform()
      break
    case 'MDS':
      result = new druid.MDS(matrix, { d: 2 }).transform()
      break
    case 'SAMMON':
      result = new druid.SAMMON(matrix, { d: 2 }).transform()
      break
    case 'TriMap':
      result = new druid.TriMap(matrix, { d: 2 }).transform()
      break
    case 'UMAP':
    default:
      result = new druid.UMAP(matrix, {
        d: 2,
        n_neighbors: nNeighbors,
        local_connectivity: 1
      }).transform()
      break
  }

  return result.to2dArray()
}

const K_PLACE = 5

/** Each point not in the sample goes to the distance-weighted mean of its nearest sampled points (cosine on unit vectors). */
function placeByNeighbours(flat: Float32Array, n: number, dim: number, picked: number[], pos: ArrayLike<number>[]): Float64Array {
  const points = new Float64Array(n * 2)
  const isPicked = new Map(picked.map((p, j) => [p, j]))
  const dot = (a: number, b: number) => {
    let d = 0
    for (let k = 0; k < dim; k++) d += flat[a * dim + k] * flat[b * dim + k]
    return d
  }
  for (let i = 0; i < n; i++) {
    const j = isPicked.get(i)
    if (j !== undefined) {
      points[i * 2] = pos[j][0]
      points[i * 2 + 1] = pos[j][1]
      continue
    }
    const best: [number, number][] = [] // [similarity, sample index], K_PLACE highest
    for (let j2 = 0; j2 < picked.length; j2++) {
      const sim = dot(i, picked[j2])
      if (best.length < K_PLACE || sim > best[best.length - 1][0]) {
        best.push([sim, j2])
        best.sort((a, b) => b[0] - a[0])
        if (best.length > K_PLACE) best.pop()
      }
    }
    let x = 0
    let y = 0
    let wsum = 0
    for (const [sim, j2] of best) {
      const w = Math.exp(10 * sim) // sharper than linear: the closest sample dominates
      x += w * pos[j2][0]
      y += w * pos[j2][1]
      wsum += w
    }
    points[i * 2] = x / wsum
    points[i * 2 + 1] = y / wsum
  }
  return points
}

function project({ method, nNeighbors, flat, n, dim }: ProjectionRequest): Float64Array {
  const row = (i: number) => flat.subarray(i * dim, (i + 1) * dim)
  const m = Math.min(n, MAX_LAYOUT_POINTS)
  const picked = Array.from({ length: m }, (_, j) => Math.floor((j * n) / m))
  const pos = layout(method, Math.min(nNeighbors, m - 1), picked.map(row))
  if (m === n) {
    const points = new Float64Array(n * 2)
    pos.forEach((r, i) => ((points[i * 2] = r[0]), (points[i * 2 + 1] = r[1])))
    return points
  }
  return placeByNeighbours(flat, n, dim, picked, pos)
}

function post(message: ProjectionResponse, transfer: Transferable[] = []): void {
  ;(self as unknown as Worker).postMessage(message, transfer)
}

self.addEventListener('message', (event: MessageEvent<ProjectionRequest>) => {
  const { id } = event.data
  try {
    const points = project(event.data)
    post({ type: 'projected', id, points }, [points.buffer])
  } catch (err) {
    const e = err instanceof Error ? err : new Error(String(err))
    post({ type: 'error', id, message: e.message, name: e.name })
  }
})
