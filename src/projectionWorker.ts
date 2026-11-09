/**
 * Projection worker: runs the DruidJS dimensionality reduction off the main
 * thread. `.transform()` is one synchronous call that takes tens of seconds on
 * a few thousand 768-d vectors. Main-thread facade: `src/projection.ts`.
 */

import * as druid from '@saehrimnir/druidjs'
import type { ProjectionRequest, ProjectionResponse } from './types'

function project({ method, nNeighbors, flat, n, dim }: ProjectionRequest): Float64Array {
  const matrix = druid.Matrix.from(
    Array.from({ length: n }, (_, i) => Float64Array.from(flat.subarray(i * dim, (i + 1) * dim)))
  )
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

  const points = new Float64Array(n * 2)
  result.to2dArray().forEach((row, i) => {
    points[i * 2] = row[0]
    points[i * 2 + 1] = row[1]
  })
  return points
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
