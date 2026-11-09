/**
 * Main-thread facade over the projection worker (`src/projectionWorker.ts`).
 * Same shape as `src/vlm.ts`: one long-lived worker, requests correlated by a
 * numeric id (a progressive and a final projection can be in flight together).
 */

import type { ProjectionMethod, ProjectionRequest, ProjectionResponse } from './types'

interface Pending {
  resolve: (points: number[][]) => void
  reject: (err: Error) => void
}

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, Pending>()

/** Fail everything in flight and drop the worker, so the next call starts a fresh one. */
function crash(message: string): void {
  const err = new Error(message)
  for (const [, entry] of pending) entry.reject(err)
  pending.clear()
  try {
    worker?.terminate()
  } catch {
    /* already gone */
  }
  worker = null
}

function ensureWorker(): Worker {
  if (worker) return worker
  worker = new Worker(new URL('./projectionWorker.ts', import.meta.url), { type: 'module' })
  worker.addEventListener('message', (event: MessageEvent<ProjectionResponse>) => {
    const msg = event.data
    const entry = pending.get(msg.id)
    if (!entry) return
    pending.delete(msg.id)

    if (msg.type === 'error') {
      const err = new Error(msg.message)
      err.name = msg.name
      entry.reject(err)
    } else {
      const { points } = msg
      entry.resolve(Array.from({ length: points.length / 2 }, (_, i) => [points[i * 2], points[i * 2 + 1]]))
    }
  })
  worker.addEventListener('error', (event) => crash(event.message || 'Projection worker crashed'))
  worker.addEventListener('messageerror', () => crash('Projection worker sent an unreadable message'))
  return worker
}

/** Project `vectors` to 2-D in the worker. The vectors are copied, not transferred. */
export function projectInWorker(
  vectors: Float32Array[],
  method: ProjectionMethod,
  nNeighbors: number
): Promise<number[][]> {
  return new Promise((resolve, reject) => {
    const n = vectors.length
    const dim = vectors[0].length
    const flat = new Float32Array(n * dim)
    for (let i = 0; i < n; i++) {
      // Same error druid's Matrix.from raised when this ran on the main thread.
      if (vectors[i].length !== dim) throw new Error('various array lengths')
      flat.set(vectors[i], i * dim)
    }
    const req: ProjectionRequest = { id: nextId++, method, nNeighbors, flat, n, dim }
    const w = ensureWorker()
    pending.set(req.id, { resolve, reject })
    w.postMessage(req, [flat.buffer])
  })
}
