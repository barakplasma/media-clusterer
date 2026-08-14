/**
 * Main-thread facade over the SmolVLM2 worker (`src/vlmWorker.ts`).
 *
 * Callers see promises; the request/response correlation, worker lifecycle and
 * bitmap transfer all live here. The message protocol is deliberately generic
 * (numeric id, typed request union) so the projection worker that
 * IMPROVEMENT_PLAN.md P1-1 wants can reuse the same shape rather than inventing
 * a second one.
 */

import type { ModelVariant, VlmDevice, VlmRequest, VlmResponse, VlmTier } from './types'
import { getVlmTier } from './vlmTiers'

export type VlmProgress = (pct: number, file: string) => void

interface Pending {
  resolve: (value: never) => void
  reject: (err: Error) => void
}

let worker: Worker | null = null
let nextId = 1
let device: VlmDevice = 'wasm'
const pending = new Map<number, Pending>()
let onProgress: VlmProgress | null = null

// Per-file byte counters, so a multi-file download reports one combined
// percentage rather than flicking back to 0% at each file boundary.
const loadedBytes = new Map<string, number>()
const totalBytes = new Map<string, number>()

function ensureWorker(): Worker {
  if (worker) return worker
  worker = new Worker(new URL('./vlmWorker.ts', import.meta.url), { type: 'module' })
  worker.addEventListener('message', (event: MessageEvent<VlmResponse>) => {
    const msg = event.data

    if (msg.type === 'progress') {
      if (msg.total > 0) totalBytes.set(msg.file, msg.total)
      loadedBytes.set(msg.file, msg.loaded)
      let done = 0
      let all = 0
      for (const v of loadedBytes.values()) done += v
      for (const v of totalBytes.values()) all += v
      onProgress?.(all > 0 ? Math.min(99, (done / all) * 100) : 0, msg.file)
      return
    }

    const entry = pending.get(msg.id)
    if (!entry) return
    pending.delete(msg.id)

    if (msg.type === 'error') {
      const err = new Error(msg.message)
      err.name = msg.name
      entry.reject(err)
    } else if (msg.type === 'loaded') {
      device = msg.device
      entry.resolve(msg as never)
    } else {
      entry.resolve(msg as never)
    }
  })
  worker.addEventListener('error', (event) => {
    // A worker-level error (module failed to parse, import blocked) never
    // resolves an individual request, so fail everything in flight rather than
    // hanging the embed loop forever.
    const err = new Error(event.message || 'VLM worker crashed')
    for (const [, entry] of pending) entry.reject(err)
    pending.clear()
  })
  return worker
}

function send<T extends VlmResponse>(req: VlmRequest, transfer: Transferable[] = []): Promise<T> {
  const w = ensureWorker()
  return new Promise<T>((resolve, reject) => {
    pending.set(req.id, { resolve: resolve as (v: never) => void, reject })
    w.postMessage(req, transfer)
  })
}

/** Backend in use after the webgpu→wasm fallback. Meaningful only after load. */
export function vlmDevice(): VlmDevice {
  return device
}

/**
 * Load a tier's vision encoder. Idempotent for the same tier.
 * Rejects if `variant` is not a SmolVLM2 tier.
 */
export async function loadVlm(variant: ModelVariant, progress?: VlmProgress): Promise<VlmTier> {
  const tier = getVlmTier(variant)
  if (!tier) throw new Error(`${variant} is not a SmolVLM2 tier`)

  onProgress = progress ?? null
  loadedBytes.clear()
  totalBytes.clear()
  try {
    await send({ type: 'load', id: nextId++, tier })
    return tier
  } finally {
    onProgress = null
  }
}

/**
 * Embed bitmaps to pooled 768-d vectors.
 *
 * The bitmaps are **transferred**, so they are neutered in the caller and must
 * not be touched afterwards. The worker closes them. Pass clones for anything
 * still needed on the main thread, exactly as the chrome-ai path does for
 * `state.thumbnails`.
 */
export async function embedWithVlm(images: ImageBitmap[]): Promise<Float32Array[]> {
  if (images.length === 0) return []
  const res = await send<Extract<VlmResponse, { type: 'embedded' }>>(
    { type: 'embed', id: nextId++, images },
    images
  )
  return res.vectors
}

/** Tear the worker down and free the model. Safe to call when never started. */
export function disposeVlm(): void {
  if (!worker) return
  try {
    worker.postMessage({ type: 'dispose', id: nextId++ } satisfies VlmRequest)
  } catch {
    /* worker may already be gone */
  }
  worker.terminate()
  worker = null
  pending.clear()
  loadedBytes.clear()
  totalBytes.clear()
}
