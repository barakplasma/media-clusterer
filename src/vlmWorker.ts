/**
 * SmolVLM2 inference worker — the repo's first Web Worker.
 *
 * Everything here runs off the main thread so the canvas stays interactive
 * while a folder embeds. That is the whole point: on the 8 GB Chromebook this
 * work targets, the previous captioner did not merely run slowly, it took the
 * tab down with it.
 *
 * Milestone M1/M2 (ADR-0001): tier A only. The vision encoder runs, its output
 * is pooled to one vector per image, and the decoder is never downloaded. Tiers
 * B/C add caption generation on the same forward pass in M4.
 */

import { AutoModel, RawImage, env } from '@huggingface/transformers'
import { assertVisionDim, poolVectors, poolVision } from './vlmTiers'
import type { VlmDevice, VlmRequest, VlmResponse, VlmTier } from './types'

// A worker gets its own module instance, so none of the `env` configuration
// done in app.ts applies here. Every setting the main thread relies on has to
// be repeated, or the worker silently behaves differently from the rest of the
// app — no model cache, wrong host, and (below) no WASM at all.
env.allowLocalModels = false
env.allowRemoteModels = true
// This is what makes the ~55 MB download a one-time cost: Transformers.js
// stores fetched weights in the Cache API and serves them from there on every
// later load. It defaults to true in a browser; set explicitly because the
// whole point of tier A is that users pay for it once.
env.useBrowserCache = true
// Mirrors app.ts. Note this is the *file system* cache directory, not the
// browser one — inert here, kept only so the two module instances are
// configured identically and neither looks like the odd one out.
env.cacheDir = 'models'

// vite.config.ts strips `ort-wasm*` from dist (Cloudflare's 25 MiB per-file
// limit), so onnxruntime-web must fetch its binaries from the CDN instead —
// exactly what src/sapiens2.ts does for the same reason.
//
// This is not only the WASM fallback's concern: onnxruntime-web's WebGPU
// backend is JSEP-on-WASM and loads the same artifacts, so without this *both*
// paths 404 at runtime in a production build while working fine under
// `vite dev`. Unguarded on purpose: `env.backends.onnx.wasm` is a read-only
// accessor that always exists, so if that ever stops being true a TypeError at
// startup beats a silent skip that only surfaces as a broken deploy.
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'
const ortWasm = env.backends.onnx.wasm
if (!ortWasm) {
  throw new Error('onnxruntime-web WASM env missing; cannot point it at the CDN build')
}
ortWasm.wasmPaths = ORT_CDN
ortWasm.numThreads = navigator.hardwareConcurrency || 4

interface VisionOutput {
  data: Float32Array
  dims: number[]
}

// Loaded lazily on the first 'load' request and reused thereafter. `AutoModel`
// (rather than AutoModelForVision2Seq) loads the vision tower alone, which is
// what keeps tier A to a ~55 MB download instead of ~189 MB.
let model: { dispose?: () => Promise<void> } | null = null
let runVision: ((pixelValues: unknown) => Promise<Record<string, VisionOutput>>) | null = null
let processor: { (images: RawImage[]): Promise<Record<string, unknown>> } | null = null
let loadedTier: VlmTier | null = null
let device: VlmDevice = 'wasm'

function post(message: VlmResponse, transfer: Transferable[] = []): void {
  ;(self as unknown as Worker).postMessage(message, transfer)
}

function toError(err: unknown): { message: string; name: string } {
  const e = err as Error
  return { message: e?.message ?? String(err), name: e?.name ?? 'Error' }
}

/**
 * Load the vision encoder, preferring WebGPU and falling back to WASM.
 *
 * Mirrors the fallback already used for the text model (`src/app.ts`) and
 * Sapiens2 (`src/sapiens2.ts`): try the fast path, and on any failure retry on
 * the slow one rather than surfacing an error, because a slow embed is a far
 * better outcome than none on the hardware this targets.
 */
async function load(id: number, tier: VlmTier, remoteHost: string): Promise<void> {
  // The custom-host setting (corporate proxy / Artifactory mirror) lives in
  // main-thread settings, so it has to be handed over on every load.
  env.remoteHost = remoteHost || 'https://huggingface.co'
  if (loadedTier?.id === tier.id && model) {
    post({ type: 'loaded', id, device, visionDim: tier.visionDim })
    return
  }
  await dispose()

  const { AutoProcessor } = await import('@huggingface/transformers')

  const progress_callback = (e: { status?: string; file?: string; loaded?: number; total?: number }) => {
    if (e.status !== 'progress') return
    post({
      type: 'progress',
      id,
      file: e.file ?? '',
      loaded: e.loaded ?? 0,
      total: e.total ?? 0
    })
  }

  processor = (await AutoProcessor.from_pretrained(tier.repo, {
    progress_callback
  })) as unknown as typeof processor

  const opts = { dtype: tier.dtype, progress_callback }
  // `model_file_name` pins the load to the vision tower; without it the loader
  // would also fetch embed_tokens and the decoder, which tier A never runs.
  const visionOpts = { ...opts, model_file_name: 'vision_encoder' }

  try {
    model = (await AutoModel.from_pretrained(tier.repo, {
      ...visionOpts,
      device: 'webgpu'
    })) as unknown as typeof model
    device = 'webgpu'
  } catch {
    model = (await AutoModel.from_pretrained(tier.repo, {
      ...visionOpts,
      device: 'wasm'
    })) as unknown as typeof model
    device = 'wasm'
  }

  runVision = model as unknown as typeof runVision
  loadedTier = tier
  post({ type: 'loaded', id, device, visionDim: tier.visionDim })
}

/**
 * Embed groups of frames to one pooled vector per group.
 *
 * A group is all the frames sampled from one item: exactly one for a still
 * image, `tier.framesPerVideo` for a video. Per-frame vectors are combined by
 * poolVectors, so a video is represented by the centroid of its sampled
 * content rather than by whichever single frame happened to be grabbed.
 *
 * Frames are processed one at a time on purpose. Batching would help on a
 * discrete GPU, but the target device has neither the VRAM headroom nor a
 * discrete GPU, and a batch that OOMs costs the whole tab.
 */
async function embed(id: number, groups: ImageBitmap[][]): Promise<void> {
  if (!runVision || !processor || !loadedTier) {
    throw new Error('VLM worker: embed before load')
  }
  const tier = loadedTier
  const vectors: Float32Array[] = []

  for (const group of groups) {
    const frameVectors: Float32Array[] = []
    for (const bitmap of group) {
      // RawImage.fromBlob would re-decode; the bitmap is already decoded, so
      // go through a canvas to hand the processor raw pixels directly.
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('VLM worker: no 2d context for frame conversion')
      ctx.drawImage(bitmap, 0, 0)
      const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height)
      const raw = new RawImage(
        new Uint8ClampedArray(imageData.data),
        imageData.width,
        imageData.height,
        4
      )

      // do_image_splitting: false is load-bearing — the default tiles each
      // frame against size.longest_edge 2048, which is the easiest way to blow
      // the memory budget on the device this exists to support.
      const inputs = (await (processor as unknown as CallableFunction)(raw, {
        do_image_splitting: false
      })) as { pixel_values: unknown }

      const out = await runVision({ pixel_values: inputs.pixel_values })
      const features = out.last_hidden_state ?? out.image_features ?? Object.values(out)[0]
      if (!features?.dims) {
        throw new Error('VLM worker: vision encoder returned no recognisable tensor')
      }

      assertVisionDim(tier, features.dims[features.dims.length - 1])
      frameVectors.push(poolVision(features.data, features.dims))

      // AGENT.md: bitmaps are closed as soon as they are done with. The worker
      // owns these because they were transferred to it.
      bitmap.close()
    }

    if (frameVectors.length === 0) {
      throw new Error('VLM worker: every frame in a group failed to embed')
    }
    vectors.push(poolVectors(frameVectors))
  }

  post(
    { type: 'embedded', id, vectors },
    vectors.map((v) => v.buffer)
  )
}

async function dispose(): Promise<void> {
  try {
    await model?.dispose?.()
  } catch {
    /* disposing a half-loaded model is best-effort */
  }
  model = null
  runVision = null
  processor = null
  loadedTier = null
}

self.addEventListener('message', (event: MessageEvent<VlmRequest>) => {
  const req = event.data
  void (async () => {
    try {
      if (req.type === 'load') await load(req.id, req.tier, req.remoteHost)
      else if (req.type === 'embed') await embed(req.id, req.groups)
      else if (req.type === 'dispose') await dispose()
    } catch (err) {
      // Free any bitmaps the failed request still owns, or the tab leaks one
      // GPU-backed image per failure.
      if (req.type === 'embed') for (const g of req.groups) for (const b of g) b.close()
      post({ type: 'error', id: req.id, ...toError(err) })
    }
  })()
})
