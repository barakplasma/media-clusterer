/**
 * EmbeddingGemma 2 on WebNN, for GPUs too slow for WebGPU (Pixel 10's PowerVR:
 * ~330 s per photo there, ~3 s per photo on WebNN). Runs inside the VLM worker
 * and returns a callable with the same contract as the transformers.js model
 * (inputs from the processor in, `sentence_embedding` out), so the worker's
 * embedding code stays the same. Measured and explained in npu-test.html.
 *
 * Differences from the WebGPU path, each forced by WebNN:
 * - fp16 files: the q4 ones need MatMulNBits / GatherBlockQuantized.
 * - Raw onnxruntime-web sessions, because the text graph is our own
 *   (scripts/webnn_split_embed.py): Chrome refuses to build its 268 MB
 *   token-embedding table, so the lookup happens here in JS.
 * - Fixed shapes: batch 1, the processor's patch count, the tier's sequence
 *   length. Image/video token counts stay free (they vary with aspect ratio
 *   and are 0 for a query); only the small merge step consuming them falls
 *   back to CPU. Video frames therefore go through the vision encoder one by one.
 */

import * as ort from 'onnxruntime-web'
import type { VlmTier, WebnnDevice } from './types'

const GRAPH_URL = `${import.meta.env.BASE_URL}models/embeddinggemma2-webnn/model_fp16_noembed.onnx`
// model.embed_tokens.weight inside model_fp16.onnx_data (printed by scripts/webnn_split_embed.py).
const EMBED_OFFSET = 1024
const EMBED_ROWS = 262144
const EMBED_WIDTH = 512
const FEATURE_WIDTH = 512 // image_features / video_features width

ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`

type Progress = (file: string, loaded: number, total: number) => void
interface ProcTensor {
  type: ort.Tensor.Type
  data: ort.Tensor.DataType
  dims: number[]
}
export type WebnnModel = ((
  inputs: Record<string, unknown>
) => Promise<{ sentence_embedding: { data: Float32Array; dims: number[] } }>) & {
  dispose: () => Promise<void>
}

/** Download once into the Cache API, reporting progress; later loads read the cache. */
async function fetchFile(url: string, file: string, progress: Progress): Promise<Uint8Array> {
  const cache = await caches.open('eg2-webnn').catch(() => null)
  const hit = await cache?.match(url)
  const res = hit ?? (await fetch(url))
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`)
  // The site's SPA fallback answers a missing file with index.html.
  if (res.headers.get('content-type')?.includes('text/html'))
    throw new Error(`${file} is not deployed`)
  const putting = hit ? null : cache?.put(url, res.clone()).catch(() => {})
  const total = Number(res.headers.get('content-length')) || 0
  const chunks: Uint8Array[] = []
  let loaded = 0
  const reader = res.body!.getReader()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    progress(file, loaded, total)
  }
  await putting
  if (chunks.length === 1) return chunks[0]
  const out = new Uint8Array(loaded)
  let at = 0
  for (const c of chunks) out.set(c, (at += c.byteLength) - c.byteLength)
  return out
}

const toOrt = (x: unknown) => {
  const t = x as ProcTensor // transformers.js bundles its own ORT; run() checks instanceof
  return new ort.Tensor(t.type, t.data, t.dims)
}

export async function loadWebnnModel(
  tier: VlmTier,
  remoteHost: string,
  patches: number,
  progress: Progress
): Promise<{ model: WebnnModel; device: WebnnDevice }> {
  const base = `${remoteHost}/${tier.repo}/resolve/main/onnx/`
  const devices: WebnnDevice[] = tier.webnnDevices?.length ? tier.webnnDevices : ['npu', 'gpu', 'cpu']
  const options = {
    freeDimensionOverrides: {
      s11: 1,
      s35: patches,
      batch_size: 1,
      sequence_length: tier.fixedSeqLen ?? 296,
      num_audio_tokens: 0
    }
  }
  // A failed WebNN graph build surfaces as an unhandled rejection and the
  // create() never settles; turn it into that attempt's error.
  let rejectBuild: (e: unknown) => void = () => {}
  const onRejection = (e: PromiseRejectionEvent) => rejectBuild(e.reason)
  self.addEventListener('unhandledrejection', onRejection)
  let used = 0 // index into devices of the least native device any part needed
  /** Build on the most native WebNN device that takes the graph: NPU, then GPU, then CPU. */
  const create = async (graph: Uint8Array, file: string, data: Uint8Array) => {
    const errors: string[] = []
    for (let i = 0; i < devices.length; i++) {
      const buildFailed = new Promise<never>((_, reject) => (rejectBuild = reject))
      try {
        const session = await Promise.race([
          ort.InferenceSession.create(graph, {
            ...options,
            executionProviders: [{ name: 'webnn', deviceType: devices[i] }],
            externalData: [{ path: file, data }]
          }),
          buildFailed
        ])
        used = Math.max(used, i)
        return session
      } catch (e) {
        errors.push(`${devices[i]}: ${(e as Error)?.message ?? e}`)
      }
    }
    throw new Error(`WebNN could not build ${file} on any device (${errors.join('; ')})`)
  }

  let vision: ort.InferenceSession
  let text: ort.InferenceSession
  let table: Uint16Array
  try {
    vision = await create(
      await fetchFile(`${base}vision_encoder_fp16.onnx`, 'vision_encoder_fp16.onnx', progress),
      'vision_encoder_fp16.onnx_data',
      await fetchFile(
        `${base}vision_encoder_fp16.onnx_data`,
        'vision_encoder_fp16.onnx_data',
        progress
      )
    )
    const graph = await fetchFile(GRAPH_URL, 'model_fp16_noembed.onnx', progress)
    const data = await fetchFile(`${base}model_fp16.onnx_data`, 'model_fp16.onnx_data', progress)
    // A copy, so the rest of the 543 MB buffer can be freed once ORT has it.
    table = new Uint16Array(
      data.slice(EMBED_OFFSET, EMBED_OFFSET + EMBED_ROWS * EMBED_WIDTH * 2).buffer
    )
    text = await create(graph, 'model_fp16.onnx_data', data)
  } finally {
    self.removeEventListener('unhandledrejection', onRejection)
  }

  const empty = new ort.Tensor('float32', new Float32Array(0), [0, FEATURE_WIDTH])

  /** Vision features for N frames, one at a time (the session is pinned to batch 1). */
  async function features(pixels: unknown, positions: unknown): Promise<ort.Tensor> {
    const pv = pixels as ProcTensor
    const pos = positions as ProcTensor
    const [n, p, w] = pv.dims
    const parts: Float32Array[] = []
    for (let i = 0; i < n; i++) {
      const out = await vision.run({
        pixel_values: new ort.Tensor(
          'float32',
          (pv.data as Float32Array).subarray(i * p * w, (i + 1) * p * w),
          [1, p, w]
        ),
        pixel_position_ids: new ort.Tensor(
          'int64',
          (pos.data as BigInt64Array).subarray(i * p * 2, (i + 1) * p * 2),
          [1, p, 2]
        )
      })
      parts.push(out.image_features.data as Float32Array)
    }
    if (n === 1)
      return new ort.Tensor('float32', parts[0], [parts[0].length / FEATURE_WIDTH, FEATURE_WIDTH])
    const all = new Float32Array(parts.reduce((a, x) => a + x.length, 0))
    let at = 0
    for (const x of parts) all.set(x, (at += x.length) - x.length)
    return new ort.Tensor('float32', all, [all.length / FEATURE_WIDTH, FEATURE_WIDTH])
  }

  const model = (async (inputs: Record<string, unknown>) => {
    const ids = toOrt(inputs.input_ids)
    // The lookup the graph no longer does: one fp16 row per token.
    const idData = ids.data as BigInt64Array
    const rows = new Uint16Array(idData.length * EMBED_WIDTH)
    idData.forEach((id, i) =>
      rows.set(
        table.subarray(Number(id) * EMBED_WIDTH, (Number(id) + 1) * EMBED_WIDTH),
        i * EMBED_WIDTH
      )
    )
    const out = await text.run({
      inputs_embeds: new ort.Tensor('float16', rows, [...ids.dims, EMBED_WIDTH]),
      input_ids: ids,
      attention_mask: toOrt(inputs.attention_mask),
      image_features: inputs.pixel_values
        ? await features(inputs.pixel_values, inputs.image_position_ids)
        : empty,
      video_features: inputs.pixel_values_videos
        ? await features(inputs.pixel_values_videos, inputs.video_position_ids)
        : empty,
      audio_features: empty
    })
    const emb = out.sentence_embedding
    return { sentence_embedding: { data: emb.data as Float32Array, dims: [...emb.dims] } }
  }) as WebnnModel
  model.dispose = async () => {
    await Promise.allSettled([vision.release(), text.release()])
  }
  return { model, device: devices[used] }
}
