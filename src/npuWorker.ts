/**
 * The WebAssembly (CPU) half of a hybrid benchmark run (npu-test.html?hybrid=webgpu,cpu): the same two fp16
 * sessions the page builds for a device, on a worker. Run on the page's own thread, a WASM session would hold
 * up the page's GPU session, which needs that thread to dispatch its kernels, and the two would not overlap.
 * No WebNN here on purpose: it is not on every device and has crashed a Chromebook.
 *
 * Messages in: load {dtype, overrides, verbose}, embed {id, inp}. Out: phase, loaded, vec {id, v}, error.
 */

import * as ort from 'onnxruntime-web'

const REPO = 'onnx-community/embeddinggemma-2-ONNX'

interface TensorLike {
  type: ort.Tensor.Type
  data: ort.Tensor.DataType
  dims: number[]
}
type Msg =
  | { type: 'load'; dtype: string; overrides: Record<string, number>; verbose: boolean }
  | { type: 'embed'; id: number; inp: Record<string, TensorLike> }

const sessions: Record<string, ort.InferenceSession> = {}
let empty: ort.Tensor | null = null

const post = (m: unknown) => (self as unknown as Worker).postMessage(m)

/** Fetch a model file once; the page's runs and this worker share the same Cache API entry. */
async function cachedFetch(file: string): Promise<Uint8Array> {
  const url = `https://huggingface.co/${REPO}/resolve/main/onnx/${file}`
  const cache = await caches.open('npu-test').catch(() => null)
  let res = await cache?.match(url)
  if (!res) {
    res = await fetch(url)
    if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`)
    await cache?.put(url, res.clone()).catch(() => {})
  }
  return new Uint8Array(await res.arrayBuffer())
}

self.addEventListener('message', (event: MessageEvent<Msg>) => {
  const m = event.data
  void (async () => {
    try {
      if (m.type === 'load') {
        ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`
        // One thread unless the page is cross-origin isolated (GitHub Pages is not).
        ort.env.wasm.numThreads = self.crossOriginIsolated ? navigator.hardwareConcurrency || 4 : 1
        ort.env.logLevel = m.verbose ? 'verbose' : 'warning'
        const suffix = m.dtype === 'fp32' ? '' : `_${m.dtype}`
        for (const name of ['vision_encoder', 'model'] as const) {
          post({ type: 'phase', phase: `download ${name}` })
          const graph = await cachedFetch(`${name}${suffix}.onnx`)
          const data = await cachedFetch(`${name}${suffix}.onnx_data`)
          post({ type: 'phase', phase: `compile ${name}` })
          sessions[name] = await ort.InferenceSession.create(graph, {
            executionProviders: ['wasm'],
            freeDimensionOverrides: m.overrides,
            externalData: [{ path: `${name}${suffix}.onnx_data`, data }],
            logSeverityLevel: m.verbose ? 0 : 2
          })
        }
        empty = new ort.Tensor('float32', new Float32Array(0), [0, 512])
        post({ type: 'loaded' })
      } else if (m.type === 'embed') {
        const t = (k: string) => new ort.Tensor(m.inp[k].type, m.inp[k].data, m.inp[k].dims)
        const image = (
          await sessions.vision_encoder.run({
            pixel_values: t('pixel_values'),
            pixel_position_ids: t('image_position_ids')
          })
        ).image_features
        const out = await sessions.model.run({
          input_ids: t('input_ids'),
          attention_mask: t('attention_mask'),
          image_features: image,
          video_features: empty!,
          audio_features: empty!
        })
        const v = Float32Array.from(out.sentence_embedding.data as Float32Array)
        post({ type: 'vec', id: m.id, v })
      }
    } catch (err) {
      post({ type: 'error', id: m.type === 'embed' ? m.id : undefined, message: (err as Error)?.message ?? String(err) })
    }
  })()
})
