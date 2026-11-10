/**
 * Main-thread side of the WASM worker (src/npuWorker.ts), for the hybrid benchmark run: a runner that embeds
 * picture-A-style processor outputs on a worker, so it runs beside the page's own device instead of holding it up.
 */

export type Inp = Record<string, { dims: number[] }>

/** Load the two fp16 sessions on a worker; the returned function embeds one processor output. */
export async function loadWasmRunner(
  overrides: Record<string, number>,
  dtype: string,
  verbose: boolean,
  step: (phase: string) => void
): Promise<(inp: Inp) => Promise<Float32Array>> {
  const w = new Worker(new URL('./npuWorker.ts', import.meta.url), { type: 'module' })
  const pending = new Map<number, { resolve: (v: Float32Array) => void; reject: (e: Error) => void }>()
  let next = 0
  const fail = (e: Error, reject: (e: Error) => void) => {
    reject(e)
    for (const p of pending.values()) p.reject(e)
    pending.clear()
  }
  await new Promise<void>((resolve, reject) => {
    w.onmessage = (e: MessageEvent<{ type: string; id?: number; phase?: string; v?: Float32Array; message?: string }>) => {
      const m = e.data
      if (m.type === 'phase') step(`wasm: ${m.phase}`)
      else if (m.type === 'loaded') resolve()
      else if (m.type === 'vec') {
        pending.get(m.id!)?.resolve(m.v!)
        pending.delete(m.id!)
      } else if (m.type === 'error') {
        const err = new Error(m.message)
        if (m.id === undefined) fail(err, reject)
        else {
          pending.get(m.id)?.reject(err)
          pending.delete(m.id)
        }
      }
    }
    w.onerror = (e) => fail(new Error(e.message || 'the WASM worker crashed'), reject)
    w.postMessage({ type: 'load', dtype, overrides, verbose })
  })
  return (inp) =>
    new Promise((resolve, reject) => {
      const id = next++
      pending.set(id, { resolve, reject })
      const pick = (k: string) => {
        const x = inp[k] as unknown as { type: string; data: unknown; dims: number[] }
        return { type: x.type, data: x.data, dims: x.dims }
      }
      w.postMessage({
        type: 'embed',
        id,
        inp: {
          input_ids: pick('input_ids'),
          attention_mask: pick('attention_mask'),
          pixel_values: pick('pixel_values'),
          image_position_ids: pick('image_position_ids')
        }
      })
    })
}

