/**
 * Portable file of cached embeddings and captions, for moving them between devices
 * (desktop overnight run → phone) or keeping a copy on a server or next to the photos.
 *
 * Layout: "MCEMB1" · u32 LE header length · header JSON · zero padding to 4 bytes · float32 vectors.
 * Keys are the IndexedDB keys as-is (`<model prefix><basename>:<size>:<lastModified>`).
 */

export const BUNDLE_FILE = 'media-clusterer.mcemb'
const MAGIC = 'MCEMB1'

interface Header {
  dims: number[] // per entry, so models of different widths share one file
  keys: string[]
  captions: [string, string][]
}

export function encodeBundle(entries: [string, Float32Array][], captions: [string, string][]): Blob {
  const header: Header = { keys: entries.map((e) => e[0]), dims: entries.map((e) => e[1].length), captions }
  const json = new TextEncoder().encode(JSON.stringify(header))
  const head = new Uint8Array(MAGIC.length + 4 + json.length + ((4 - ((MAGIC.length + 4 + json.length) % 4)) % 4))
  head.set(new TextEncoder().encode(MAGIC))
  new DataView(head.buffer).setUint32(MAGIC.length, json.length, true)
  head.set(json, MAGIC.length + 4)
  return new Blob([head, ...entries.map((e) => e[1] as Float32Array<ArrayBuffer>)], { type: 'application/octet-stream' })
}

export function decodeBundle(buf: ArrayBuffer): { entries: [string, Float32Array][]; captions: [string, string][] } {
  const bytes = new Uint8Array(buf)
  if (bytes.length < MAGIC.length + 4 || new TextDecoder().decode(bytes.subarray(0, MAGIC.length)) !== MAGIC) {
    throw new Error('Not a media-clusterer embeddings file')
  }
  const len = new DataView(buf).getUint32(MAGIC.length, true)
  const start = MAGIC.length + 4
  if (start + len > bytes.length) throw new Error('Embeddings file is truncated')
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(start, start + len))) as Header
  if (!Array.isArray(header.keys) || !Array.isArray(header.dims) || header.keys.length !== header.dims.length) {
    throw new Error('Embeddings file header is malformed')
  }
  let off = start + len + ((4 - ((start + len) % 4)) % 4)
  const entries: [string, Float32Array][] = []
  for (let i = 0; i < header.keys.length; i++) {
    const d = header.dims[i]
    if (!Number.isInteger(d) || d <= 0 || off + d * 4 > buf.byteLength) throw new Error('Embeddings file is truncated')
    entries.push([String(header.keys[i]), new Float32Array(buf, off, d).slice()])
    off += d * 4
  }
  const captions = (Array.isArray(header.captions) ? header.captions : []).filter(
    (c): c is [string, string] => Array.isArray(c) && typeof c[0] === 'string' && typeof c[1] === 'string'
  )
  return { entries, captions }
}

