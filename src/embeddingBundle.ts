/**
 * Portable file of cached embeddings and captions, for moving them between devices
 * (desktop overnight run → phone) or keeping a copy on a server or next to the photos.
 *
 * Layout: "MCEMB1" · u32 LE header length · header JSON · zero padding to 4 bytes · float32 vectors.
 * Keys are the IndexedDB keys as-is (`<model prefix><basename>:<size>:<lastModified>`). `sha256` maps
 * a file key (`<basename>:<size>:<lastModified>`) to the file's hash, so another device, where the same
 * file has a different lastModified, can still find which of its files an entry belongs to.
 */

export const BUNDLE_FILE = 'media-clusterer.mcemb'
const MAGIC = 'MCEMB1'

interface Header {
  dims: number[] // per entry, so models of different widths share one file
  keys: string[]
  captions: [string, string][]
  sha256?: Record<string, string>
}

export interface Bundle {
  entries: [string, Float32Array][]
  captions: [string, string][]
  sha256: Record<string, string>
}

export function encodeBundle({ entries, captions, sha256 }: Bundle): Blob {
  const header: Header = { keys: entries.map((e) => e[0]), dims: entries.map((e) => e[1].length), captions, sha256 }
  const json = new TextEncoder().encode(JSON.stringify(header))
  const head = new Uint8Array(MAGIC.length + 4 + json.length + ((4 - ((MAGIC.length + 4 + json.length) % 4)) % 4))
  head.set(new TextEncoder().encode(MAGIC))
  new DataView(head.buffer).setUint32(MAGIC.length, json.length, true)
  head.set(json, MAGIC.length + 4)
  return new Blob([head, ...entries.map((e) => e[1] as Float32Array<ArrayBuffer>)], { type: 'application/octet-stream' })
}

export function decodeBundle(buf: ArrayBuffer): Bundle {
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
    if (!Number.isInteger(d) || d < 0 || off + d * 4 > buf.byteLength) throw new Error('Embeddings file is truncated')
    entries.push([String(header.keys[i]), new Float32Array(buf, off, d).slice()])
    off += d * 4
  }
  const captions = (Array.isArray(header.captions) ? header.captions : []).filter(
    (c): c is [string, string] => Array.isArray(c) && typeof c[0] === 'string' && typeof c[1] === 'string'
  )
  const sha256: Record<string, string> = {}
  for (const [k, v] of Object.entries(header.sha256 ?? {})) if (/^[0-9a-f]{64}$/.test(String(v))) sha256[k] = String(v)
  return { entries, captions, sha256 }
}

/** The file key an IndexedDB key belongs to: the key without its model prefix. */
export const fileKeyOf = (key: string) => key.slice(key.lastIndexOf('/') + 1)

/**
 * Re-key `bundle` onto this device's files: an entry whose file key is not here is moved to the local
 * file with the same sha256. `hash` is only called for local files that could match (same size).
 */
export async function rekeyBundle(
  bundle: Bundle,
  localKeys: string[],
  hash: (localKey: string) => Promise<string | null>
): Promise<{ bundle: Bundle; matched: number }> {
  const local = new Set(localKeys)
  const foreign = new Map<string, string>() // sha → foreign file key
  for (const [fk, sha] of Object.entries(bundle.sha256)) if (!local.has(fk)) foreign.set(sha, fk)
  const sizes = new Set([...foreign.values()].map((fk) => fk.split(':').at(-2)))
  const known = new Set(Object.keys(bundle.sha256))
  const move = new Map<string, string>() // foreign file key → local file key
  for (const lk of localKeys) {
    if (known.has(lk) || !sizes.has(lk.split(':').at(-2))) continue
    const sha = await hash(lk)
    const fk = sha ? foreign.get(sha) : undefined
    if (fk && !move.has(fk)) move.set(fk, lk)
  }
  const to = (key: string) => {
    const lk = move.get(fileKeyOf(key))
    return lk ? key.slice(0, key.length - fileKeyOf(key).length) + lk : key
  }
  return {
    matched: move.size,
    bundle: {
      entries: bundle.entries.map(([k, v]) => [to(k), v]),
      captions: bundle.captions.map(([k, v]) => [to(k), v]),
      sha256: bundle.sha256
    }
  }
}

export async function sha256Hex(file: Blob): Promise<string | null> {
  // shortcut: hashes the whole file in memory, so files over 1 GB (long videos) are skipped and never match by hash; stream through a WASM SHA-256 if that matters.
  if (file.size > 1 << 30) return null
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer()))
  return Array.from(d, (b) => b.toString(16).padStart(2, '0')).join('')
}

