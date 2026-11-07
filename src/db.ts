/**
 * IndexedDB cache for embeddings and captions.
 *
 * Captions moved here in v2. They previously lived in localStorage, written
 * synchronously one `setItem` per file from inside the embed loop, with no
 * eviction and only a `console.warn` on quota exceed (IMPROVEMENT_PLAN P1-1).
 * Multi-frame video captions are longer than image captions, so that path had
 * to go before tiers B/C could ship.
 */

import type { CacheKey } from './types'

let db: IDBDatabase | null = null

const DB_NAME = 'photo-organizer-v1'
const STORE_NAME = 'embeddings'
const CAPTION_STORE = 'captions'
const DB_VERSION = 2

/**
 * Open IndexedDB database
 */
export async function openDB(): Promise<IDBDatabase> {
  if (db) return db

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onupgradeneeded = () => {
      const db = request.result
      // Guarded individually rather than switching on oldVersion, so a browser
      // at any prior version converges to the same schema.
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME)
      }
      if (!db.objectStoreNames.contains(CAPTION_STORE)) {
        db.createObjectStore(CAPTION_STORE)
      }
    }

    request.onsuccess = () => {
      db = request.result
      resolve(db)
    }

    request.onerror = () => {
      reject(request.error)
    }
  })
}

/**
 * Coerce a cached value to Float32Array. Embeddings written by older builds
 * were stored as Float64Array; convert on read so callers see one type and
 * rewrites gradually shrink the store.
 */
export function asFloat32(value: unknown): Float32Array | null {
  // compactVector also trims any oversized buffer an older build stored.
  if (value instanceof Float32Array || value instanceof Float64Array) return compactVector(value)
  return null
}

/**
 * Get single embedding from cache
 */
export async function cacheGet(key: CacheKey): Promise<Float32Array | null> {
  const [result] = await cacheGetBatch([key])
  return result
}

/**
 * Batch get embeddings from cache
 */
export async function cacheGetBatch(keys: CacheKey[]): Promise<(Float32Array | null)[]> {
  const database = await openDB()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readonly')
    const store = transaction.objectStore(STORE_NAME)
    const results: (Float32Array | null)[] = new Array(keys.length).fill(null)
    let count = 0

    keys.forEach((key, i) => {
      const request = store.get(key)
      request.onsuccess = () => {
        results[i] = asFloat32(request.result)
        count++
        if (count === keys.length) resolve(results)
      }
      request.onerror = () => {
        reject(request.error)
      }
    })

    if (keys.length === 0) resolve([])
  })
}

/**
 * The value as a Float32Array that owns exactly its bytes. IndexedDB clones the
 * whole underlying ArrayBuffer, so storing a 768-float view into a batch tensor
 * would store the entire tensor; any other numeric type is narrowed to Float32.
 */
export function compactVector(value: ArrayLike<number>): Float32Array {
  if (value instanceof Float32Array && value.byteOffset === 0 && value.byteLength === value.buffer.byteLength) return value
  return Float32Array.from(value)
}

/**
 * Put embedding in cache
 */
export async function cachePut(key: CacheKey, value: Float32Array): Promise<void> {
  const database = await openDB()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite')
    const store = transaction.objectStore(STORE_NAME)
    store.put(compactVector(value), key)

    transaction.oncomplete = () => {
      resolve()
    }

    transaction.onerror = () => {
      reject(transaction.error)
    }
  })
}

/**
 * Batch put embeddings (20 per transaction for performance)
 */
export async function cachePutBatch(entries: [CacheKey, Float32Array][]): Promise<void> {
  if (entries.length === 0) return
  const database = await openDB()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite')
    const store = transaction.objectStore(STORE_NAME)

    for (const [key, value] of entries) {
      store.put(compactVector(value), key)
    }

    transaction.oncomplete = () => {
      resolve()
    }

    transaction.onerror = () => {
      reject(transaction.error)
    }
  })
}

/**
 * Return count and total bytes of cached embeddings, optionally filtered by key prefix.
 */
export async function cacheStats(prefix?: string): Promise<{ count: number; bytes: number }> {
  const database = await openDB()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readonly')
    const store = transaction.objectStore(STORE_NAME)
    let count = 0
    let bytes = 0

    const req = store.openCursor()
    req.onsuccess = () => {
      const cursor = req.result
      if (cursor) {
        const key = String(cursor.key)
        if (!prefix || key.startsWith(prefix)) {
          count++
          // The buffer, not the view: it is what IndexedDB actually stored.
          if (ArrayBuffer.isView(cursor.value)) bytes += cursor.value.buffer.byteLength
        }
        cursor.continue()
      } else {
        resolve({ count, bytes })
      }
    }
    req.onerror = () => reject(req.error)
  })
}

/** Which of `keys` are cached, positionally. Key-only lookups: no vector is read. */
export async function cacheHasBatch(keys: CacheKey[]): Promise<boolean[]> {
  const database = await openDB()
  return new Promise((resolve, reject) => {
    const tx = database.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const out = new Array<boolean>(keys.length).fill(false)
    keys.forEach((k, i) => {
      const req = store.getKey(k)
      req.onsuccess = () => {
        out[i] = req.result !== undefined
      }
    })
    tx.oncomplete = () => resolve(out)
    tx.onerror = () => reject(tx.error)
  })
}

/** Every embedding key, all models. One getAllKeys call; keys are small strings. */
export async function cacheKeys(): Promise<string[]> {
  const database = await openDB()
  return new Promise((resolve, reject) => {
    const req = database.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).getAllKeys()
    req.onsuccess = () => resolve(req.result.map(String))
    req.onerror = () => reject(req.error)
  })
}

// ── Captions ────────────────────────────────────────────────────────────────
// Same key shape as embeddings (`name:size:lastModified`, folder-independent)
// but a separate store, because captions are model-agnostic text while
// embeddings are namespaced per model.

/** Batch-read captions. Missing entries come back as null, positionally. */
export async function captionGetBatch(keys: CacheKey[]): Promise<(string | null)[]> {
  if (keys.length === 0) return []
  const database = await openDB()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(CAPTION_STORE, 'readonly')
    const store = transaction.objectStore(CAPTION_STORE)
    const results: (string | null)[] = new Array(keys.length).fill(null)
    let count = 0

    keys.forEach((key, i) => {
      const request = store.get(key)
      request.onsuccess = () => {
        results[i] = typeof request.result === 'string' ? request.result : null
        count++
        if (count === keys.length) resolve(results)
      }
      request.onerror = () => reject(request.error)
    })
  })
}

/**
 * Batch-write captions in one transaction.
 *
 * The whole point of this function is that it is *batched* and asynchronous:
 * the caller queues writes alongside the embedding write queue rather than
 * blocking the embed loop once per file.
 */
export async function captionPutBatch(entries: [CacheKey, string][]): Promise<void> {
  if (entries.length === 0) return
  const database = await openDB()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(CAPTION_STORE, 'readwrite')
    const store = transaction.objectStore(CAPTION_STORE)

    for (const [key, value] of entries) {
      store.put(value, key)
    }

    transaction.oncomplete = () => resolve()
    transaction.onerror = () => reject(transaction.error)
  })
}

// ── Bundle export / import ──────────────────────────────────────────────────

/**
 * Every embedding and caption, optionally only those of the given file keys (`name:size:lastModified`).
 * shortcut: holds the whole cache in memory at once (≈3 KB per vector); stream through a cursor if exports reach GBs.
 */
export async function exportAll(onlyFileKeys?: Set<string>): Promise<{
  entries: [string, Float32Array][]
  captions: [string, string][]
}> {
  const database = await openDB()
  const keep = (k: string) => !onlyFileKeys || onlyFileKeys.has(k.slice(k.lastIndexOf('/') + 1))
  const readStore = <T>(name: string, map: (v: unknown) => T | null) =>
    new Promise<[string, T][]>((resolve, reject) => {
      const out: [string, T][] = []
      const req = database.transaction(name, 'readonly').objectStore(name).openCursor()
      req.onsuccess = () => {
        const c = req.result
        if (!c) return resolve(out)
        const v = keep(String(c.key)) ? map(c.value) : null
        if (v !== null) out.push([String(c.key), v])
        c.continue()
      }
      req.onerror = () => reject(req.error)
    })
  return {
    entries: await readStore(STORE_NAME, asFloat32),
    captions: await readStore(CAPTION_STORE, (v) => (typeof v === 'string' ? v : null))
  }
}

/**
 * Copy vectors stored under another device's key for the same file (same basename and size,
 * different lastModified — copies and phone galleries rewrite it) to this device's key.
 * Returns how many were adopted.
 */
export async function adoptByNameSize(fileKeys: string[]): Promise<number> {
  const all = await cacheKeys()
  const have = new Set(all)
  const byNameSize = new Map<string, string>() // `${prefix}${name}:${size}` → stored key
  const prefixes = new Set<string>()
  for (const k of all) {
    const cut = k.lastIndexOf(':')
    const p = k.slice(0, k.lastIndexOf('/', cut) + 1)
    if (p !== '' && !/^@[^/]+\/$/.test(p)) continue // legacy full-path keys, not a model prefix
    byNameSize.set(k.slice(0, cut), k)
    prefixes.add(p)
  }
  const pairs: [from: CacheKey, to: CacheKey][] = []
  for (const fk of fileKeys) {
    const ns = fk.slice(0, fk.lastIndexOf(':'))
    for (const p of prefixes) {
      const from = byNameSize.get(p + ns)
      if (from && !have.has(p + fk)) pairs.push([from as CacheKey, (p + fk) as CacheKey])
    }
  }
  if (pairs.length === 0) return 0
  const vectors = await cacheGetBatch(pairs.map((p) => p[0]))
  await cachePutBatch(pairs.flatMap(([, to], i) => (vectors[i] ? [[to, vectors[i]!] as [CacheKey, Float32Array]] : [])))
  // Captions are keyed without a model prefix.
  const capFrom = pairs.filter(([f]) => !f.includes('/')).map(([f]) => f)
  if (capFrom.length) {
    const caps = await captionGetBatch(capFrom)
    const toOf = new Map(pairs)
    await captionPutBatch(capFrom.flatMap((f, i) => (caps[i] ? [[toOf.get(f)!, caps[i]!] as [CacheKey, string]] : [])))
  }
  return pairs.length
}
