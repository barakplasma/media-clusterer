/**
 * OpenAI-compatible remote inference — HTTP surface.
 *
 * This module is the entire network boundary for "Remote AI Mode": the app
 * points at an OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, Jina,
 * Infinity, llama.cpp, …) and embeds media there instead of in the browser.
 * See docs/adr/0002-openai-compatible-remote-inference.md and
 * REMOTE_INFERENCE_PLAN.md.
 *
 * Deliberately dependency-light and DOM-free (except `imageToDataURL`, which
 * needs canvas): no app imports, no `state`, no settings. That keeps it unit
 * testable against a stubbed `fetch`, the same way `src/modelFallback.ts` is.
 *
 * MVP scope — this is milestone M1 cut down to the smallest useful core:
 *   - text and image embedding via `POST {baseUrl}/embeddings`
 *   - images use the `openai-multimodal` wire format only (the OpenRouter
 *     shape). The vLLM / Jina / Infinity / llama.cpp variants and the
 *     auto-probe that picks between them are not implemented yet.
 *   - no VLM caption pipeline, no multi-frame video.
 *
 * Secrets rule (AGENT.md): the API key must never reach a log line, an error
 * report, or a URL. Every error message constructed here goes through
 * `redactSecrets()` at construction, so an un-redacted string cannot escape by
 * any route — including one thrown from a provider's own error body.
 */

import pLimit from 'p-limit'
import { l2normalize } from './embeddings'

/** Connection details for an OpenAI-compatible endpoint. */
export interface OpenAICompatConfig {
  /** Normalized base URL, e.g. `https://openrouter.ai/api/v1`. */
  baseUrl: string
  /** Bearer token. Empty string means "send no Authorization header". */
  apiKey: string
  /** Model id passed as `model` in each request body. */
  model: string
  /** Parallel in-flight requests. Default 4. */
  concurrency?: number
  /** Per-request timeout in ms. Default 60_000. */
  timeoutMs?: number
  /** Retry attempts after the first try. Default 3. */
  maxRetries?: number
  /** Inputs per request. Default 16 for text, 8 for images. */
  batchSize?: number
}

const DEFAULT_CONCURRENCY = 4
const DEFAULT_TIMEOUT_MS = 60_000
const DEFAULT_MAX_RETRIES = 3
const DEFAULT_TEXT_BATCH = 16
const DEFAULT_IMAGE_BATCH = 8

/**
 * Error bodies where a *gateway* returns a non-auth status but the quoted
 * upstream failure is a rejected provider credential. Matched against the
 * response body, so the wording is the provider's, not the gateway's.
 */
const UPSTREAM_KEY_REJECTED =
  /api[_ -]?key not valid|invalid[_ -]?api[_ -]?key|API_KEY_INVALID|api key expired/i

/** Ceiling on an honoured `Retry-After`, so a hostile value can't hang the UI. */
const MAX_RETRY_AFTER_MS = 20_000
const BASE_BACKOFF_MS = 500
const MAX_BACKOFF_MS = 8_000

// ── URL handling ────────────────────────────────────────────────────────────

/** Hostnames that should default to `http`, not `https`. */
function isLocalHostname(host: string): boolean {
  const h = host.toLowerCase().replace(/^\[|\]$/g, '')
  if (h === 'localhost' || h.endsWith('.localhost') || h.endsWith('.local')) return true
  if (h === '::1' || h === '0.0.0.0') return true
  if (/^127\./.test(h)) return true
  if (/^10\./.test(h)) return true
  if (/^192\.168\./.test(h)) return true
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(h)) return true
  return false
}

/**
 * Turn whatever the user pasted into a base URL ending at the API root.
 *
 * Handles the three shapes people actually paste: a bare host with a port
 * (`localhost:11434`), a host with the version path already on it
 * (`openrouter.ai/api/v1`), and a full endpoint URL copied out of provider
 * docs (`https://api.openai.com/v1/embeddings`).
 *
 * Returns `''` for unparseable input rather than a URL that would fail
 * confusingly on the first request.
 */
export function normalizeBaseUrl(input: string | undefined | null): string {
  const raw = (input ?? '').trim()
  if (!raw) return ''

  // `localhost:11434` parses as scheme `localhost:` unless we spot that the
  // scheme separator `://` is absent and supply one ourselves.
  const hasScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw)
  let candidate = raw
  if (!hasScheme) {
    const hostPart = raw.split(/[/?#]/)[0]
    const hostname = hostPart.replace(/:\d+$/, '')
    candidate = `${isLocalHostname(hostname) ? 'http' : 'https'}://${raw}`
  }

  let url: URL
  try {
    url = new URL(candidate)
  } catch {
    return ''
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return ''
  if (!url.hostname) return ''

  // Drop a pasted endpoint suffix so `/v1/embeddings` becomes `/v1`.
  let path = url.pathname.replace(/\/+$/, '')
  path = path.replace(/\/(embeddings|chat\/completions|completions|models)$/, '')

  // Only supply `/v1` when the user gave no path at all — providers that mount
  // the API at the root, or at something other than `/v1`, must be left alone.
  if (path === '') path = '/v1'

  return `${url.protocol}//${url.host}${path}`
}

// ── Secret redaction ────────────────────────────────────────────────────────

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Strip the live key and anything else key-shaped out of a string.
 *
 * Two independent passes on purpose: the exact-key pass catches our own key,
 * and the generic-shape pass catches a provider that echoes a *different*
 * token back inside an error body. Either alone would leak in the other case.
 */
export function redactSecrets(input: string, key?: string): string {
  let out = input ?? ''

  // Short keys are not distinctive enough to replace safely — a 3-character
  // "key" would shred unrelated text.
  if (key && key.length >= 8) {
    out = out.replace(new RegExp(escapeRegExp(key), 'g'), '***')
  }

  out = out.replace(/\bsk-[A-Za-z0-9._~+/-]{8,}/g, 'sk-***')
  out = out.replace(/\bBearer\s+[A-Za-z0-9._~+/-]+=*/gi, 'Bearer ***')
  out = out.replace(/([?&](?:api[-_]?key|access[-_]?token|token|key)=)[^&\s]+/gi, '$1***')

  return out
}

// ── Errors ──────────────────────────────────────────────────────────────────

export type OpenAIErrorKind =
  | 'network'
  | 'timeout'
  | 'aborted'
  | 'auth'
  | 'rate-limit'
  | 'server'
  | 'bad-request'
  | 'not-found'
  | 'payload-too-large'
  | 'parse'

/**
 * Error from the remote endpoint or the transport to it.
 *
 * Carries `host` only — never the full URL (which may hold a query-string key)
 * and never the config object (which holds the key outright).
 */
export class OpenAICompatError extends Error {
  readonly kind: OpenAIErrorKind
  readonly status?: number
  readonly retryable: boolean
  readonly hint: string
  readonly host: string

  constructor(
    kind: OpenAIErrorKind,
    message: string,
    opts: { status?: number; retryable?: boolean; hint?: string; host?: string; key?: string } = {}
  ) {
    // Redact at construction: there is no path from here to an un-redacted
    // `.message`, however the error is later logged or reported.
    super(redactSecrets(message, opts.key))
    this.name = 'OpenAICompatError'
    this.kind = kind
    this.status = opts.status
    this.retryable = opts.retryable ?? false
    this.hint = redactSecrets(opts.hint ?? '', opts.key)
    this.host = opts.host ?? ''
  }
}

function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).host
  } catch {
    return ''
  }
}

/** Map an HTTP status onto our error taxonomy. */
function classifyStatus(status: number): { kind: OpenAIErrorKind; retryable: boolean; hint: string } {
  if (status === 401 || status === 403) {
    return {
      kind: 'auth',
      retryable: false,
      hint: 'The endpoint rejected the API key. Check the key and that it is valid for this host.'
    }
  }
  if (status === 404) {
    return {
      kind: 'not-found',
      retryable: false,
      hint: 'No such endpoint or model. Check the base URL includes the right version path (often /v1) and that the model id exists.'
    }
  }
  if (status === 413) {
    return {
      kind: 'payload-too-large',
      retryable: false,
      hint: 'The request body was too large. Lower the batch size or the thumbnail width.'
    }
  }
  if (status === 429) {
    return { kind: 'rate-limit', retryable: true, hint: 'Rate limited by the provider.' }
  }
  if (status === 408) {
    return { kind: 'timeout', retryable: true, hint: 'The endpoint timed out.' }
  }
  if (status >= 500) {
    return { kind: 'server', retryable: true, hint: 'The endpoint reported a server error.' }
  }
  return {
    kind: 'bad-request',
    retryable: false,
    hint: 'The endpoint rejected the request. The model may not accept this input type.'
  }
}

/** One-line, user-facing summary. Safe to render — already redacted. */
export function describeOpenAIError(err: unknown): string {
  if (err instanceof OpenAICompatError) {
    const where = err.host ? ` (${err.host})` : ''
    return err.hint ? `${err.message}${where} — ${err.hint}` : `${err.message}${where}`
  }
  if (err instanceof Error) return redactSecrets(err.message)
  return 'Unknown error'
}

// ── Abort / timing helpers ──────────────────────────────────────────────────

function abortError(host: string): OpenAICompatError {
  return new OpenAICompatError('aborted', 'Request cancelled', { host })
}

/** `setTimeout` that rejects the moment `signal` aborts. */
function sleep(ms: number, signal: AbortSignal | undefined, host: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError(host))
      return
    }
    const onAbort = () => {
      clearTimeout(timer)
      reject(abortError(host))
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/**
 * `AbortSignal.any`, with a manual fallback.
 *
 * jsdom does not implement `AbortSignal.any`, so the fallback is not
 * hypothetical — it is the path the tests take.
 */
function anySignal(signals: (AbortSignal | undefined)[]): AbortSignal | undefined {
  const present = signals.filter((s): s is AbortSignal => !!s)
  if (present.length === 0) return undefined
  if (present.length === 1) return present[0]

  const AnyFn = (AbortSignal as unknown as { any?: (s: AbortSignal[]) => AbortSignal }).any
  if (typeof AnyFn === 'function') return AnyFn(present)

  const ctrl = new AbortController()
  for (const s of present) {
    if (s.aborted) {
      ctrl.abort(s.reason)
      break
    }
    s.addEventListener('abort', () => ctrl.abort(s.reason), { once: true })
  }
  return ctrl.signal
}

/** `Retry-After` in either integer-seconds or HTTP-date form. `null` if absent/bogus. */
export function parseRetryAfter(value: string | null, nowMs: number = Date.now()): number | null {
  if (!value) return null
  const trimmed = value.trim()

  if (/^\d+$/.test(trimmed)) {
    return Math.min(Number(trimmed) * 1000, MAX_RETRY_AFTER_MS)
  }

  const when = Date.parse(trimmed)
  if (Number.isNaN(when)) return null
  return Math.min(Math.max(when - nowMs, 0), MAX_RETRY_AFTER_MS)
}

/** Full-jitter exponential backoff. */
function backoffMs(attempt: number): number {
  const ceiling = Math.min(BASE_BACKOFF_MS * 2 ** attempt, MAX_BACKOFF_MS)
  return Math.random() * ceiling
}

// ── The single HTTP chokepoint ──────────────────────────────────────────────

/**
 * POST JSON to `{baseUrl}{path}` and parse the JSON response.
 *
 * Every request this module makes goes through here, so the auth header,
 * retry policy, timeout and redaction are decided in exactly one place.
 */
export async function requestJSON<T>(
  cfg: OpenAICompatConfig,
  path: string,
  body: unknown,
  signal?: AbortSignal,
  method: 'GET' | 'POST' = 'POST'
): Promise<T> {
  const host = hostOf(cfg.baseUrl)
  const url = `${cfg.baseUrl}${path}`
  const maxRetries = cfg.maxRetries ?? DEFAULT_MAX_RETRIES
  const timeoutMs = cfg.timeoutMs ?? DEFAULT_TIMEOUT_MS

  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  // A bare `Bearer` with no token is worse than no header at all: local
  // servers reject it outright, and they need no key in the first place.
  if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`

  let lastError: OpenAICompatError | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (signal?.aborted) throw abortError(host)

    const timeoutCtrl = new AbortController()
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      timeoutCtrl.abort()
    }, timeoutMs)

    let res: Response
    try {
      res = await fetch(url, {
        method,
        headers,
        // A GET with a body is rejected outright by fetch, so omit it.
        ...(method === 'POST' ? { body: JSON.stringify(body) } : {}),
        signal: anySignal([signal, timeoutCtrl.signal])
      })
    } catch (err) {
      clearTimeout(timer)
      // The user's cancel wins over our timeout when both fired.
      if (signal?.aborted) throw abortError(host)
      if (timedOut) {
        lastError = new OpenAICompatError('timeout', `Request to ${host} timed out`, {
          retryable: true,
          hint: `No response within ${Math.round(timeoutMs / 1000)}s.`,
          host,
          key: cfg.apiKey
        })
      } else {
        lastError = new OpenAICompatError(
          'network',
          `Could not reach ${host}: ${(err as Error)?.message ?? 'network error'}`,
          {
            retryable: true,
            hint: 'Check the base URL, your connection, and that the server allows cross-origin requests from this page.',
            host,
            key: cfg.apiKey
          }
        )
      }
      if (attempt < maxRetries) {
        await sleep(backoffMs(attempt), signal, host)
        continue
      }
      throw lastError
    }
    clearTimeout(timer)

    if (!res.ok) {
      let { kind, retryable, hint } = classifyStatus(res.status)
      // Provider error bodies routinely quote the request back, key included.
      // `OpenAICompatError` redacts at construction, so this is safe to embed.
      const detail = await res.text().catch(() => '')

      // A gateway can answer 400 when the real fault is an *upstream*
      // credential. OpenRouter's BYOK forwards to the provider using a key you
      // configured there, so a stale Google key comes back as Google's own
      // "API key not valid" wrapped in a 400. The generic 400 hint blames the
      // input type, which sends people off rewriting a request that was
      // already correct — say what actually happened instead.
      if (UPSTREAM_KEY_REJECTED.test(detail)) {
        kind = 'auth'
        retryable = false
        hint =
          'An upstream provider rejected its own API key, not yours. If the gateway is configured to bring-your-own-key for this model, that stored provider key is invalid or expired — fix or remove it in the gateway’s integration settings.'
      }

      const err = new OpenAICompatError(
        kind,
        `${host} returned ${res.status}${detail ? `: ${detail.slice(0, 300)}` : ''}`,
        { status: res.status, retryable, hint, host, key: cfg.apiKey }
      )

      if (!retryable || attempt >= maxRetries) throw err
      lastError = err

      const after = parseRetryAfter(res.headers?.get?.('Retry-After') ?? null)
      await sleep(after ?? backoffMs(attempt), signal, host)
      continue
    }

    try {
      return (await res.json()) as T
    } catch {
      throw new OpenAICompatError('parse', `${host} returned a non-JSON response`, {
        status: res.status,
        hint: 'The base URL may point at a web page rather than an API root.',
        host,
        key: cfg.apiKey
      })
    }
  }

  throw lastError ?? new OpenAICompatError('network', `Request to ${host} failed`, { host })
}

// ── Response parsing ────────────────────────────────────────────────────────

interface EmbeddingsResponse {
  data?: Array<{ embedding?: number[]; index?: number }>
  embedding?: number[]
}

/**
 * Pull vectors out of an embeddings response, in request order.
 *
 * Sorting by `index` matters more than it looks: the spec does not guarantee
 * response order, and getting it wrong silently pairs the wrong vector with
 * the wrong file — invisible until someone notices the map is nonsense.
 */
export function parseEmbeddings(
  json: EmbeddingsResponse,
  expected: number,
  host = ''
): Float32Array[] {
  // llama.cpp returns a single bare embedding rather than a `data` array.
  if (!json?.data && Array.isArray(json?.embedding)) {
    return [l2normalize(Float32Array.from(json.embedding))]
  }

  const rows = json?.data
  if (!Array.isArray(rows)) {
    throw new OpenAICompatError('parse', 'Response had no `data` array of embeddings', {
      host,
      hint: 'The endpoint may not be an OpenAI-compatible embeddings API.'
    })
  }

  const ordered = rows.every((r) => typeof r?.index === 'number')
    ? [...rows].sort((a, b) => (a.index as number) - (b.index as number))
    : rows

  const out = ordered.map((r) => {
    if (!Array.isArray(r?.embedding)) {
      throw new OpenAICompatError('parse', 'An entry in `data` had no numeric `embedding`', { host })
    }
    return l2normalize(Float32Array.from(r.embedding))
  })

  if (out.length !== expected) {
    throw new OpenAICompatError(
      'parse',
      `Expected ${expected} embeddings but the endpoint returned ${out.length}`,
      { host, hint: 'Try a smaller batch size.' }
    )
  }
  return out
}

// ── Image downscaling ───────────────────────────────────────────────────────

/**
 * Target dimensions for a thumbnail bounded by `maxWidth`.
 *
 * Split out from `imageToDataURL` because jsdom has neither `OffscreenCanvas`
 * nor `createImageBitmap` — keeping the arithmetic pure makes it exhaustively
 * testable without brittle canvas stubs.
 *
 * Note this bounds *width* only, matching `createImageBitmap`'s `resizeWidth`.
 * A tall portrait therefore keeps its height; bounding the longest edge would
 * upload less, and is worth revisiting when real payload sizes are measured.
 */
export function computeTargetSize(
  width: number,
  height: number,
  maxWidth: number
): { width: number; height: number } {
  if (!(width > 0) || !(height > 0) || !(maxWidth > 0)) return { width: 0, height: 0 }
  if (width <= maxWidth) return { width: Math.round(width), height: Math.round(height) }
  const scale = maxWidth / width
  return { width: Math.round(maxWidth), height: Math.max(1, Math.round(height * scale)) }
}

/**
 * Downscale to a JPEG data URI suitable for upload.
 *
 * JPEG rather than PNG (5–10× smaller for photos); alpha is flattened onto
 * white since JPEG has no alpha channel.
 *
 * Ownership rule (AGENT.md): this closes only bitmaps it created itself. An
 * `ImageBitmap` passed in by the caller stays the caller's to close — see the
 * comments at `src/app.ts:1491` and `:1514` for why that rule exists.
 */
export async function imageToDataURL(
  source: Blob | ImageBitmap,
  maxWidth = 384,
  quality = 0.8
): Promise<string> {
  let bitmap: ImageBitmap
  let owned = false

  if (typeof Blob !== 'undefined' && source instanceof Blob) {
    // resizeWidth decodes at the target size — a 48 MP original is never
    // fully decoded just to be thrown away.
    bitmap = await createImageBitmap(source, { resizeWidth: maxWidth, resizeQuality: 'high' })
    owned = true
  } else {
    bitmap = source as ImageBitmap
  }

  try {
    const { width, height } = computeTargetSize(bitmap.width, bitmap.height, maxWidth)
    const canvas = new OffscreenCanvas(width, height)
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Could not get a 2D context for thumbnail encoding')

    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, width, height)
    ctx.drawImage(bitmap, 0, 0, width, height)

    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality })
    const buf = new Uint8Array(await blob.arrayBuffer())

    let binary = ''
    for (let i = 0; i < buf.length; i++) binary += String.fromCharCode(buf[i])
    return `data:image/jpeg;base64,${btoa(binary)}`
  } finally {
    if (owned) bitmap.close()
  }
}

// ── Embedding ───────────────────────────────────────────────────────────────

function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size))
  return out
}

/**
 * Run batches through the endpoint with bounded concurrency, preserving order.
 *
 * `pLimit` is created here rather than per-batch so a caller can pass an array
 * of any size without stampeding the endpoint.
 */
async function embedBatches(
  cfg: OpenAICompatConfig,
  batches: unknown[][],
  buildBody: (batch: unknown[]) => unknown,
  signal: AbortSignal | undefined
): Promise<Float32Array[]> {
  const limit = pLimit(cfg.concurrency ?? DEFAULT_CONCURRENCY)
  const host = hostOf(cfg.baseUrl)

  const results = await Promise.all(
    batches.map((batch) =>
      limit(async () => {
        const json = await requestJSON<EmbeddingsResponse>(
          cfg,
          '/embeddings',
          buildBody(batch),
          signal
        )
        return parseEmbeddings(json, batch.length, host)
      })
    )
  )
  return results.flat()
}

/**
 * Embed text via `POST {baseUrl}/embeddings`.
 *
 * Index-time and query-time vectors must come from here alike — mixing this
 * with a locally-computed vector puts the two sides in different spaces and
 * silently ruins search (AGENT.md, "Embedding space").
 */
export async function embedTexts(
  cfg: OpenAICompatConfig,
  texts: string[],
  signal?: AbortSignal
): Promise<Float32Array[]> {
  if (texts.length === 0) return []
  const batches = chunk(texts, cfg.batchSize ?? DEFAULT_TEXT_BATCH)
  return embedBatches(cfg, batches, (batch) => ({ model: cfg.model, input: batch }), signal)
}

/** Convenience wrapper for the single-query search path. */
export async function embedQuery(
  cfg: OpenAICompatConfig,
  text: string,
  signal?: AbortSignal
): Promise<Float32Array> {
  const [vec] = await embedTexts(cfg, [text], signal)
  if (!vec) {
    throw new OpenAICompatError('parse', 'The endpoint returned no embedding for the query', {
      host: hostOf(cfg.baseUrl)
    })
  }
  return vec
}

/**
 * Embed images given as data URIs, using the `openai-multimodal` wire format.
 *
 * MVP limitation: this one format only. vLLM, Jina, Infinity and llama.cpp
 * each want a different body shape, and picking between them automatically is
 * the `probeWireFormat()` work deferred out of this milestone. Against those
 * providers this will fail with a `bad-request`, which `describeOpenAIError`
 * reports as the model not accepting the input type.
 */
export async function embedImages(
  cfg: OpenAICompatConfig,
  dataUrls: string[],
  signal?: AbortSignal
): Promise<Float32Array[]> {
  if (dataUrls.length === 0) return []
  const batches = chunk(dataUrls, cfg.batchSize ?? DEFAULT_IMAGE_BATCH)
  return embedBatches(
    cfg,
    batches,
    (batch) => ({
      model: cfg.model,
      input: (batch as string[]).map((url) => ({
        content: [{ type: 'image_url', image_url: { url } }]
      }))
    }),
    signal
  )
}

// ── Model lookup ────────────────────────────────────────────────────────────

/** One entry from the endpoint's `/models` catalogue. */
export interface ModelInfo {
  id: string
  name?: string
  inputModalities?: string[]
  outputModalities?: string[]
  /** True only when the catalogue positively says it takes images and emits vectors. */
  imageEmbedding: boolean
}

export interface ModelLookup {
  /** Models the catalogue confirms can embed images. */
  imageCapable: ModelInfo[]
  /** Everything the endpoint listed, in catalogue order. */
  all: ModelInfo[]
  /**
   * False when the endpoint returned no capability metadata at all.
   *
   * A plain OpenAI-compatible `/models` returns only ids — Ollama, LM Studio
   * and vLLM all do — so there is nothing to filter on and `imageCapable` is
   * necessarily empty. That is not the same as "this endpoint has no image
   * models", and the UI must not present it as such.
   */
  hasModalityMetadata: boolean
}

interface RawModelEntry {
  id?: string
  name?: string
  architecture?: { input_modalities?: string[]; output_modalities?: string[]; modality?: string }
}

/** Case-insensitive membership, tolerating the odd `Image` / `IMAGE`. */
function hasModality(list: string[] | undefined, want: string): boolean {
  return Array.isArray(list) && list.some((m) => String(m).toLowerCase() === want)
}

export function toModelInfo(raw: RawModelEntry): ModelInfo {
  const inputModalities = raw.architecture?.input_modalities
  const outputModalities = raw.architecture?.output_modalities
  return {
    id: String(raw.id ?? ''),
    name: raw.name,
    inputModalities,
    outputModalities,
    imageEmbedding:
      hasModality(inputModalities, 'image') && hasModality(outputModalities, 'embeddings')
  }
}

/**
 * Find models on this endpoint that can embed images.
 *
 * Two things make this less obvious than it looks:
 *
 * 1. OpenRouter's `/models` defaults to `output_modalities=text`, so an
 *    unqualified GET returns **no embedding models at all**. The query string
 *    asks for embeddings explicitly; servers that don't know the parameter
 *    ignore it, and if the filtered call comes back empty we retry unfiltered
 *    rather than report "none found" off the back of a rejected filter.
 * 2. Only richer catalogues carry modality metadata. Without it nothing can be
 *    filtered, so `hasModalityMetadata` is false and the caller should offer
 *    the raw list instead of claiming there are no image models.
 */
export async function lookupImageEmbeddingModels(
  cfg: OpenAICompatConfig,
  signal?: AbortSignal
): Promise<ModelLookup> {
  const fetchList = async (query: string) => {
    const json = await requestJSON<{ data?: RawModelEntry[]; models?: RawModelEntry[] }>(
      cfg,
      `/models${query}`,
      undefined,
      signal,
      'GET'
    )
    // `data` is the OpenAI shape; Ollama's /v1/models also uses it, but some
    // servers answer with `models`.
    const rows = json?.data ?? json?.models
    return Array.isArray(rows) ? rows : []
  }

  let rows = await fetchList('?output_modalities=embeddings')
  if (rows.length === 0) rows = await fetchList('')

  const all = rows.map(toModelInfo).filter((m) => m.id)
  const hasModalityMetadata = all.some(
    (m) => (m.inputModalities?.length ?? 0) > 0 || (m.outputModalities?.length ?? 0) > 0
  )

  return {
    imageCapable: all.filter((m) => m.imageEmbedding),
    all,
    hasModalityMetadata
  }
}

/**
 * Ask the endpoint for its embedding width by embedding one short string.
 *
 * Must run before any cache key is computed: the dimension is part of the
 * namespace, and a cache keyed at the wrong width would serve vectors from a
 * different space.
 */
export async function probeDimension(
  cfg: OpenAICompatConfig,
  signal?: AbortSignal
): Promise<number> {
  const vec = await embedQuery(cfg, 'dimension probe', signal)
  return vec.length
}

/**
 * IndexedDB cache namespace for a remote configuration.
 *
 * Host and model both participate: either changing means the vectors are from
 * a different space and must not be reused (AGENT.md, `currentCachePrefix`).
 *
 * Dimension is deliberately *not* in the key, because the namespace has to be
 * computable before the endpoint has been probed (the settings panel shows a
 * cache count on load). Host+model determines the dimension in practice, and
 * the case it wouldn't — a provider silently re-pointing a model id at a
 * different width — is caught by the width check at cache-read time instead,
 * which is a stronger guard than a namespace anyway.
 *
 * The API key is absent by design: it is not part of the embedding space, and
 * these keys are persisted to disk.
 */
export function openaiCacheNamespace(cfg: OpenAICompatConfig): string {
  return `@openai:${hostOf(cfg.baseUrl)}:${cfg.model}/`
}

// ── Key storage ─────────────────────────────────────────────────────────────

const KEY_STORAGE = 'mc_openai_key'
const CONSENT_STORAGE = 'mc_openai_consent'

/**
 * The API key is stored under its own storage key, never inside `mc_settings`.
 *
 * Two concrete reasons (ADR-0002): `src/sentry.ts` reads `mc_settings` to
 * decide on error reporting, and `saveSettings()` rewrites that whole blob on
 * nearly every UI interaction. Keeping the key out of it means neither path
 * can pick it up.
 *
 * `remember: false` puts it in `sessionStorage`, so it dies with the tab.
 */
export function getOpenAIKey(): string {
  try {
    return sessionStorage.getItem(KEY_STORAGE) || localStorage.getItem(KEY_STORAGE) || ''
  } catch {
    return ''
  }
}

export function setOpenAIKey(key: string, remember: boolean): void {
  try {
    // Always clear both, or switching "remember" off would leave the old copy
    // sitting in localStorage.
    sessionStorage.removeItem(KEY_STORAGE)
    localStorage.removeItem(KEY_STORAGE)
    if (!key) return
    ;(remember ? localStorage : sessionStorage).setItem(KEY_STORAGE, key)
  } catch {
    /* storage unavailable (private mode, quota) — the key just won't persist */
  }
}

export function clearOpenAIKey(): void {
  setOpenAIKey('', false)
}

/** True when the key is currently in `localStorage` rather than `sessionStorage`. */
export function isOpenAIKeyRemembered(): boolean {
  try {
    return !!localStorage.getItem(KEY_STORAGE)
  } catch {
    return false
  }
}

// ── Upload consent ──────────────────────────────────────────────────────────

/**
 * Consent is per destination host, not global.
 *
 * Agreeing to send thumbnails to a machine on your own LAN says nothing about
 * agreeing to send them to a third-party API, so consent granted for one host
 * must not carry over to another.
 */
export function hasOpenAIConsent(baseUrl: string): boolean {
  const host = hostOf(baseUrl)
  if (!host) return false
  try {
    const raw = localStorage.getItem(CONSENT_STORAGE)
    return raw ? (JSON.parse(raw) as string[]).includes(host) : false
  } catch {
    return false
  }
}

export function recordOpenAIConsent(baseUrl: string): void {
  const host = hostOf(baseUrl)
  if (!host) return
  try {
    const raw = localStorage.getItem(CONSENT_STORAGE)
    const hosts = raw ? (JSON.parse(raw) as string[]) : []
    if (!hosts.includes(host)) hosts.push(host)
    localStorage.setItem(CONSENT_STORAGE, JSON.stringify(hosts))
  } catch {
    /* storage unavailable — the user will be asked again next time */
  }
}

/** Host shown in the consent modal and error messages. Never the full URL. */
export function openaiHost(baseUrl: string): string {
  return hostOf(baseUrl)
}
