import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

import {
  normalizeBaseUrl,
  getOpenAIKey,
  setOpenAIKey,
  clearOpenAIKey,
  isOpenAIKeyRemembered,
  hasOpenAIConsent,
  recordOpenAIConsent,
  redactSecrets,
  OpenAICompatError,
  describeOpenAIError,
  parseRetryAfter,
  requestJSON,
  parseEmbeddings,
  computeTargetSize,
  embedTexts,
  embedQuery,
  embedImages,
  openaiCacheNamespace,
  probeDimension,
  lookupImageEmbeddingModels,
  toModelInfo,
  type OpenAICompatConfig
} from './openaiCompat'

const cfg = (over: Partial<OpenAICompatConfig> = {}): OpenAICompatConfig => ({
  baseUrl: 'https://api.example.com/v1',
  apiKey: '',
  model: 'test-model',
  maxRetries: 0,
  timeoutMs: 1000,
  ...over
})

/** Minimal Response stand-ins — jsdom's fetch plumbing is not involved here. */
function ok(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    headers: { get: () => null },
    json: async () => body,
    text: async () => JSON.stringify(body)
  } as unknown as Response
}

function fail(status: number, body = '', headers: Record<string, string> = {}): Response {
  return {
    ok: false,
    status,
    headers: { get: (k: string) => headers[k] ?? null },
    json: async () => JSON.parse(body),
    text: async () => body
  } as unknown as Response
}

/** `data` rows for `n` distinct unit vectors. */
function vecs(n: number, dim = 2) {
  return {
    data: Array.from({ length: n }, (_, i) => ({
      index: i,
      embedding: Array.from({ length: dim }, (_, j) => (j === 0 ? i + 1 : 1))
    }))
  }
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  fetchMock = vi.fn()
  vi.stubGlobal('fetch', fetchMock)
  // Full-jitter backoff -> 0ms, so retry paths don't add real delay.
  vi.spyOn(Math, 'random').mockReturnValue(0)
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('normalizeBaseUrl', () => {
  it('infers https for public hosts and appends /v1 only when no path is given', () => {
    expect(normalizeBaseUrl('openrouter.ai/api/v1')).toBe('https://openrouter.ai/api/v1')
    expect(normalizeBaseUrl('api.openai.com')).toBe('https://api.openai.com/v1')
  })

  it('infers http for localhost and private ranges', () => {
    expect(normalizeBaseUrl('localhost:11434')).toBe('http://localhost:11434/v1')
    expect(normalizeBaseUrl('127.0.0.1:8080')).toBe('http://127.0.0.1:8080/v1')
    expect(normalizeBaseUrl('192.168.1.9:1234/v1')).toBe('http://192.168.1.9:1234/v1')
    expect(normalizeBaseUrl('nas.local:3000')).toBe('http://nas.local:3000/v1')
  })

  it('keeps an explicit scheme even when it contradicts the host heuristic', () => {
    expect(normalizeBaseUrl('https://localhost:11434/v1')).toBe('https://localhost:11434/v1')
  })

  it('strips a pasted endpoint suffix', () => {
    expect(normalizeBaseUrl('https://api.openai.com/v1/embeddings')).toBe(
      'https://api.openai.com/v1'
    )
    expect(normalizeBaseUrl('http://localhost:8000/v1/chat/completions')).toBe(
      'http://localhost:8000/v1'
    )
  })

  it('trims whitespace and trailing slashes', () => {
    expect(normalizeBaseUrl('  https://h.co/v1///  ')).toBe('https://h.co/v1')
  })

  it('does not force /v1 onto a provider that mounts elsewhere', () => {
    expect(normalizeBaseUrl('https://h.co/openai/deployments/x')).toBe(
      'https://h.co/openai/deployments/x'
    )
  })

  it('returns empty string for blank or unparseable input', () => {
    expect(normalizeBaseUrl('')).toBe('')
    expect(normalizeBaseUrl('   ')).toBe('')
    expect(normalizeBaseUrl(null)).toBe('')
    expect(normalizeBaseUrl('ftp://h.co')).toBe('')
    expect(normalizeBaseUrl('http://')).toBe('')
  })
})

describe('redactSecrets', () => {
  it('removes the live key wherever it appears', () => {
    expect(redactSecrets('failed with mykey-12345678 twice: mykey-12345678', 'mykey-12345678')).toBe(
      'failed with *** twice: ***'
    )
  })

  it('ignores a key too short to replace safely', () => {
    // Replacing "abc" would shred unrelated prose.
    expect(redactSecrets('abcdef the abc thing', 'abc')).toBe('abcdef the abc thing')
  })

  it('scrubs generic token shapes even without knowing the key', () => {
    expect(redactSecrets('key sk-or-v1-abcdef123456 rejected')).toBe('key sk-*** rejected')
    expect(redactSecrets('Authorization: Bearer abc123def456')).toBe('Authorization: Bearer ***')
    expect(redactSecrets('GET /v1/x?api_key=secretvalue&z=1')).toBe('GET /v1/x?api_key=***&z=1')
  })
})

describe('OpenAICompatError', () => {
  it('redacts the message and hint at construction', () => {
    const err = new OpenAICompatError('auth', 'rejected token sk-abcdef123456', {
      status: 401,
      hint: 'check sk-abcdef123456',
      host: 'api.example.com',
      key: 'sk-abcdef123456'
    })
    expect(err.message).not.toContain('sk-abcdef123456')
    expect(err.hint).not.toContain('sk-abcdef123456')
    expect(err.status).toBe(401)
    expect(err.retryable).toBe(false)
  })

  it('carries the host only, never a full URL', () => {
    const err = new OpenAICompatError('network', 'boom', { host: 'api.example.com' })
    expect(err.host).toBe('api.example.com')
    expect(describeOpenAIError(err)).toContain('api.example.com')
  })
})

describe('parseRetryAfter', () => {
  it('reads integer seconds', () => {
    expect(parseRetryAfter('2')).toBe(2000)
  })

  it('reads HTTP-date form relative to now', () => {
    const now = Date.parse('2026-01-01T00:00:00Z')
    expect(parseRetryAfter('Thu, 01 Jan 2026 00:00:05 GMT', now)).toBe(5000)
  })

  it('clamps a hostile value so cancel stays responsive', () => {
    expect(parseRetryAfter('3600')).toBe(20_000)
  })

  it('returns null for absent or bogus values', () => {
    expect(parseRetryAfter(null)).toBeNull()
    expect(parseRetryAfter('soon')).toBeNull()
  })

  it('never returns negative for a date already past', () => {
    const now = Date.parse('2026-01-01T00:00:10Z')
    expect(parseRetryAfter('Thu, 01 Jan 2026 00:00:00 GMT', now)).toBe(0)
  })
})

describe('requestJSON auth header', () => {
  it('omits Authorization entirely when the key is empty', async () => {
    fetchMock.mockResolvedValue(ok({ hi: true }))
    await requestJSON(cfg({ apiKey: '' }), '/embeddings', {})

    const headers = fetchMock.mock.calls[0][1].headers
    expect('Authorization' in headers).toBe(false)
  })

  it('sends a bearer token when the key is set', async () => {
    fetchMock.mockResolvedValue(ok({ hi: true }))
    await requestJSON(cfg({ apiKey: 'sk-abcdef123456' }), '/embeddings', {})

    expect(fetchMock.mock.calls[0][1].headers.Authorization).toBe('Bearer sk-abcdef123456')
  })

  it('never puts the key in the URL', async () => {
    fetchMock.mockResolvedValue(ok({ hi: true }))
    await requestJSON(cfg({ apiKey: 'sk-abcdef123456' }), '/embeddings', {})

    expect(fetchMock.mock.calls[0][0]).toBe('https://api.example.com/v1/embeddings')
  })
})

describe('requestJSON retry policy', () => {
  it('retries 429 and succeeds', async () => {
    fetchMock
      .mockResolvedValueOnce(fail(429, 'slow down'))
      .mockResolvedValueOnce(ok({ ok: 1 }))

    const out = await requestJSON(cfg({ maxRetries: 2 }), '/embeddings', {})
    expect(out).toEqual({ ok: 1 })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('retries 5xx up to maxRetries then throws', async () => {
    fetchMock.mockResolvedValue(fail(503, 'unavailable'))

    await expect(requestJSON(cfg({ maxRetries: 2 }), '/embeddings', {})).rejects.toMatchObject({
      kind: 'server',
      status: 503
    })
    expect(fetchMock).toHaveBeenCalledTimes(3) // initial + 2 retries
  })

  it('never retries 401', async () => {
    fetchMock.mockResolvedValue(fail(401, 'bad key'))

    await expect(requestJSON(cfg({ maxRetries: 3 }), '/embeddings', {})).rejects.toMatchObject({
      kind: 'auth',
      retryable: false
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it.each([
    [400, 'bad-request'],
    [404, 'not-found'],
    [413, 'payload-too-large']
  ])('never retries %i', async (status, kind) => {
    fetchMock.mockResolvedValue(fail(status, 'nope'))

    await expect(requestJSON(cfg({ maxRetries: 3 }), '/embeddings', {})).rejects.toMatchObject({
      kind
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('retries a network TypeError', async () => {
    fetchMock
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(ok({ ok: 1 }))

    await expect(requestJSON(cfg({ maxRetries: 1 }), '/embeddings', {})).resolves.toEqual({ ok: 1 })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('honours Retry-After instead of backoff', async () => {
    fetchMock
      .mockResolvedValueOnce(fail(429, 'slow', { 'Retry-After': '1' }))
      .mockResolvedValueOnce(ok({ ok: 1 }))

    const started = Date.now()
    await requestJSON(cfg({ maxRetries: 1 }), '/embeddings', {})
    // Backoff is stubbed to 0, so any real wait can only come from Retry-After.
    expect(Date.now() - started).toBeGreaterThanOrEqual(900)
  })

  it('redacts a provider error body that echoes the key back', async () => {
    fetchMock.mockResolvedValue(fail(401, 'invalid key sk-abcdef123456 for org'))

    const err = (await requestJSON(cfg({ apiKey: 'sk-abcdef123456' }), '/embeddings', {}).catch(
      (e: unknown) => e
    )) as OpenAICompatError
    expect(err.message).not.toContain('sk-abcdef123456')
    expect(err.message).toContain('***')
  })

  it('reports an upstream key rejection as auth, not as a bad request', async () => {
    // Real shape from OpenRouter with a stale Google BYOK credential: a 400
    // whose body is Google's own auth failure. Classifying it by status alone
    // blames the input type and sends the user to rewrite a correct request.
    const body = JSON.stringify({
      error: {
        message:
          'HTTP 400: {"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}'
      }
    })
    fetchMock.mockResolvedValue(fail(400, body))

    const err = (await requestJSON(cfg({ maxRetries: 3 }), '/embeddings', {}).catch(
      (e: unknown) => e
    )) as OpenAICompatError
    expect(err.kind).toBe('auth')
    expect(err.retryable).toBe(false)
    expect(err.hint).toMatch(/upstream provider rejected/i)
    expect(err.hint).not.toMatch(/input type/i)
    expect(fetchMock).toHaveBeenCalledTimes(1) // still not retried
  })

  it('leaves an ordinary 400 classified as a bad request', async () => {
    fetchMock.mockResolvedValue(fail(400, 'model does not support image input'))

    const err = (await requestJSON(cfg(), '/embeddings', {}).catch(
      (e: unknown) => e
    )) as OpenAICompatError
    expect(err.kind).toBe('bad-request')
    expect(err.hint).toMatch(/input type/i)
  })

  it('reports a non-JSON body as a parse error', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => {
        throw new SyntaxError('Unexpected token <')
      },
      text: async () => '<html>'
    } as unknown as Response)

    await expect(requestJSON(cfg(), '/embeddings', {})).rejects.toMatchObject({ kind: 'parse' })
  })
})

describe('requestJSON cancellation', () => {
  it('rejects immediately when the signal is already aborted', async () => {
    const ctrl = new AbortController()
    ctrl.abort()

    await expect(requestJSON(cfg(), '/embeddings', {}, ctrl.signal)).rejects.toMatchObject({
      kind: 'aborted'
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('reports a user abort as aborted, not as a network failure', async () => {
    const ctrl = new AbortController()
    fetchMock.mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        })
    )

    const p = requestJSON(cfg({ maxRetries: 3 }), '/embeddings', {}, ctrl.signal)
    ctrl.abort()

    await expect(p).rejects.toMatchObject({ kind: 'aborted' })
    expect(fetchMock).toHaveBeenCalledTimes(1) // aborted, not retried
  })

  it('times out a hanging request', async () => {
    fetchMock.mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        })
    )

    await expect(
      requestJSON(cfg({ timeoutMs: 20, maxRetries: 0 }), '/embeddings', {})
    ).rejects.toMatchObject({ kind: 'timeout', retryable: true })
  })

  it('cancels during backoff instead of waiting it out', async () => {
    const ctrl = new AbortController()
    fetchMock.mockResolvedValue(fail(429, 'slow', { 'Retry-After': '20' }))

    const p = requestJSON(cfg({ maxRetries: 3 }), '/embeddings', {}, ctrl.signal)
    await Promise.resolve()
    setTimeout(() => ctrl.abort(), 10)

    const started = Date.now()
    await expect(p).rejects.toMatchObject({ kind: 'aborted' })
    expect(Date.now() - started).toBeLessThan(1000)
  })
})

describe('parseEmbeddings', () => {
  it('returns L2-normalized vectors', () => {
    const out = parseEmbeddings({ data: [{ index: 0, embedding: [3, 4] }] }, 1)
    expect(out[0][0]).toBeCloseTo(0.6, 5)
    expect(out[0][1]).toBeCloseTo(0.8, 5)
  })

  it('sorts by index so vectors are not paired with the wrong file', () => {
    const out = parseEmbeddings(
      {
        data: [
          { index: 2, embedding: [0, 0, 1] },
          { index: 0, embedding: [1, 0, 0] },
          { index: 1, embedding: [0, 1, 0] }
        ]
      },
      3
    )
    expect(out[0][0]).toBeCloseTo(1, 5)
    expect(out[1][1]).toBeCloseTo(1, 5)
    expect(out[2][2]).toBeCloseTo(1, 5)
  })

  it('falls back to array order when no index is present', () => {
    const out = parseEmbeddings({ data: [{ embedding: [1, 0] }, { embedding: [0, 1] }] }, 2)
    expect(out[0][0]).toBeCloseTo(1, 5)
    expect(out[1][1]).toBeCloseTo(1, 5)
  })

  it("accepts llama.cpp's single bare embedding", () => {
    const out = parseEmbeddings({ embedding: [3, 4] }, 1)
    expect(out).toHaveLength(1)
    expect(out[0][0]).toBeCloseTo(0.6, 5)
  })

  it('throws when the count does not match the request', () => {
    expect(() => parseEmbeddings({ data: [{ index: 0, embedding: [1, 0] }] }, 2)).toThrow(
      OpenAICompatError
    )
  })

  it('throws on a response with no data array', () => {
    expect(() => parseEmbeddings({} as never, 1)).toThrow(/no .data. array/)
  })

  it('throws when an entry has no numeric embedding', () => {
    expect(() => parseEmbeddings({ data: [{ index: 0 }] }, 1)).toThrow(OpenAICompatError)
  })
})

describe('computeTargetSize', () => {
  it('leaves an image narrower than the bound untouched', () => {
    expect(computeTargetSize(200, 100, 384)).toEqual({ width: 200, height: 100 })
  })

  it('scales width down and preserves aspect ratio', () => {
    expect(computeTargetSize(768, 512, 384)).toEqual({ width: 384, height: 256 })
  })

  it('never rounds a dimension down to zero', () => {
    expect(computeTargetSize(4000, 3, 384)).toEqual({ width: 384, height: 1 })
  })

  it('returns zeroes for degenerate input rather than NaN', () => {
    expect(computeTargetSize(0, 100, 384)).toEqual({ width: 0, height: 0 })
    expect(computeTargetSize(100, 100, 0)).toEqual({ width: 0, height: 0 })
  })
})

describe('embedTexts', () => {
  it('returns [] without touching the network for an empty list', async () => {
    await expect(embedTexts(cfg(), [])).resolves.toEqual([])
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('sends the model and input array', async () => {
    fetchMock.mockResolvedValue(ok(vecs(2)))
    await embedTexts(cfg(), ['a', 'b'])

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body).toEqual({ model: 'test-model', input: ['a', 'b'] })
  })

  it('splits into batches and preserves overall order', async () => {
    fetchMock
      .mockResolvedValueOnce(ok({ data: [{ index: 0, embedding: [1, 0] }] }))
      .mockResolvedValueOnce(ok({ data: [{ index: 0, embedding: [0, 1] }] }))

    const out = await embedTexts(cfg({ batchSize: 1 }), ['a', 'b'])
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(out).toHaveLength(2)
    expect(out[0][0]).toBeCloseTo(1, 5)
    expect(out[1][1]).toBeCloseTo(1, 5)
  })
})

describe('embedQuery and probeDimension', () => {
  it('returns a single vector', async () => {
    fetchMock.mockResolvedValue(ok({ data: [{ index: 0, embedding: [3, 4] }] }))
    const vec = await embedQuery(cfg(), 'sunset over water')
    expect(vec[0]).toBeCloseTo(0.6, 5)
  })

  it('reports the endpoint dimension', async () => {
    fetchMock.mockResolvedValue(ok({ data: [{ index: 0, embedding: new Array(1536).fill(1) }] }))
    await expect(probeDimension(cfg())).resolves.toBe(1536)
  })
})

describe('embedImages', () => {
  it('uses the openai-multimodal wire format', async () => {
    fetchMock.mockResolvedValue(ok(vecs(1)))
    await embedImages(cfg(), ['data:image/jpeg;base64,AAAA'])

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body).toEqual({
      model: 'test-model',
      input: [
        {
          content: [{ type: 'image_url', image_url: { url: 'data:image/jpeg;base64,AAAA' } }]
        }
      ]
    })
  })

  it('returns [] without touching the network for an empty list', async () => {
    await expect(embedImages(cfg(), [])).resolves.toEqual([])
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('lookupImageEmbeddingModels', () => {
  // Shape taken from OpenRouter's live catalogue.
  const geminiEmbed = {
    id: 'google/gemini-embedding-2',
    name: 'Google: Gemini Embedding 2',
    architecture: {
      modality: 'text+image+file+audio+video->embeddings',
      input_modalities: ['text', 'image', 'file', 'audio', 'video'],
      output_modalities: ['embeddings']
    }
  }
  const textEmbed = {
    id: 'openai/text-embedding-3-small',
    architecture: { input_modalities: ['text'], output_modalities: ['embeddings'] }
  }
  const visionChat = {
    id: 'anthropic/claude-sonnet-4',
    architecture: { input_modalities: ['text', 'image'], output_modalities: ['text'] }
  }

  it('asks for embedding models explicitly, since /models defaults to text only', async () => {
    fetchMock.mockResolvedValue(ok({ data: [geminiEmbed] }))
    await lookupImageEmbeddingModels(cfg())

    expect(fetchMock.mock.calls[0][0]).toContain('/models?output_modalities=embeddings')
    expect(fetchMock.mock.calls[0][1].method).toBe('GET')
  })

  it('sends no body on the GET, which fetch would reject', async () => {
    fetchMock.mockResolvedValue(ok({ data: [geminiEmbed] }))
    await lookupImageEmbeddingModels(cfg())

    expect(fetchMock.mock.calls[0][1].body).toBeUndefined()
  })

  it('keeps only models that take images AND return embeddings', async () => {
    fetchMock.mockResolvedValue(ok({ data: [geminiEmbed, textEmbed, visionChat] }))
    const found = await lookupImageEmbeddingModels(cfg())

    expect(found.imageCapable.map((m) => m.id)).toEqual(['google/gemini-embedding-2'])
    expect(found.all).toHaveLength(3)
    expect(found.hasModalityMetadata).toBe(true)
  })

  it('retries unfiltered when the modality filter yields nothing', async () => {
    // A server that rejects the unknown query param by returning an empty list
    // must not be reported as having no models.
    fetchMock
      .mockResolvedValueOnce(ok({ data: [] }))
      .mockResolvedValueOnce(ok({ data: [geminiEmbed] }))

    const found = await lookupImageEmbeddingModels(cfg())
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toMatch(/\/models$/)
    expect(found.imageCapable).toHaveLength(1)
  })

  it('flags missing metadata rather than claiming there are no image models', async () => {
    // Ollama / LM Studio / vLLM return ids only.
    fetchMock.mockResolvedValue(
      ok({ data: [{ id: 'nomic-embed-text' }, { id: 'llava' }] })
    )
    const found = await lookupImageEmbeddingModels(cfg())

    expect(found.hasModalityMetadata).toBe(false)
    expect(found.imageCapable).toEqual([])
    expect(found.all.map((m) => m.id)).toEqual(['nomic-embed-text', 'llava'])
  })

  it('accepts a `models` array as well as `data`', async () => {
    fetchMock.mockResolvedValueOnce(ok({ models: [geminiEmbed] }))
    const found = await lookupImageEmbeddingModels(cfg())
    expect(found.all.map((m) => m.id)).toEqual(['google/gemini-embedding-2'])
  })

  it('drops entries with no id instead of offering blanks', async () => {
    fetchMock.mockResolvedValue(ok({ data: [geminiEmbed, {}, { name: 'nameless' }] }))
    const found = await lookupImageEmbeddingModels(cfg())
    expect(found.all).toHaveLength(1)
  })

  it('surfaces an auth failure rather than reporting an empty catalogue', async () => {
    fetchMock.mockResolvedValue(fail(401, 'no credentials'))
    await expect(lookupImageEmbeddingModels(cfg())).rejects.toMatchObject({ kind: 'auth' })
  })
})

describe('toModelInfo', () => {
  it('is case-insensitive about modality names', () => {
    const info = toModelInfo({
      id: 'x',
      architecture: { input_modalities: ['Image'], output_modalities: ['Embeddings'] }
    })
    expect(info.imageEmbedding).toBe(true)
  })

  it('does not treat a vision chat model as an embedder', () => {
    const info = toModelInfo({
      id: 'x',
      architecture: { input_modalities: ['text', 'image'], output_modalities: ['text'] }
    })
    expect(info.imageEmbedding).toBe(false)
  })

  it('does not treat a text-only embedder as image-capable', () => {
    const info = toModelInfo({
      id: 'x',
      architecture: { input_modalities: ['text'], output_modalities: ['embeddings'] }
    })
    expect(info.imageEmbedding).toBe(false)
  })

  it('reports false, not a guess, when metadata is absent', () => {
    expect(toModelInfo({ id: 'x' }).imageEmbedding).toBe(false)
  })
})

describe('openaiCacheNamespace', () => {
  it('changes when host or model changes', () => {
    const base = openaiCacheNamespace(cfg())
    expect(openaiCacheNamespace(cfg({ model: 'other' }))).not.toBe(base)
    expect(openaiCacheNamespace(cfg({ baseUrl: 'https://other.example.com/v1' }))).not.toBe(base)
  })

  it('is stable across keys, so rotating a key does not orphan the cache', () => {
    expect(openaiCacheNamespace(cfg({ apiKey: 'sk-aaaaaaaa' }))).toBe(
      openaiCacheNamespace(cfg({ apiKey: 'sk-bbbbbbbb' }))
    )
  })

  it('never embeds the API key in a persisted cache key', () => {
    const ns = openaiCacheNamespace(cfg({ apiKey: 'sk-abcdef123456' }))
    expect(ns).not.toContain('sk-abcdef123456')
  })

  it('is prefix-shaped so it matches the other cache namespaces', () => {
    expect(openaiCacheNamespace(cfg())).toMatch(/^@openai:.*\/$/)
  })
})

describe('key storage', () => {
  beforeEach(() => {
    localStorage.clear()
    sessionStorage.clear()
  })

  it('keeps the key out of mc_settings, which sentry and saveSettings both touch', () => {
    localStorage.setItem('mc_settings', JSON.stringify({ modelVariant: 'openai' }))
    setOpenAIKey('sk-abcdef123456', true)

    expect(localStorage.getItem('mc_settings')).not.toContain('sk-abcdef123456')
  })

  it('persists to localStorage when remembered', () => {
    setOpenAIKey('sk-abcdef123456', true)
    expect(getOpenAIKey()).toBe('sk-abcdef123456')
    expect(isOpenAIKeyRemembered()).toBe(true)
  })

  it('uses sessionStorage when not remembered, so it dies with the tab', () => {
    setOpenAIKey('sk-abcdef123456', false)
    expect(getOpenAIKey()).toBe('sk-abcdef123456')
    expect(isOpenAIKeyRemembered()).toBe(false)
    expect(localStorage.getItem('mc_openai_key')).toBeNull()
    expect(sessionStorage.getItem('mc_openai_key')).toBe('sk-abcdef123456')
  })

  it('leaves no copy behind when "remember" is switched off', () => {
    setOpenAIKey('sk-abcdef123456', true)
    setOpenAIKey('sk-abcdef123456', false)

    expect(localStorage.getItem('mc_openai_key')).toBeNull()
    expect(isOpenAIKeyRemembered()).toBe(false)
  })

  it('clears from both storages', () => {
    setOpenAIKey('sk-abcdef123456', true)
    clearOpenAIKey()
    expect(getOpenAIKey()).toBe('')
    expect(localStorage.getItem('mc_openai_key')).toBeNull()
    expect(sessionStorage.getItem('mc_openai_key')).toBeNull()
  })
})

describe('upload consent', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('is withheld until granted', () => {
    expect(hasOpenAIConsent('https://api.example.com/v1')).toBe(false)
  })

  it('is remembered per host once granted', () => {
    recordOpenAIConsent('https://api.example.com/v1')
    expect(hasOpenAIConsent('https://api.example.com/v1')).toBe(true)
  })

  it('does not carry over to a different host', () => {
    // Agreeing to upload to your own LAN box is not agreement to upload to a
    // third-party API — this is the whole point of keying consent by host.
    recordOpenAIConsent('http://localhost:11434/v1')
    expect(hasOpenAIConsent('https://openrouter.ai/api/v1')).toBe(false)
  })

  it('ignores the path, so /v1 vs /v1/embeddings is the same host', () => {
    recordOpenAIConsent('https://api.example.com/v1')
    expect(hasOpenAIConsent('https://api.example.com/openai/deployments')).toBe(true)
  })

  it('accumulates hosts rather than replacing them', () => {
    recordOpenAIConsent('http://localhost:11434/v1')
    recordOpenAIConsent('https://openrouter.ai/api/v1')
    expect(hasOpenAIConsent('http://localhost:11434/v1')).toBe(true)
    expect(hasOpenAIConsent('https://openrouter.ai/api/v1')).toBe(true)
  })

  it('treats unparseable input as not consented', () => {
    expect(hasOpenAIConsent('')).toBe(false)
  })
})
