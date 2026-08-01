# Plan: Remote OpenAI-Compatible Inference

Implementation plan for [ADR-0002](docs/adr/0002-openai-compatible-remote-inference.md), which decides to
add a `openai` model variant that runs inference against any OpenAI-compatible endpoint — localhost or
internet — instead of in the browser. Written 2026-08-01.
This document is a plan only — no changes are implemented by the PR that introduces it.

Milestones are `M0`…`M7`, in dependency order. `M0` is a pure refactor that stands alone, and `M1` is a
self-contained module with no app wiring.

## Context

**Current problem:**

- Inference is browser-only. The default `sapiens2-fp16` costs a 229 MB download before the first photo
  is processed (`src/app.ts:206`, `src/sapiens2.ts:17-34`), and throughput is bounded by the local GPU —
  or by WASM when WebGPU is missing (`src/app.ts:1175-1181`, `src/sapiens2.ts:218-224`).
- `customModelHost` (`src/types.ts:61`, `src/app.ts:2896-2901`) looks like it addresses this but only
  redirects **model file downloads** to a HuggingFace-compatible mirror. Inference still runs locally.
- Models that would most improve clustering and search — purpose-built multimodal retrieval models, video
  LLMs — cannot run in a tab at all.
- Videos are represented by a single frame (`src/app.ts:504-555`), the same limitation
  `VIDEO_LM_PLAN.md` documents.

**Desired outcome:**

- A user can point the app at `http://localhost:11434/v1`, `https://openrouter.ai/api/v1`, or anything
  else, supply a key, and start embedding within seconds.
- Images are embedded directly where the provider supports it; VLMs and video LLMs are supported through
  a second pipeline.
- Only downscaled JPEG thumbnails ever leave the device, and only after explicit consent.
- The API key cannot reach a log line, an error report, or a URL.

## Implementation approach

### Libraries to add

**None.** `p-limit` is already a dependency and is already used for exactly this shape of concurrency
control (`src/app.ts:678`). Everything else is `fetch`, `createImageBitmap` and `OffscreenCanvas`.

### The two pipelines

| Pipeline | Setting | Request | Produces | Works with |
| --- | --- | --- | --- | --- |
| **A** `direct` | `openai.pipeline = 'direct'` | `POST {baseUrl}/embeddings` | vector | OpenRouter, vLLM, Jina, Infinity |
| **B** `vlm` | `openai.pipeline = 'vlm'` | `POST {baseUrl}/chat/completions` → then embed the caption | caption + vector | every chat provider, incl. Ollama; the only route for video LLMs |

Pipeline B's second stage is itself configurable: remote `{baseUrl}/embeddings`, or the local
`nomic-embed-text` that the `chrome-ai` path already loads (`src/app.ts:1054-1082`). Local is the safer
default — it works against providers with no embeddings endpoint at all, and needs no second round-trip
per image.

Both sit behind one interface so `embedAll()` gains one branch, not two:

```typescript
export interface RemoteEmbedder {
  embedMedia(items: MediaInput[], signal?: AbortSignal): Promise<PromiseSettledResult<EmbedResult>[]>
  embedQuery(text: string, signal?: AbortSignal): Promise<Float32Array>
  readonly dim: number        // probed at load, before any cache key is computed
  readonly namespace: string  // IndexedDB cache namespace
}

export interface EmbedResult { vector: Float32Array; caption?: string }  // caption: pipeline B only
export type MediaInput =
  | { kind: 'image'; frames: [ImageBitmap] }
  | { kind: 'video'; frames: ImageBitmap[] }
```

`embedQuery` **must** reach the same endpoint and model as `embedMedia`. See "Vector-space consistency".

### Wire formats

`/v1/chat/completions` with an `image_url` part is universal. Image input to `/v1/embeddings` is not —
see the table in ADR-0002. Model it as a discriminated union in `src/types.ts`:

```typescript
export type EmbedWireFormat =
  | 'openai-multimodal'  // OpenRouter: input: [{ content: [{ type: 'image_url', image_url: { url } }] }]
  | 'chat-messages'      // vLLM:       messages: [{ role: 'user', content: [...] }]
  | 'jina'               // Jina:       input: [{ image: '<b64>' }] / [{ text: '...' }]
  | 'plain-input'        // Infinity:   input: ['<data URI>']
  | 'llamacpp'           // llama.cpp:  { content: 'Image: [img-1]', image_data: [{ id, data }] }
```

`probeWireFormat()` sends one 64×64 solid-colour JPEG through each in order and keeps the first that
returns a numeric vector, persisting the winner. This is what the **Test connection** button runs, and it
reports both the winning format and the resulting dimension. Default `'auto'`; a manual override exists
for endpoints where probing costs money.

Response parsing handles `{ data: [{ embedding, index }] }` and llama.cpp's `{ embedding: [...] }`.
**Sort by `index`** — order is not guaranteed by the spec, and getting it wrong silently assigns the
wrong vector to a file, which is invisible until someone notices the map is nonsense.

### Files to modify

1. **`src/types.ts`** — extend `ModelVariant` with `'openai'`; add `OpenAISettings`, `EmbedWireFormat`,
   `MediaInput`, `EmbedResult`, `RemoteEmbedder`, `OpenAICompatConfig`; add the new `DOMElements` refs.
   Per `AGENT.md`, interfaces live here.
2. **`src/app.ts`** — new branch in `loadModelOnce()` (`:1013`); new branch in `embedAll()` (`:1433`)
   alongside `isSapiens2` / `isChromeAI` (`:1442-1443`); `currentCachePrefix()` (`:354-360`) gains the
   remote namespace; `embedText()` (`:623-634`) is rewritten onto a shared funnel; four
   `new Float32Array(768)` sites become `zeroVector()`; lazy modal captions (`:2500-2553`) become
   variant-aware; settings listeners (`:2760-2900`).
3. **`index.html`** — one `<option>` in `#model-select` (`:807-812`); a new `#openai-setting` block after
   `#chrome-ai-prompt-setting` (`:823-830`); consent `<dialog>`; corrected privacy copy at `:745` and
   `:781`, and in the `<meta name="description">` at `:7`/`:10`.
4. **`src/compute.ts`** — `searchByCosine()` (`:266-291`) reports a dimension-mismatch count instead of
   silently clamping.
5. **`src/modelFallback.ts`** — `isDownloadError()` (`:111-124`) must not claim API errors.
6. **`src/sentry.ts`** — `beforeSend` / `beforeBreadcrumb` redaction.
7. **`README.md`**, **`AGENT.md`** — privacy claim, new mode, new rules.

### New files

- `src/openaiCompat.ts` — the whole HTTP surface. No DOM, no app imports; only `p-limit` and
  `./embeddings`. Mirrors the testability of `src/modelFallback.ts`.
- `src/openaiCompat.test.ts` — unit tests against a stubbed `fetch`.
- `src/captions.ts` — backend-namespaced caption storage (see M0).

### Data flow

```text
                       collectImages() → embedAll()                    src/app.ts:1433
                                            │
                        ┌───────────────────┴──────────────────┐
                        ↓                                      ↓
                  image file                             video file
                        │                                      │
                        │                        extractVideoFrames(file, n)   [M6]
                        ↓                                      ↓
                  1 ImageBitmap                          n ImageBitmaps
                        └───────────────────┬──────────────────┘
                                            ↓
                            imageToDataURL(384 px, JPEG q0.8)          [M1]
                                            ↓
                        ┌───────────────────┴──────────────────┐
                        ↓                                      ↓
              pipeline A: direct                     pipeline B: vlm
              POST /embeddings                       POST /chat/completions
                        │                                      ↓
                        │                                 caption text
                        │                                      ↓
                        │                    ┌─────────────────┴──────────────┐
                        │                    ↓                                ↓
                        │            POST /embeddings              local nomic-embed-text
                        │             (remote text)                  (src/app.ts:1054-1082)
                        └───────────────────┬──────────────────────────────────┘
                                            ↓
                                     l2normalize()                src/embeddings.ts:10
                                            ↓
                              IndexedDB `embeddings`  @openai/<hash>/
                                            ↓
              runProjection() → kmeansAsync() → canvas       (unchanged, src/compute.ts)
```

## M0 — Refactor only, no feature

Nine defects that exist today and become severe with a network backend. This milestone changes no
behaviour and ships on its own.

| # | Issue | Location |
| --- | --- | --- |
| B1 | `isDownloadError()` matches `unauthorized`, `failed to fetch` and bare `403\|404\|429\|500`, so an API 401 opens the *HuggingFace model-upload fallback modal*. Check the API error type first in `loadModel()`. | `src/modelFallback.ts:111-124`, `src/app.ts:1346-1351` |
| B2 | Cache-prefix logic is duplicated three times and can diverge. A stale prefix on resume loads wrong-dimension vectors silently. Collapse onto `currentCachePrefix()`. | `src/app.ts:354-360`, `:1450-1456`, `:3159-3166` |
| B3 | `searchByCosine()` clamps to the shorter vector, so a dimension mismatch produces a plausible ranking rather than an error. Report a mismatch count and surface it. | `src/compute.ts:280-285` |
| B4 | `embedText()` returns `Float32Array.from(output.data)` with **no L2 normalisation**, relying on Transformers.js `normalize: true`. `searchByCosine()` is a raw dot product; remote responses are not guaranteed unit-norm. | `src/app.ts:633` |
| B5 | The `chrome-ai` batch uses `Promise.all`, so one rejection zero-fills the whole batch. With a remote API a single 429 would take its neighbours with it. Use `allSettled` + per-item fallback. | `src/app.ts:1575-1579`, `:1642-1647` |
| B6 | A zero vector scores exactly `0` in cosine search — **above** every genuinely dissimilar item, which scores negative. Failed embeddings therefore rank high. Track failed indices and exclude them. | `src/app.ts:1536`, `:1622`, `:1645`, `:3182` |
| B7 | Caption keys are not namespaced by backend, and use the folder-relative `f.name` while the embedding cache uses the basename (`makeCacheKey`, `src/embeddings.ts:117`). Captions therefore miss in exactly the cases embeddings hit. Extract `src/captions.ts` keyed `@caption/<backend>/<basename>:<size>:<lastModified>`, with read-through migration from both legacy forms. | `src/app.ts:1551-1553`, `:1585`, `:2491`, `:2537-2540` |
| B8 | `loadModel()`'s early-return guard lists the three existing backends; a fourth must be added or a second load re-probes the endpoint and costs a real API call. | `src/app.ts:1333`, `:2831` |
| B9 | The text-model loading block is copy-pasted three times. Extract `loadTextExtractor()` before adding a fourth caller. | `src/app.ts:1054-1082`, `:1187-1213`, `:2794-2820` |

Also in M0: introduce `activeEmbeddingDim` + `zeroVector()` and replace the four hard-coded
`new Float32Array(768)` sites. For every existing variant the value stays 768, so this is provably inert.

> Note on `computeOptimalBatchSize()` (`src/hardware.ts:26-27`): it models ViT activation memory
> (`197 * 768 * 12 * 4 * 4`). That is meaningless for a remote backend and must not drive remote
> concurrency — default to 4, or 8 for a localhost base URL.

## M1 — `src/openaiCompat.ts`, standalone

Pure module, no app wiring, fully unit-tested before anything depends on it.

**`normalizeBaseUrl(input)`** — trim; strip trailing slashes (same shape as `normalizeHost`,
`src/modelFallback.ts:28-30`); strip a pasted `/embeddings` or `/chat/completions` suffix; infer the
scheme (`http` for `localhost`, `127.0.0.1`, `[::1]`, `*.local` and private ranges, `https` otherwise);
append `/v1` only when the path is empty. So `openrouter.ai/api/v1`, `localhost:11434` and
`localhost:1234/v1` all resolve correctly, and unparseable input returns `''` rather than a URL that will
fail confusingly later.

**`requestJSON<T>(cfg, path, init, signal, retry)`** — the single HTTP chokepoint.

- `Authorization: Bearer` header only. **Omitted entirely when the key is empty** — local servers reject
  a header whose value is `Bearer` followed by nothing, and localhost servers need no key.
- Retries 408 / 429 / 5xx and network `TypeError`. Never retries 400 / 401 / 403 / 404 / 413 / 422.
- Honours `Retry-After` in both integer-seconds and HTTP-date form, clamped to a ceiling so a hostile
  `Retry-After: 3600` cannot hang the UI.
- Otherwise full-jitter exponential backoff.
- **The backoff sleep is itself abortable.** A plain `setTimeout` would leave Cancel unresponsive for up
  to the maximum delay.
- Per-request timeout via `AbortSignal.any` with a manual `AbortController` fallback — jsdom does not
  have it, so the tests need the fallback path anyway.

**`imageToDataURL(source, maxWidth, quality)`** — `createImageBitmap(blob, { resizeWidth })` so a large
original is never decoded at full size, then `OffscreenCanvas` → `convertToBlob({ type: 'image/jpeg' })`.
Defaults 384 px / 0.8. JPEG rather than PNG (5–10× smaller); alpha flattened onto white. Ownership rule:
close bitmaps this function created, never one the caller owns — `AGENT.md` is strict here, and
`src/app.ts:1491` and `:1514` carry the comments explaining why.

Extract **`computeTargetSize(w, h, maxWidth)`** as a separate pure function. jsdom has neither
`OffscreenCanvas` nor `createImageBitmap`, so this keeps the arithmetic exhaustively testable without
brittle canvas stubs.

**`OpenAICompatError`** — carries `kind`, `status`, `retryable`, `hint`, and `host` (host only, never the
full URL, never the config object). The message is passed through `redactSecrets()` **at construction**,
so an un-redacted string cannot escape by any route.

**`redactSecrets(input, key?)`** — removes the live key, and independently scrubs generic token shapes
(`sk-…`, `sk-or-v1-…`, `Bearer …`, `?api_key=…`) so a provider echoing a key back in an error body is
also caught.

Plus `embedMediaDirect`, `describeMedia`, `embedTexts`, `listModels`, `probeWireFormat`,
`describeOpenAIError`, `openaiCacheNamespace`. `pLimit(cfg.concurrency)` is created once and applied
*inside* the batch functions, so callers can pass an array of any size without stampeding the endpoint.

## M2 — Types, settings, key storage, consent

`ModelVariant` gains `'openai'`; `Settings` gains an `openai: OpenAISettings` object. The settings merge
at `src/app.ts:225-227` is shallow (`{ ...DEFAULT_SETTINGS, ...parsed }`), so a partially-written nested
object would not be filled in — add an explicit `openai: { ...DEFAULT_SETTINGS.openai, ...parsed.openai }`
fix-up next to the existing `modelVariant` migration (`:213-224`).

**The key is not in `Settings`.** It lives in `localStorage['mc_openai_key']` (or `sessionStorage` when
"remember" is off), reached only through `getOpenAIKey()` / `setOpenAIKey()` / `clearOpenAIKey()` in
`src/openaiCompat.ts`. Rationale in ADR-0002; the short version is that `src/sentry.ts:7-8` already reads
`mc_settings`, and `saveSettings()` (`src/app.ts:2678-2680`) writes the whole blob on nearly every UI
interaction.

**Consent modal** — a `<dialog>` naming the destination host, shown once per base URL, remembered in
`localStorage['mc_openai_consent']`. Gated in `loadModelOnce()`. The `modelFallbackModal` markup and its
promise-wrapped handlers are the pattern to copy.

## M3 — Settings UI and Test connection

A new `#openai-setting` block in `index.html`, shown by generalising `updateChromeAIPromptVisibility()`
(`src/app.ts:2716-2719`) into `updateVariantSettingsVisibility()`.

| Field | Notes |
| --- | --- |
| Preset | OpenRouter / Venice / Ollama / LM Studio / llama.cpp / vLLM / Jina / Infinity / Custom. Prefills base URL and suggested models. **Disables pipeline A for Ollama with an explanatory hint** rather than letting the user find out via HTTP 400. |
| Base URL | `change` handler runs `normalizeBaseUrl` and writes the normalised value back, mirroring `customModelHost` (`:2896-2901`). |
| API key | `type="password"`, `autocomplete="off"`. Never bound to `state`. `change`, not `input`. |
| Remember key | Unchecked ⇒ `sessionStorage`. Plus a clear-key button. |
| Pipeline | `direct` / `vlm`. |
| Models | `<input list=…>` with `<datalist>` populated from `/models` after a successful test. |
| Embedding source | remote / local — pipeline B only. |
| Frames per video, concurrency, max image width, wire format | With cost hints. |
| Describe prompt | **Reuse the existing `#chrome-ai-prompt` textarea** — relabel its container and show it for both `chrome-ai` and `openai`, keeping the `mc_chrome_ai_prompt` storage key. One prompt, one control. Its hint currently reads "Changes apply to new embeddings only", which is accurate for `chrome-ai` and **wrong** for `openai`, where the prompt is namespaced: rewrite it to say editing the prompt re-embeds, and that reverting it restores the previous cache. |
| Test connection | Runs `/models`, then `probeWireFormat`, and reports the winning format and dimension. |

Changing **any** vector-affecting setting after a load must trigger the same page reload as
`#model-select` (`:2827-2834`), because the cache namespace changes underneath `state.vectors`. That is
every input in the namespace list below: base URL, pipeline, either model field, **embedding source**,
wire format, describe prompt, max image width, and frames per video.

The embedding-source switch (remote `/embeddings` ↔ local `nomic-embed-text`) is the one most easily
missed and the most dangerous, because the dimension guard cannot catch it: `nomic-embed-text` is 768-d
and so are many remote models, so flipping the source after a run leaves `state.vectors` in the old space
while queries are embedded in the new one, at matching width. Search then returns plausible, meaningless
rankings with nothing to trip on. Treat it as reload-required, not as a live toggle.

Doing the UI before the inference path means the error messages below are reviewable in isolation.

### Error UX

`describeOpenAIError(err, ctx)` — pure, table-driven, used by both Test connection and the in-run toast
(`showToast`, `src/app.ts:290-299`), rate-limited to one toast per failure *kind* per run.

- **401** — key rejected; check for whitespace and that it belongs to this host.
- **403** — credits, model access, or region.
- **404 on `/embeddings`** — this server has no embeddings endpoint; switch to pipeline B with local text
  embeddings. A first-class expected path, not an edge case.
- **400/422 mentioning image or vision** — this model is text-only; pick an image-capable one. (The
  Ollama-in-pipeline-A case that the preset should already have prevented.)
- **429** — retry-after, and lower Concurrency.
- **413** — lower Max image width.
- **`TypeError: Failed to fetch`** — the browser cannot distinguish DNS failure, connection refused and
  CORS rejection; all three are the same opaque error. So **branch on the URL, not the error**:
  - *localhost or private IP from an `https:` page* → Chrome's Local Network Access / Private Network
    Access restriction. Offer: run the app locally over `http://localhost` (`npm run dev`), put the
    server behind HTTPS, or accept the permission prompt. Firefox and Safari differ and may block
    outright — say so rather than promising it works.
  - *remote host* → CORS, with the specific fix per provider: Ollama `OLLAMA_ORIGINS` (and
    `launchctl setenv` on macOS); LM Studio Developer tab → *Enable CORS*; vLLM `--allowed-origins`;
    llama.cpp is permissive by default but a fronting proxy must forward `Authorization` and answer
    `OPTIONS`.
- **AbortError** — silent, matching existing behaviour (`src/app.ts:1345`, `:3212`,
  `src/modelFallback.ts:114`).

## M4 — `loadModelOnce` and `embedAll`

A new branch in `loadModelOnce()` before the `chrome-ai` branch (`src/app.ts:1019`), following the shape
of the chrome-ai unavailable path (`:1023-1030`): validate config → consent gate → build client → probe
wire format and dimension → pipeline B with local embeddings additionally calls `loadTextExtractor()`
(B9) → `state.phase = 'model_ready'`. No large download, so this is near-instant.

`updateDeviceBadge()` (`:305-352`) gains a `Remote · <host>` state.

In `embedAll()`:

- `isOpenAI` flag alongside `:1442-1443`; batch size from `openai.concurrency` at `:1459` — semantically
  a concurrency window here, since `p-limit` does the real limiting inside the client.
- Cache prefix from `currentCachePrefix()` (B2), namespaced on a synchronous FNV-1a hash. Deliberately
  **not** `crypto.subtle.digest`, which is async and would force `currentCachePrefix()` and all its
  callers to become promises.

  **The rule is: every input that changes the resulting vector belongs in the namespace.**

  | Component | Why |
  | --- | --- |
  | full normalized base URL | not just the host — two paths on one host can be different services behind a gateway, and the port distinguishes Ollama from LM Studio |
  | `pipeline` | `direct` and `vlm` produce incomparable vectors |
  | vision/VLM model | in pipeline B the caption, and therefore the vector, depends entirely on it |
  | `embedderId` | `remote:<model>` vs `local:nomic-embed-text-v1.5` |
  | `dim` | providers change dimensions silently, and dim mismatch is the one failure producing garbage rather than an error (B3) |
  | `wireFormat` | different request shapes can reach different model paths on the same server |
  | **describe prompt** (pipeline B) | the caption is the embedding input; a reworded prompt is a different vector |
  | **`maxImageWidth`** / **`jpegQuality`** | changes the pixels the model sees |
  | **`framesPerVideo`** | changes what the model sees for videos |

  The last three are the ones easiest to leave out, and leaving them out is what makes a setting look
  broken: change it, reload, and `readCachedEmbeddings()` serves vectors built from the old input, so
  nothing visibly happens.

  This is a change of position from a first draft that excluded the prompt on the grounds that `chrome-ai`
  already behaves that way (its hint reads "Changes apply to new embeddings only"). That precedent is
  real but it is a bug, not a contract — and it is cheaper to be wrong about locally, where re-embedding
  costs only time.

  Namespacing is **non-destructive**, which is what makes the strict rule affordable: vectors under the
  old namespace stay in IndexedDB, so reverting a setting silently restores its cache rather than
  re-embedding. The cost of a change is therefore one re-embed, not permanent loss — and the cost guard
  below already tells the user how many files that is before anything is sent.

  `framesPerVideo` is the one imprecise entry: it only affects videos, so including it globally
  re-embeds images that did not change. Correctness first; split the namespace per media type only if
  this proves annoying in practice.
- `allSettled` results (B5); rejected entries join the existing `failedInputs` set (`:1563`) so `:1638`
  skips the cache write. That mechanism already exists and is exactly right — reuse it.
- On 429, call `batcher.recordFailure()`. `createAdaptiveBatcher` (`src/batching.ts:48`) already halves
  on failure and grows back 25 % after five successes, which fits rate limits with zero changes. Do
  **not** use `embedBatchAdaptive` (`src/batching.ts:14-35`) — bisecting a rate-limited batch just
  retries faster.
- Thread the abort signal into every request. Without it, cancelling a folder leaves dozens of in-flight
  paid requests running.
- **Cost guard**: count cache misses before the loop and confirm *"This will send N thumbnails to
  `<host>`"*.
- Abort the run early if more than ~25 % fails, rather than producing a map of zero vectors.

### Vector-space consistency

This is a correctness requirement, not a nicety. One funnel both sides call:

```typescript
async function embedTextsForActiveBackend(
  texts: string[], role: 'document' | 'query', signal?: AbortSignal
): Promise<Float32Array[]>
```

- `openai` with remote embeddings → the same endpoint and model as `embedMedia`, **no prefix**.
- everything else → local `textExtractor` with nomic's `search_document:` / `search_query:` prefixes
  (`src/app.ts:1590`, `:627`).

Sending `search_query:` to a non-nomic model is wrong twice over: to `text-embedding-3-small` it is
meaningless tokens that shift the vector, and to a remote `nomic-embed-text` it is only correct if the
indexing side used `search_document:`, which the remote path will not. Isolate the decision in a pure
`applyEmbeddingPrefix(text, role, family)` so it is unit-testable — that helper is where the invariant
actually lives.

`embedText()` (`:623-634`) becomes a two-line wrapper over the funnel, which also fixes B4. Guard
`searchImages()` (`:637`) on the dimension-mismatch count from B3 and report *"cached vectors came from a
different embedding model"* rather than presenting a garbage ranking.

## M5 — Lazy modal captions

The condition at `src/app.ts:2500` (`enableLazyCaption && chromeAIAvailability !== 'unavailable'`) becomes
variant-aware, dispatching to `ChromeAISessionManager` or the remote client. Both already receive an
`AbortController` signal, so remote abort works for free — which matters, because flipping through a
gallery would otherwise fire a paid request per image. Raise the 400 ms debounce to ~700 ms on the remote
path.

## M6 — Multi-frame video

Generalise `extractVideoFrame()` (`src/app.ts:504-555`) to `extractVideoFrames(file, n)`, keeping
`extractVideoFrame` as `extractVideoFrames(file, 1)[0]` so the thumbnail path (`:606`) is untouched.

One `<video>` element, *n* seeks, one teardown. The teardown (`:518-526`) exists because Chrome caps
concurrent `WebMediaPlayer`s at ~75, which is also why `pLimit(4)` wraps it (`:678`); creating *n*
elements would multiply exactly the pressure those guards contain. Keep the limiter at the **file** level,
not the frame level. Timestamps uniform over `(0, duration)` avoiding the endpoints — first and last
frames are frequently black or a title card.

Frames go into one `/chat/completions` request as multiple `image_url` content parts, which every
OpenAI-compatible chat server accepts. A `video_url` part is an optional extra behind the same probe, not
the mechanism.

> This overlaps `VIDEO_LM_PLAN.md` milestone M3, which needs the same generalisation for a local video
> model. Whichever lands first should own the function; the other consumes it unchanged.

## M7 — Secret hygiene and docs

1. **`Authorization` header only, never a query string.** Sentry's default breadcrumbs record fetch
   **URLs**, so a key in `?api_key=` would be uploaded on the next unrelated `console.error`. This is the
   hardest constraint in the feature.
2. **No `console.error` anywhere on this path** — `captureConsoleIntegration({ levels: ['error'] })`
   (`src/sentry.ts:15`) ships it. Use `console.warn`, or nothing. Audit the existing sites reachable from
   here: `src/app.ts:662` (`console.error('Search failed:', err)`) is on the query path and *will* fire.
3. **Harden `src/sentry.ts`** as defence in depth: `beforeBreadcrumb` scrubbing fetch URLs, `beforeSend`
   running `redactSecrets()` over message, exception values and `request.url`.
4. **`.gitleaks.toml`** — use a non-key-shaped placeholder in `index.html` (`"Your provider API key"`).
   A realistic `sk-or-v1-…` example would fail the CI secret scan.
5. **`README.md`** — `:15` ("No media is uploaded. No server involved"), the tagline, and the footer all
   become conditional. Same for `index.html:745`, `:781`, and the `<meta name="description">` at `:7`
   and `:10`. Add a *Remote API backend* section with the provider table, CORS setup, the
   localhost-from-HTTPS caveat, and a cost note. While there: the Project Structure block lists a
   `similarity.ts` that does not exist and omits eight modules that do.
6. **`AGENT.md`** — add Remote AI Mode; a **Secrets** rule (keys only via `getOpenAIKey()`, never in
   `state`, `mc_settings`, a URL, or `console.error`); an **embedding-space invariant** rule (both sides
   through `embedTextsForActiveBackend()`, under that model's own role-prefix scheme — which for nomic
   means the prefixes differ by role; any change of embedder, model, dimension, or of an input that
   alters what gets embedded must change the cache namespace).

## Tests

`src/openaiCompat.test.ts`, Vitest + jsdom, `vi.stubGlobal('fetch', …)`, matching the pure-module style of
`src/modelFallback.test.ts`. No app import, no DOM.

- **`normalizeBaseUrl`** — `openrouter.ai/api/v1` → https; `localhost:11434` → `http://…/v1`;
  `http://localhost:1234/v1/` → unchanged minus the slash; a pasted `/chat/completions` suffix stripped;
  `'  '` and garbage → `''`.
- **Request shape per flavour** — URL, method, `Authorization` present *and absent*, and the exact body
  for each of the five `EmbedWireFormat` values including the `data:image/jpeg;base64,` prefix.
- **`probeWireFormat`** — tries in order, stops at the first success; all-fail throws something useful.
- **Response ordering** — feed `data[]` with `index: [1, 0]` and assert the outputs map back to the right
  inputs. The highest-value test in the file.
- **Normalisation** — un-normalised server vectors come back with `Σx² ≈ 1`.
- **Retry**, with `vi.useFakeTimers()` and a stubbed `Math.random` for deterministic jitter: 429 → 200
  retries once; `Retry-After` in seconds and as an HTTP-date; an over-clamp value fails fast;
  400/401/403/404 make exactly one fetch call each; aborting mid-backoff rejects promptly without a
  further fetch.
- **`describeOpenAIError`** — table-driven; localhost + https mentions Local Network Access, remote
  mentions CORS and the per-provider toggles.
- **`redactSecrets`** — removes the live key and the generic shapes, is idempotent, leaves innocuous text
  alone; and an `OpenAICompatError` built from a body containing a key has a redacted `.message`.
- **`openaiCacheNamespace`** — deterministic for identical input, and differs when *any* single
  component changes: base URL (including path and port), pipeline, vision/VLM model, embedder, dim, wire
  format, describe prompt, `maxImageWidth`, `jpegQuality`, `framesPerVideo`. One assertion per component
  — this is the anti-collision contract, and a component silently missing from the hash is exactly the
  bug the test exists to catch.
- **`computeTargetSize`** — aspect ratio preserved, never upscales, clamps to `maxWidth`.
- **`applyEmbeddingPrefix`** — prefix applied for nomic, absent otherwise.

Plus `src/captions.test.ts` for M0's B7 migration: new key hit, legacy folder-relative key migrated,
legacy basename key migrated, backend namespacing keeps two backends' captions apart.

## Risks

| Risk | Mitigation | What settles it |
| --- | --- | --- |
| OpenRouter `/embeddings` may not accept `data:` URIs (docs show `https://` URLs only) | Pipeline A still works on Jina, vLLM and Infinity; OpenRouter users take pipeline B | One `curl`, before M1 |
| CORS or Local Network Access blocks localhost servers | Actionable per-provider error text; `npm run dev` documented as the friction-free path | Manual test against LM Studio with CORS off |
| Wire-format probe misidentifies a server | Manual override in settings; the probe reports what it chose | Test connection against each preset |
| Per-image cost surprises a user | Cost guard before the run; thumbnails at 384 px; free model documented for testing | Cost guard shown on a 1 000-file folder |
| API key leaks into an error report | Redaction at error construction, `console.warn` only, Sentry `beforeSend`/`beforeBreadcrumb` | Unit tests + a deliberate 401 with the network tab open |
| `Cross-Origin-Embedder-Policy: credentialless` (`public/_headers:3`) interferes with authenticated CORS fetches | COEP applies to `no-cors` responses, so a CORS-mode fetch should pass | Smoke test on the deployed HTTPS build, not only `vite dev` |
| Seven model variants is a lot of UI | Group `#model-select` with `<optgroup>` | Settings modal review |

## Verification

1. `npm run type-check && npm test` — existing suites stay green. `src/boot.test.ts` catches
   `index.html` ↔ `app.ts` binding drift as the new settings are added.
2. `npm run build && npm run preview` — verify against a production bundle, not only `vite dev`.
3. **The decisive test.** Under `npm run dev` (an `http://localhost` origin, which sidesteps Local Network
   Access), configure OpenRouter with `nvidia/llama-nemotron-embed-vl-1b-v2:free` on pipeline A. Confirm
   the probe selects `openai-multimodal`, note the reported dimension, embed a small folder, then run a
   text search. Semantically correct results are the proof that image vectors and the query vector share
   a space — the property the whole design turns on. The model is free, so this costs nothing.
4. Ollama on pipeline B with a VLM — confirms the non-embedding provider path, and that pipeline A is
   disabled with an explanation rather than a raw 400.
5. LM Studio with CORS disabled — must produce the actionable CORS message, not a bare "Failed to fetch".
6. A deliberately wrong API key — must **not** open the HuggingFace fallback modal (B1), and the key must
   not appear in the network tab's request URLs or in any console output.
7. Switch provider mid-session — the page reloads and re-embeds rather than reusing vectors from the old
   namespace. Repeat for **each** namespaced setting, and specifically for the embedding-source switch
   with a 768-d remote model, where the dimension guard cannot help: after flipping the source, search
   must either re-embed or refuse, never return rankings against the old vectors. Then revert one setting
   and confirm its previous cache is served again rather than re-embedded — the non-destructive property
   the strict namespace rule depends on.
8. Cancel mid-run — DevTools → Network shows in-flight requests actually aborting.
9. A video file with a video LLM on pipeline B — multiple frames reach the model in one request, and the
   caption describes change across the clip.
10. Reload → embeddings come from IndexedDB under `@openai/<hash>/`, and switching back to
    `sapiens2-fp16` finds its own cached vectors intact.
11. Deployed HTTPS build — an authenticated CORS fetch succeeds under the existing COEP header.

## Suggested sequencing

| Step | Work | Why this order |
| --- | --- | --- |
| 1 | M0 refactors | Fixes real bugs, changes no behaviour, and every later step depends on them |
| 2 | M1 `openaiCompat.ts` + tests | The bulk of the logic, testable with no app wiring |
| 3 | M2 types, settings, consent | Small, and unblocks the UI |
| 4 | M3 settings UI + Test connection | A user can configure and validate an endpoint before any inference path exists — which makes the error UX reviewable on its own |
| 5 | M4 `loadModelOnce` + `embedAll` **and** the query funnel | Must land together: a remote index with a local query embedder is silently wrong, so the intermediate state is worse than not shipping |
| 6 | M5 lazy captions | Small, independent |
| 7 | M6 multi-frame video | Independent of the transport; coordinate with `VIDEO_LM_PLAN.md` M3 |
| 8 | M7 secret hygiene + docs | Sentry hardening should not wait, but the doc edits want the final feature shape |
