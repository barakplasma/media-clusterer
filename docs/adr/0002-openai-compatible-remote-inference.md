# 0002. Offer remote inference against any OpenAI-compatible endpoint

- **Status:** Proposed
- **Date:** 2026-08-01
- **Supersedes:** nothing

## Context and problem statement

Every embedding the app produces is computed in the browser. There are three backends and no fourth
option (`src/types.ts:41-46`):

| Variant                     | Mechanism                                                                | Download           |
|-----------------------------|--------------------------------------------------------------------------|--------------------|
| `nomic`                     | Transformers.js `image-feature-extraction` (`src/app.ts:1163-1181`)      | 380 MB             |
| `sapiens2-{int8,fp16,fp32}` | hand-rolled onnxruntime-web session (`src/sapiens2.ts`)                  | 116 / 229 / 458 MB |
| `chrome-ai`                 | Gemini Nano caption → `nomic-embed-text` vector (`src/app.ts:1019-1101`) | 0 + 134 MB         |

Three consequences follow, and all three are the same problem seen from different angles.

**Nothing happens until a large download finishes.** The default variant is `sapiens2-fp16`
(`src/app.ts:206`) at 229 MB. On a metered connection or a cold cache, that is the entire first-run
experience.

**Throughput is capped by the local GPU.** When WebGPU is unavailable the app silently falls back to
WASM (`src/app.ts:1175-1181`, `src/sapiens2.ts:218-224`) and the device badge turns orange
(`src/app.ts:342-347`). That path works, but for a large library it is slow enough to change what the
tool is for.

**The model ceiling is the device ceiling.** The app can only run what fits in a browser tab. Purpose-
built multimodal retrieval models — the ones that would most improve clustering and search quality — are
out of reach by construction.

There is already a setting that *looks* like it addresses this and does not. `customModelHost`
(`src/types.ts:61`, `src/app.ts:2896-2901`) points at an alternative HuggingFace-compatible **mirror for
downloading model files**. It changes where weights come from, never where inference runs.

Meanwhile the surrounding ecosystem converged on one HTTP shape. Ollama, LM Studio, llama.cpp, vLLM,
Infinity and MLX servers all expose an OpenAI-compatible API on localhost; OpenRouter, Venice, Jina and
others expose the same shape over the internet. A user who already runs one of these has capable
inference sitting idle a few milliseconds away, and no way to point the app at it.

## Decision drivers

1. **Embed the media, not a description of it.** The remote path must be a peer of the `nomic` /
   `sapiens2` image-feature-extraction backends. Captioning as the *only* remote strategy would make the
   remote backend a variant of `chrome-ai`, which is a different and lower-fidelity thing.
2. **Support VLMs and video LLMs too.** These emit text, not vectors. Supporting them is a second
   pipeline, not a parameter of the first — and it is the only way to get a model that reasons about a
   video as a video.
3. **Any base URL, any key.** Localhost and internet, paid and free, no provider allowlist.
4. **Send thumbnails, not originals.** Both for cost and because CLIP-family encoders resize to 224–336 px
   internally; a full-resolution upload is pure waste.
5. **Opt-in, and visibly so.** The product's headline claim is that nothing leaves the device. A backend
   that uploads media must be chosen deliberately, never arrived at by accident.
6. **Reuse what exists.** The 768-d vector contract is not special — but the IndexedDB cache (`src/db.ts`),
   `l2normalize()` (`src/embeddings.ts:10-26`), `searchByCosine()` (`src/compute.ts:266-291`), the
   adaptive batcher (`src/batching.ts:48`) and `p-limit` (already a dependency) should all survive.
7. **Never leak the key.** `captureConsoleIntegration({ levels: ['error'] })` (`src/sentry.ts:15`) ships
   every `console.error` to a remote error sink. That is a live exfiltration path if a key ever reaches a
   log line.

## Considered options

| Option                                    | Embeds the image?         | Works with the user's stated targets                             | Verdict                                                                           |
|-------------------------------------------|---------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| Status quo — local only                   | Yes                       | —                                                                | **Rejected.** Does not address any driver                                         |
| Remote `/embeddings` with image input     | **Yes**                   | OpenRouter, vLLM, Jina, Infinity                                 | **Chosen — pipeline A**                                                           |
| Remote VLM caption → text embedding       | No — embeds a description | Every chat provider, incl. Ollama; the only route for video LLMs | **Chosen — pipeline B**                                                           |
| Remote captions only (extend `chrome-ai`) | No                        | —                                                                | Rejected — violates driver 1                                                      |
| Proxy service to normalise providers      | Yes                       | —                                                                | Rejected — reintroduces a server (the product constraint the app exists to avoid) |
| Upload originals rather than thumbnails   | Yes                       | —                                                                | Rejected — violates driver 4                                                      |

### The awkward fact: there is no standard for image embeddings

`/v1/chat/completions` with an `image_url` content part is genuinely universal. `/v1/embeddings` with an
image is not. Verified state as of this ADR:

| Target     | Endpoint                       | Image input shape                                                                                                                                |
|------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| OpenRouter | `POST /v1/embeddings`          | `input: [{ content: [{ type: 'image_url', image_url: { url } }] }]`                                                                              |
| vLLM       | `POST /v1/embeddings`          | `messages: [{ role, content: [...] }]` — chat-shaped                                                                                             |
| Jina AI    | `POST /v1/embeddings`          | `input: [{ image: '<b64\|url>' }]`, plus a `dimensions` param                                                                                    |
| Infinity   | `POST /embeddings`             | `input: ['<url \| data URI>']` — server auto-detects                                                                                             |
| llama.cpp  | `POST /embedding`              | `{ content: 'Image: [img-1]', image_data: [{ id, data }] }` — and [reported unreliable](https://github.com/ggml-org/llama.cpp/discussions/13666) |
| **Ollama** | `/api/embed`, `/v1/embeddings` | **none — text only** ([ollama#5304](https://github.com/ollama/ollama/issues/5304), open since June 2024)                                         |

Two things follow.

**Ollama cannot do pipeline A today.** It runs VLMs perfectly well over `/v1/chat/completions`, so it is
fully served by pipeline B. This must be stated in the UI, not discovered as an HTTP 400.

**A single hard-coded request body would work with roughly one provider.** The wire format has to be a
selectable, probeable dimension of the configuration. This is the main reason the implementation is
larger than "add a `fetch` call".

### What pipeline A buys on the user's preferred provider

`GET https://openrouter.ai/api/v1/embeddings/models` returns 31 models, of which three accept images:

| Model                                       | Modalities                          | Notes                                                 |
|---------------------------------------------|-------------------------------------|-------------------------------------------------------|
| `nvidia/llama-nemotron-embed-vl-1b-v2:free` | text, image                         | **free** — makes the whole path testable at zero cost |
| `voyageai/voyage-multimodal-3.5`            | text, image                         |                                                       |
| `google/gemini-embedding-2`                 | text, image, **video**, audio, file | direct video embedding is reachable                   |

## Decision outcome

Add one new `ModelVariant`, `openai`, with two selectable pipelines behind a single client, a single
settings block, and a single consent gate.

- **Pipeline A — `direct`.** Media → `POST {baseUrl}/embeddings` → vector. One request. Image and text
  land in the same space because they go through the same endpoint and model, which makes semantic
  search strictly better-founded than today's caption-mediated `chrome-ai` search.
- **Pipeline B — `vlm`.** Media → `POST {baseUrl}/chat/completions` → caption → vector, where the vector
  comes from either `{baseUrl}/embeddings` or the local `nomic-embed-text` the `chrome-ai` path already
  loads (`src/app.ts:1054-1082`). Works with every chat provider, and is the route for video LLMs.

Both sit behind one interface so `embedAll()` (`src/app.ts:1433`) gains one branch rather than two.

### Thumbnails, not originals

Frames are re-encoded to JPEG at a default longest edge of **384 px**, quality 0.8 — roughly 25–40 KB, or
35–55 KB once base64 inflates it. `createImageBitmap(blob, { resizeWidth })` does the downscale during
decode, so a 48 MP original is never decoded at full size. The existing 224 px video frame
(`src/app.ts:504-555`) is reused directly where it is already in hand.

### Video

`extractVideoFrame()` seeks once to `min(1.0, duration / 2)` and returns one bitmap — the same single-
still limitation ADR-0001 documents. Pipeline B generalises it to *n* evenly-spaced frames sent as
multiple `image_url` content parts in one request, which every OpenAI-compatible chat server accepts.
`video_url` parts exist on some providers and are treated as an optional extra, not the mechanism.

The `<video>` teardown in that function (`src/app.ts:518-526`) is load-bearing: Chrome caps concurrent
`WebMediaPlayer`s at ~75, which is also why `pLimit(4)` guards it (`src/app.ts:678`). *n* frames means
*n* seeks on **one** element and one teardown.

### The key never enters application state

The API key lives in `localStorage['mc_openai_key']` (or `sessionStorage`), reachable only through
accessor functions — never in `mc_settings`, never in `state.settings`, never in a URL.

This is structural, not disciplinary. `src/sentry.ts:7-8` already parses `mc_settings` at module load;
`state` is the obvious target of any future debug dump; and Sentry's default breadcrumbs record fetch
**URLs**, so a key in a query string would be uploaded on the next unrelated `console.error`. Keeping the
secret out of those objects means no accidental serialisation path exists to audit. There is precedent:
`mc_chrome_ai_prompt` (`src/app.ts:267`) already lives outside `mc_settings`.

### Consent is explicit and named

A one-time modal, naming the destination host, before the first remote run; remembered per base URL. The
existing `modelFallbackModal` `<dialog>` and its handlers are the pattern. The device badge
(`src/app.ts:305-352`) gains a `Remote · <host>` state so the condition stays visible afterwards, and the
model `<option>` says so in the dropdown itself (`index.html:807-812`).

## Consequences

### Good

- First embedding starts in seconds instead of after a 229 MB download.
- Access to models that cannot run in a tab, including purpose-built multimodal retrieval models and
  video LLMs.
- Pipeline A gives a genuinely shared image/text embedding space — the property ADR-0001 notes is *lost*
  when moving to pooled SigLIP features. The two ADRs pull in opposite directions here on purpose: 0001
  optimises for a device that cannot afford a network round-trip, 0002 for one that can.
- Works offline-ish in the sense that matters to a homelab user: a localhost server is not "the cloud".
- Forces the fix of several latent bugs (below) that are real today.

### Bad

- **The privacy claim becomes conditional.** `README.md:15` says "No media is uploaded. No server
  involved"; `index.html:745` says "zero uploads, zero server"; `index.html:781` says "Built with ❤️ for
  Privacy". These are unqualified today and would be false for a user on this backend. They must be
  rewritten to say local backends never upload and this one does. Watering the claim down is a real cost
  and is accepted deliberately, not hidden.
- **Per-image cost and rate limits become failure modes** the app has never had. A folder of 10 000
  photos is 10 000 requests.
- **Results depend on a third party** — availability, model deprecation, and silent dimension changes.
- **A key in browser storage is readable by anything with access to that browser profile.** Mitigated by
  a `sessionStorage` option and by saying so in the UI; not eliminated.
- **More configuration surface than the rest of the app combined**: base URL, key, pipeline, two model
  fields, wire format, concurrency, image size, frame count.
- Six model variants become seven, and this one has a settings panel.

### Neutral — the 768 assumption ends

Remote embedding dimensions are provider-chosen and unknown until the first response. Four hard-coded
`new Float32Array(768)` fallbacks (`src/app.ts:1536`, `:1622`, `:1645`, `:3182`) become a
`zeroVector()` reading a dimension probed at load time. `extractVector()` and `extractBatchedVectors()`
(`src/embeddings.ts`) already derive width from `dims` and need no change.

The IndexedDB cache namespace correspondingly widens from `@sapiens2/`-style constants
(`src/app.ts:354-360`) to cover **every input that changes the resulting vector** — the full base URL,
pipeline, model, embedder, dimension, wire format, and also the describe prompt, image size and video
frame count. Anything left out is a setting that appears to do nothing: change it, reload, and the cache
serves vectors built from the old input. Namespacing is non-destructive, so the cost of including a
setting is one re-embed, and reverting restores the previous cache for free.

### Latent bugs this surfaces

None of these are introduced by this ADR; all are load-bearing for it. Full list in
`REMOTE_INFERENCE_PLAN.md`, but three set the shape of the design:

- `isDownloadError()` (`src/modelFallback.ts:111-124`) matches `unauthorized`, `failed to fetch` and
  bare status codes, so an HTTP 401 from an API would open the *HuggingFace model-upload fallback modal*
  (`src/app.ts:1346-1351`).
- `searchByCosine()` clamps to the shorter vector on a dimension mismatch (`src/compute.ts:280-285`), so
  mixing embedding spaces yields a plausible ranking rather than an error.
- `embedText()` (`src/app.ts:623-634`) never calls `l2normalize()`, relying on Transformers.js
  `normalize: true`. Remote `/embeddings` responses are not guaranteed unit-norm, and `searchByCosine()`
  is a raw dot product.

## Verification and rollback

**Verification.** The decisive test is end-to-end and cheap: configure OpenRouter with
`nvidia/llama-nemotron-embed-vl-1b-v2:free`, embed a small folder, then run a text search and confirm the
results are semantically right. That single result proves the image vectors and the query vector share a
space — the property the whole design turns on. Run it under `npm run dev` (an `http://localhost` origin)
so Chrome's Local Network Access restrictions do not confound the result.

**One assumption needs confirming before anything is built on it.** OpenRouter's embeddings documentation
shows image inputs as `https://` URLs. Local files cannot be public URLs, so pipeline A on OpenRouter
requires `data:` URIs to be accepted. This is inferred from the fact that the adjacent chat-completions
and rerank endpoints both document data-URI support — it has not been observed on `/embeddings`. One
`curl` settles it. If it fails, pipeline A still stands on Jina, vLLM and Infinity, and OpenRouter users
take pipeline B; nothing else in the design changes.

**Rollback.** `openai` is an additional `ModelVariant`. Switching variants already forces a page reload
(`src/app.ts:2827-2834`) and each variant writes under its own IndexedDB cache prefix
(`src/app.ts:1450-1456`). Rolling back is "choose a different model in Settings" — no migration, no data
loss, previously cached local vectors untouched. Removing the feature entirely means deleting one branch
in `loadModelOnce()`, one in `embedAll()`, one module, and one settings block.
