# Plan: Video Language Model Integration

Implementation plan for [ADR-0001](docs/adr/0001-small-video-language-model.md), which decides to adopt
SmolVLM2 as a small, video-native, app-controlled replacement for the Gemini Nano captioner and as the
source of the clustering vector. Written 2026-08-01.
This document is a plan only — no changes are implemented by the PR that introduces it.

Milestones are `M1`…`M6`, in dependency order. `M2` is deliberately shippable on its own.

## Context

**Current Problem:**

- Captioning via Chrome's Prompt API crashes 8 GB Chromebooks. Gemini Nano's floor is >4 GB VRAM, or
  16 GB RAM + 4 cores on CPU, plus 22 GB free disk — and the app cannot request anything smaller because
  the browser owns the model (`src/chromeAI.ts`, `src/app.ts:1019-1101`).
- Videos are treated as one still. `extractVideoFrame()` (`src/app.ts:504-555`) seeks once to
  `min(1.0, duration/2)` at 224 px; that frame is the entire representation of the video
  (`src/app.ts:1476-1507`).
- The caption prompt is image-specific in its wording (`src/chromeAI.ts:52-56`) — it asks about "dominant
  colors" and "lighting", never about motion or change.
- The `chrome-ai` path downloads two models to produce one vector: Nano for the caption, then 134 MB of
  `nomic-embed-text-v1.5` to vectorise it (`src/app.ts:1054-1082`, `src/app.ts:1590-1594`).
- Captions are written with a synchronous `localStorage.setItem` per file inside the embed hot loop
  (`src/app.ts:1585`), with no eviction and only a `console.warn` on quota exceed — already flagged as
  `IMPROVEMENT_PLAN.md` P1-1.

**Desired Outcome:**

- Captioning and embedding fit in roughly 1 GB on an 8 GB Chromebook, with the app choosing the tier.
- Videos are described from several frames, with a prompt that asks about action over time.
- One model produces both the caption and the 768-d clustering vector.
- Semantic text search becomes opt-in rather than a mandatory 134 MB download.
- Captions live in IndexedDB, off the synchronous hot path.

## Implementation Approach

### Libraries to Add

**None.** `@huggingface/transformers` is already a dependency at `^4.2.0` (`package.json`), and SmolVLM
support landed upstream in transformers.js 3.4.0. No new packages, no version bump.

### Model facts that drive every sizing decision

Verified against the Hub for `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`:

| Fact | Value | Why it matters |
| --- | --- | --- |
| `vision_config.hidden_size` | **768** | Exactly the app's existing vector width — pooled features are a drop-in |
| `vision_config.image_size` / `patch_size` | 512 / 16 | 1024 patches per tile |
| `scale_factor` / `pixel_shuffle_factor` | 4 | 1024 patches → **64 visual tokens per frame** into the LLM |
| `text_config` | 576 hidden, 30 layers, vocab 49280 | Tiny decoder; 4 frames ≈ 256 tokens of visual prefill |
| `video_sampling` | `fps 1`, `max_frames 64`, `longest_edge 512` | The model's native video protocol; we use 4 frames, far below the ceiling |
| `do_image_splitting` default | `true`, `size.longest_edge 2048` | **Must be set to `false`** — tiling multiplies memory and is the easiest way to blow the budget |

ONNX file sizes, which set the tier download figures:

| File (q4f16) | 256M | 500M |
| --- | --- | --- |
| `vision_encoder` | 55 MB | 58 MB |
| `embed_tokens` | 57 MB | 95 MB |
| `decoder_model_merged` | 77 MB | 205 MB |
| **Total** | **189 MB** | **358 MB** |

For scale, the app's current embedders cost 116 MB (`sapiens2-int8`), 229 MB (`sapiens2-fp16`), 380 MB
(`nomic`). Tier A at 55 MB is the cheapest embedder the app has ever shipped.

> ⚠️ **Confirm before building on it (M2, day one).** `vision_encoder.onnx` at fp32 is 374 MB ≈ 93.6 M
> parameters ≈ the bare SigLIP-base tower, which implies the export stops *before* the pixel-shuffle
> connector and therefore emits 768-d patch embeddings. That is inferred from file size, not read off the
> graph. Load the model, print `session.outputNames` and the output dims, and check. If it is instead the
> post-connector tensor it will be 576-d — nothing breaks, because each variant already writes under its
> own cache prefix; the only change is the `visionDim` field on the tier descriptor.

### Tiers

| Tier | `ModelVariant` | Download | Produces | Requires |
| --- | --- | --- | --- | --- |
| A | `smolvlm2-vision` | 55 MB | 768-d vectors | WASM is fine |
| B | `smolvlm2-256m` | 189 MB | Captions + vectors | WebGPU recommended |
| C | `smolvlm2-500m` | 358 MB | Captions + vectors | WebGPU |

The tier is a data row, not a code branch — see "Key Design Decisions".

### Files to Modify

1. **`src/types.ts`**
   - Extend `ModelVariant` with `'smolvlm2-vision' | 'smolvlm2-256m' | 'smolvlm2-500m'`.
   - Add `VlmTier`, and the worker request/response message union.
   - Add `framesPerVideo: number` to `Settings`.

2. **`src/app.ts`**
   - New branch in `loadModelOnce()` (`:1013`) for the `smolvlm2-*` variants.
   - New branch in `embedAll()` (`:1433`) alongside the existing `isSapiens2` / `isChromeAI` paths
     (`:1442-1443`, `:1566-1595`); new cache prefixes in the `cachePrefix` chain (`:1450-1456`).
   - Generalise `extractVideoFrame()` → `extractVideoFrames(file, n)` (`:504-555`).
   - Caption read/write sites move to IndexedDB (`:1551-1553`, `:1585`, `:2489-2493`, `:2536-2540`).

3. **`src/db.ts`**
   - `DB_VERSION` 1 → 2 (`:11`), new `captions` object store, `captionGetBatch` / `captionPutBatch`
     mirroring `cacheGetBatch` / `cachePutBatch` (`:62`, `:111`).

4. **`src/hardware.ts`**
   - Add `pickVlmTier()` next to the existing `computeOptimalBatchSize()` / `getMemoryPressure()`.

5. **`src/modelFallback.ts`**
   - Add SmolVLM2 repo/file lists to `modelDownloadUrls()` (`:42`). `buildUploadCache()` (`:70`) needs no
     change — it already matches on the path after `/resolve/<rev>/`.

6. **`index.html`**
   - Three new `<option>`s in `#model-select` (`:807-812`).
   - Frames-per-video setting in the settings modal.
   - Rename the `#chrome-ai-prompt` setting to a model-agnostic caption-prompt setting (`:823-830`).

7. **New files**
   - `src/vlm.ts` — main-thread facade.
   - `src/vlmWorker.ts` — the worker.
   - `src/vlm.test.ts` — unit tests for the pure parts.

### Data Flow

```text
                       collectImages() → embedAll()
                                            │
                        ┌───────────────────┴──────────────────┐
                        ↓                                      ↓
                  image file                             video file
                        │                                      │
                        │                        extractVideoFrames(file, n)   [M3]
                        │                            (one <video>, n seeks)
                        ↓                                      ↓
                  1 ImageBitmap                        n ImageBitmaps @ 512px
                        └───────────────────┬──────────────────┘
                                            ↓
                                postMessage → src/vlmWorker.ts        [M1]
                                            ↓
                              AutoProcessor (do_image_splitting: false)
                                            ↓
                                    vision_encoder.onnx
                                            ↓
                        ┌───────────────────┴──────────────────┐
                        ↓                                      ↓
              mean-pool + l2normalize                  decoder (tiers B/C only)
                (src/embeddings.ts)                             ↓
                        ↓                                  caption text
              Float32Array(768)                                 ↓
                        ↓                                       ↓
            IndexedDB `embeddings`                    IndexedDB `captions`  [M5]
              @smolvlm2-*/ prefix
                        ↓
        runProjection() → kmeansAsync() → canvas    (unchanged, src/compute.ts)
```

Tier A stops at the left branch; the decoder is never downloaded.

### Key Design Decisions

**One forward pass, two outputs.** The vision tower runs whether or not we want a caption, so pooling its
output is nearly free. This is the whole reason `nomic-embed-text` can leave the default path — see
ADR-0001. Tiers B and C must read the pooled tensor off the *same* run that produced the caption, not
re-run the encoder.

**The tier is data, not control flow.** Define one descriptor and select from a table:

```typescript
interface VlmTier {
  id: ModelVariant
  repo: string                  // e.g. 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
  dtype: 'q4f16' | 'q4' | 'int8' | 'fp16' | 'fp32'
  visionOnly: boolean           // tier A: never load embed_tokens/decoder
  visionDim: number             // 768, pending the day-one confirmation above
  framesPerVideo: number
  cachePrefix: string           // '@smolvlm2-256m/' etc.
}
```

Adding a future sub-1 B video model is then a new row. Avoid `variant.startsWith('smolvlm2')` string
tests scattered through `app.ts` — the existing `isSapiens2` / `isChromeAI` pattern (`:1442-1443`) is
already at the edge of what is readable with three backends and will not survive six.

**Frame extraction reuses one `<video>` element.** The current function creates an element per call and
tears it down with `pause()` → `src=''` → `load()`. That teardown exists because Chrome caps concurrent
`WebMediaPlayer`s at ~75, and `pLimit(4)` (`:678`) exists for the same reason. Extracting *n* frames must
seek the **same** element *n* times and tear down once — creating *n* elements would multiply the
pressure that those guards were added to contain. Per `AGENT.md`, every `ImageBitmap` gets `.close()`
after use, and the object URL is revoked once.

**Two resolutions, two purposes.** Display thumbnails stay at 96 px via `lazyDecodeThumbnail()`
(`:563-620`) — untouched. VLM frames are extracted at longest-edge 512 to match `video_size`, and are not
cached in `state.thumbnails`. The current code conflates these (the 224 px video frame serves as both),
which is why video thumbnails are oddly large today.

**Separate prompts for stills and video.** `DEFAULT_DESCRIBE_PROMPT` (`src/chromeAI.ts:52-56`) asks about
colour, mood and lighting — correct for a photograph. The video prompt should ask what happens and what
changes across the frames. Both stay user-editable through the existing settings textarea.

**Captions move to IndexedDB before tiers B/C ship.** Video captions are longer than image captions, and
the current write is a synchronous `localStorage.setItem` per file inside the embed loop (`:1585`) with no
eviction. Shipping multi-frame captions onto that path would make an existing defect materially worse.
`M5` is therefore a hard prerequisite for `M4` reaching users, not a follow-up.

**Search degrades deliberately.** `enableTextSearch` (`src/types.ts:53`) already gates the text-model
download. Keep the flag, flip the default to off, and route default search through a lexical index over
captions. Users who want cosine semantic search turn it on and get exactly today's behaviour.

**Worker boundary chosen once.** `M1` is the repo's first `Worker`. `IMPROVEMENT_PLAN.md` P1-1 wants
projections moved off-thread too; pick a message protocol here that a projection worker can reuse, rather
than inventing a second one later.

### Implementation Steps

#### M1 — Worker scaffold

`src/vlmWorker.ts` + `src/vlm.ts` (facade). Model loading mirrors the existing webgpu→wasm fallback
(`src/app.ts:1069-1081`, `src/sapiens2.ts:270-321`):

```typescript
// src/vlmWorker.ts
import { AutoProcessor, AutoModelForVision2Seq } from '@huggingface/transformers'

let processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>> | null = null
let model: Awaited<ReturnType<typeof AutoModelForVision2Seq.from_pretrained>> | null = null

async function load(tier: VlmTier, onProgress: (e: ProgressEvent) => void) {
  processor ??= await AutoProcessor.from_pretrained(tier.repo, { progress_callback: onProgress })
  try {
    model ??= await AutoModelForVision2Seq.from_pretrained(tier.repo, {
      dtype: tier.dtype, device: 'webgpu', progress_callback: onProgress
    })
  } catch {
    model = await AutoModelForVision2Seq.from_pretrained(tier.repo, {
      dtype: tier.dtype, device: 'wasm', progress_callback: onProgress
    })
  }
}
```

Smoke-test the build early: worker + Vite 8 + the `excludeOrtWasm` plugin (`vite.config.ts`) that strips
`ort-wasm*` from `dist` and loads it from jsDelivr at runtime. `public/_headers` already sets
COOP `same-origin` / COEP `credentialless`, so `SharedArrayBuffer` is available.

#### M2 — Tier A: embeddings only (shippable alone)

Load `vision_encoder` only, mean-pool, normalise with the existing helper:

```typescript
import { l2normalize } from './embeddings'

function poolVision(features: { data: Float32Array; dims: number[] }): Float32Array {
  const [, tokens, dim] = features.dims          // expected [1, 1024, 768] — confirm on day one
  const out = new Float32Array(dim)
  for (let t = 0; t < tokens; t++)
    for (let d = 0; d < dim; d++) out[d] += features.data[t * dim + d]
  for (let d = 0; d < dim; d++) out[d] /= tokens
  return l2normalize(out)
}
```

Wire into `embedAll()` behind cache prefix `@smolvlm2-vision/`, reusing `readCachedEmbeddings()`
(`src/app.ts:1464`) and `cachePutBatch()` (`src/db.ts:111`) unchanged.

**This milestone is the measurement gate.** Before anything depends on pooled SigLIP features, compare
cluster quality against `sapiens2-fp16` on the same folder. If it is materially worse, ADR-0001's vector
decision gets revisited — cheaply, because nothing else has been built yet.

#### M3 — Multi-frame extraction

```typescript
async function extractVideoFrames(file: File, n: number): Promise<ImageBitmap[]> {
  // One <video>, n seeks, one teardown. See "Key Design Decisions".
  // Timestamps: uniform over (0, duration), avoiding the exact endpoints —
  // the first and last frames are frequently black or a title card.
  // Each frame: createImageBitmap(video, { resizeWidth: 512, resizeQuality: 'medium' })
  // Teardown once, exactly as extractVideoFrame does today (src/app.ts:518-526).
}
```

Keep `extractVideoFrame()` as `extractVideoFrames(file, 1)[0]` so the thumbnail path (`:606`) is
unaffected. Keep the `pLimit(4)` wrapper (`:678`) at the *file* level, not the frame level.

#### M4 — Tiers B/C: captions and vectors in one pass

`processor(text, images, { do_image_splitting: false })` → `model.generate()`, reading the pooled vision
tensor from the same run. Batch size 1 on tier B/C — the `chrome-ai` path's hard-coded 2 (`:1459`) exists
to feed two Gemini Nano session slots and does not apply here.

#### M5 — Captions to IndexedDB

`DB_VERSION` 1 → 2 in `src/db.ts:11`; add a `captions` store in `onupgradeneeded` (`:22-27`). Add
`captionGetBatch` / `captionPutBatch` mirroring the existing embedding functions. Batch caption writes
into the same queue as `writeQueue` in `embedAll()` (`:1437`). One-time read-through migration: on a miss,
check `localStorage` for `@caption/${name}:${size}:${lastModified}`, write it through, remove the key.

Cross-reference: this closes the caption half of `IMPROVEMENT_PLAN.md` P1-1.

#### M6 — Search and device gating

- Lexical caption index (BM25 or TF-IDF) in `src/compute.ts`, alongside `searchByCosine()` (`:266-291`).
- `enableTextSearch` default flips to `false`; when `true`, load `nomic-embed-text` exactly as today
  (`src/app.ts:1054-1082`).
- Search-by-example: `searchByCosine()` with a selected item's vector as the query.
- `pickVlmTier()` in `src/hardware.ts`, using `navigator.deviceMemory` and the WebGPU adapter limits
  already probed in `src/sapiens2.ts:270-321`. An 8 GB Chromebook must land on **256M / q4f16 / 4 frames /
  batch 1** without the user needing to know any of that. Do not offer tiers B/C when WebGPU is absent.

## Risks

| Risk | Mitigation | What settles it |
| --- | --- | --- |
| Pooled SigLIP clusters worse than `sapiens2-fp16` | M2 ships standalone and early; all existing variants stay selectable | Side-by-side cluster comparison at the end of M2 |
| WASM fallback too slow for multi-frame captioning | Tier A is WASM-viable; `pickVlmTier()` withholds B/C without WebGPU | Timing run on the Chromebook, WebGPU disabled |
| `vision_encoder.onnx` output shape assumption | `visionDim` on the tier descriptor; per-variant cache prefixes | `session.outputNames` + dims, M2 day one |
| First worker in the repo (Vite 8 + ORT CDN WASM) | Smoke-test the built bundle, not just `npm run dev` | `npm run build && npm run preview` at the end of M1 |
| Eight model variants is a lot of UI | Group the `<select>` with `<optgroup>`; `pickVlmTier()` picks a sensible default | Settings modal review |

## Verification

1. `npm run type-check && npm test` — existing suites stay green. `src/boot.test.ts` catches
   `index.html` ↔ `app.ts` binding drift when the new settings are added.
2. `npm run build && npm run preview` — verify the worker and the ORT WASM CDN path work in a
   production bundle, not only under `vite dev`.
3. `npm run dev`, open a folder of ~50 mixed photos and MP4/WEBM files. DevTools → Network shows exactly
   the expected model files for the selected tier and nothing else; Application → Cache Storage shows the
   transformers.js `models` bucket.
4. **The acceptance test for the original bug:** run step 3 on the 8 GB Chromebook, not only on a dev
   machine. DevTools → Memory: peak JS heap and GPU memory stay under roughly 1 GB on tier B. No tab
   crash.
5. The canvas stays interactive and pannable throughout embedding — the observable proof the worker is
   doing its job.
6. Open a video in the modal. The caption describes motion or change across the clip, not a single still
   — the observable proof M3 works.
7. Toggle `enableTextSearch` off → `nomic-embed-text` is never fetched (check Network), and typing a
   query still returns results via the lexical caption index.
8. Select an item and use search-by-example → visually similar media ranks first.
9. Reload → embeddings come from IndexedDB under `@smolvlm2-*/`, captions from the new `captions` store,
   and no `@caption/*` `localStorage` writes occur (check Application → Local Storage).
10. Switch model variant → the page reloads and the previous variant's cached vectors are still intact
    (the rollback path from ADR-0001).

## Suggested sequencing

| Step | Work | Why this order |
| --- | --- | --- |
| 1 | M1 worker scaffold | Everything else runs inside it; find build problems before writing features |
| 2 | M2 tier A embeddings | Shippable alone, and it is the gate on ADR-0001's vector decision |
| 3 | M5 captions to IndexedDB | Must land before multi-frame captions hit the localStorage hot path |
| 4 | M3 multi-frame extraction | Independent of the model; testable against tier A |
| 5 | M4 tiers B/C | Needs M3 for frames and M5 for storage |
| 6 | M6 search and gating | Needs captions (M4) to have something to index |
