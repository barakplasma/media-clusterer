# 0001. Adopt a small video language model for captioning and embeddings

- **Status:** Proposed
- **Date:** 2026-08-01
- **Supersedes:** nothing (first ADR)

## Context and problem statement

Two problems converge on the same decision.

**The captioner does not fit the target device.** The app's caption backend is Chrome's built-in Prompt
API, i.e. Gemini Nano (`src/chromeAI.ts`, wired at `src/app.ts:1019-1101`, `src/app.ts:1569-1595`,
`src/app.ts:2500-2553`). Chrome's published hardware floor for that model is **more than 4 GB of VRAM, or
16 GB of RAM plus 4 CPU cores for the CPU path, and 22 GB of free disk**. An 8 GB Chromebook clears none
of those bars, and captioning crashes the tab there. The size is not negotiable: the model is chosen and
managed by the browser, so the app cannot ask for a smaller quantisation, a shorter context, or a lower
resolution. Its only lever today is to disable the feature (`src/app.ts:2696-2713`).

**The app claims video support it does not have.** `README.md` advertises MP4/WEBM support, but
`extractVideoFrame()` (`src/app.ts:504-555`) seeks once to `min(1.0, duration / 2)` and returns a single
224 px `ImageBitmap`. That one still is the only thing ever embedded, clustered, or captioned. Nothing
temporal is modelled anywhere in the pipeline, and the caption prompt itself is image-specific in its
wording (`src/chromeAI.ts:52-56`). A tool whose stated purpose includes video is, in practice, treating
each video as one badly-chosen photograph.

A third fact shapes the solution. The `chrome-ai` path is really *two* models: Gemini Nano turns an image
into a caption, then a separately downloaded 134 MB `nomic-embed-text-v1.5` turns that caption into the
768-dimensional vector the rest of the app actually consumes (`src/app.ts:1054-1082`,
`src/app.ts:1590-1594`). We pay twice — once in download size, once in latency — to get one vector.

## Decision drivers

1. **Fit an 8 GB Chromebook.** Roughly a 1 GB inference budget, and the app must control quantisation.
2. **Video-native.** Multi-frame temporal reasoning, because that is what the product claims to do.
3. **Under 1 B parameters.** A hard ceiling for the low-end target.
4. **Zero-server.** Non-negotiable product constraint (`README.md`): nothing leaves the device.
5. **Reuse what exists.** The 768-d vector contract, the IndexedDB embedding cache (`src/db.ts`), DruidJS
   projection and k-means (`src/compute.ts`), and the offline/proxy model fallback
   (`src/modelFallback.ts`) should all survive unchanged.
6. **Degrade, don't disable.** Weaker hardware should get a smaller tier, not an error message.

## Considered options

| Option                           | Params          | Video-native    | Browser-ready                                  | Verdict                                                                                 |
|----------------------------------|-----------------|-----------------|------------------------------------------------|-----------------------------------------------------------------------------------------|
| Status quo — Gemini Nano         | Browser-managed | No              | Built-in                                       | **Rejected.** Exceeds the device budget; size and context are not negotiable by the app |
| **SmolVLM2-256M-Video-Instruct** | 256 M           | **Yes**         | ONNX in-repo, `transformers.js_config` present | **Chosen — default tier**                                                               |
| **SmolVLM2-500M-Video-Instruct** | 507 M           | **Yes**         | Same                                           | **Chosen — opt-in tier** for capable machines                                           |
| SmolVLM2-2.2B                    | 2.2 B           | Yes             | Yes                                            | Rejected — over the 1 B ceiling                                                         |
| FastVLM-0.5B                     | 0.5 B           | No — image only | Yes                                            | Rejected — does not address the video problem                                           |
| LFM2-VL-450M                     | 0.45 B          | No — image only | Partial                                        | Rejected — same reason                                                                  |
| Moondream2 / Qwen3-VL-2B         | ~2 B            | Partial         | Yes                                            | Rejected — over the ceiling                                                             |
| Server-side inference            | —               | —               | —                                              | Rejected — violates driver 4                                                            |

SmolVLM2's 256M and 500M checkpoints are, as of this writing, the only sub-1 B **video-native** VLMs with
shipped ONNX weights and a working WebGPU story. That is a statement about what exists in 2026, not a
permanent commitment — see "the model is an interface" below.

## Decision outcome

Adopt **SmolVLM2** via `@huggingface/transformers`, running in a Web Worker, in three selectable tiers.

`@huggingface/transformers` is already a dependency at `^4.2.0` and SmolVLM support landed upstream in
3.4.0, so **no dependency bump is required**.

### Tiers

| Tier | `ModelVariant`    | Download (q4f16)                      | Produces                        |
|------|-------------------|---------------------------------------|---------------------------------|
| A    | `smolvlm2-vision` | `vision_encoder` only — **55 MB**     | 768-d vectors only, no captions |
| B    | `smolvlm2-256m`   | vision + embed + decoder — **189 MB** | Captions **and** 768-d vectors  |
| C    | `smolvlm2-500m`   | vision + embed + decoder — **358 MB** | Same, higher quality            |

For comparison, the app's existing embedders download 116 MB (`sapiens2-int8`), 229 MB (`sapiens2-fp16`)
and 380 MB (`nomic`). **Tier A is the cheapest embedder the app has ever had**, and tier B — a full
video captioner *plus* an embedder — costs less than `nomic` alone.

### The load-bearing insight

When the captioner runs, the vision tower runs anyway. Pooling its output costs one extra reduction over
a tensor that already exists in memory. So tiers B and C yield **the caption and the clustering vector
from a single forward pass**, which is what lets us drop `nomic-embed-text` (134 MB) from the default
path. Today's `chrome-ai` chain needs two models to produce one vector; this needs one model to produce
two things.

The dimensions line up: SmolVLM2's `vision_config.hidden_size` is **768**, exactly the width the app
already uses everywhere (`src/db.ts`, `src/compute.ts`, `src/spatial.ts`, and the zero-vector fallback at
`src/app.ts:1536`). Mean-pooling the vision tower's patch embeddings and running the existing
`l2normalize()` (`src/embeddings.ts`) produces a drop-in replacement vector.

### Multi-frame video is affordable

`scale_factor` / `pixel_shuffle_factor` is 4, so each 512×512 frame's 1024 patches shuffle down to
**64 visual tokens** before entering the language model. Four sampled frames cost about 256 tokens of
prefill against a 576-hidden, 30-layer decoder — not the 4096 a naive reading would suggest. The model's
own `preprocessor_config.json` allows up to 64 frames at 1 fps; the app will default to **4**, well under
that ceiling, and expose the frame count as a setting.

`do_image_splitting` must be set to `false`. It defaults to `true` with `size.longest_edge: 2048`, which
tiles each frame into many sub-images — the single easiest way to blow the memory budget on the device
this ADR exists to support.

### The model is an interface, not a hard dependency

The implementation defines a tier descriptor — repository, dtype, frames per video, vision dimension —
and selects from a table. Adding a future sub-1 B video model is then a new row, not a rewrite. This
matters because the sub-1 B video-VLM field is young and moving; we are choosing the best current
occupant of a slot, and the slot should outlive the occupant.

### Chrome AI stays

`chrome-ai` remains a selectable variant, unchanged. On hardware that genuinely supports Gemini Nano it
costs 0 MB of download, which is a real advantage worth keeping. This ADR removes its status as the only
captioner, not its existence.

## Consequences

### Good

- The app fits the device it crashed on. Hugging Face reports SmolVLM2-256M inference under 1 GB of GPU
  RAM via WebGPU in the browser.
- Video is captioned as video — multiple frames, temporal prompt — for the first time.
- One model replaces two; `nomic-embed-text` leaves the default path.
- The app controls quantisation, frame count and resolution, so weak hardware degrades to a smaller tier
  instead of failing.
- A new capability falls out for free: **find visually similar media**, cosine over the pooled vision
  vectors, reusing `searchByCosine()` (`src/compute.ts:266-291`) with an image as the query.

### Bad

- **The shared image/text embedding space is lost.** `nomic` embeds images and text into one space, so
  typing a query and comparing cosines Just Works. Pooled SigLIP features have no such alignment with
  query text. Search must be re-designed (below).
- Mean-pooled SigLIP features are a general-purpose visual signal, not a purpose-trained retrieval
  embedder. Clustering quality relative to `sapiens2-fp16` **must be measured, not assumed** — which is
  why tier A ships standalone and early.
- This introduces the repository's first Web Worker. `public/_headers` already sets COOP/COEP, so the
  prerequisites exist, but worker + Vite 8 + `onnxruntime-web`'s CDN WASM path (`vite.config.ts`,
  `excludeOrtWasm`) is new ground and needs a smoke test before anything is built on it.
- More total code paths: five model variants become eight.

### Neutral — how search changes

The `enableTextSearch` setting already exists (`src/types.ts:53`) and already governs whether the text
model is downloaded. That contract is kept, with a new default:

- **Off (new default):** lexical search over captions — BM25/TF-IDF, no model, 0 MB.
- **On:** lazily load `nomic-embed-text` exactly as today (`src/app.ts:1054-1082`) and get the current
  cosine semantic search back, unchanged.
- **New:** search-by-example over pooled vision vectors.

Nothing regresses for a user who wants semantic search; it is simply no longer mandatory, and no longer
downloaded by users who never type a query.

## Verification and rollback

**Verification.** The acceptance test for the original bug is not a unit test: load a folder of mixed
photos and videos on the 8 GB Chromebook and confirm peak JS heap and GPU memory stay under roughly 1 GB
while the canvas remains interactive. Everything else — cache hits, caption quality, download sizes — is
checkable in DevTools. `VIDEO_LM_PLAN.md` carries the full checklist.

**One assumption needs confirming on day one.** `vision_encoder.onnx` at fp32 is 374 MB, which is about
93.6 M parameters — the size of the bare SigLIP-base tower, implying the export stops *before* the
pixel-shuffle connector and therefore emits 768-d patch embeddings. That inference is from file size, not
from the graph. Load the model, print `session.outputNames` and the output dims, and confirm. If it turns
out to be the post-connector tensor, it is 576-d and nothing breaks — each variant already gets its own
IndexedDB cache prefix, so the only change is the `visionDim` field on the tier descriptor.

**Rollback.** Tiers A/B/C are additional `ModelVariant` values. Switching variants already forces a page
reload (`src/app.ts:2827-2834`), and each variant writes under its own IndexedDB cache prefix
(`src/app.ts:1450-1456`). Rolling back is "choose a different model in Settings" — no migration, no data
loss, previously cached vectors untouched.
