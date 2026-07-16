# Improvement Plan

A prioritized plan of improvements for media-clusterer, based on a full survey of the
codebase (app/UI layer, AI/compute pipeline, and testing/CI/tooling) conducted 2026-07-15.
This document is a plan only — no changes are implemented by the PR that introduces it.

Priorities:

- **P1** — deep dives: highest-impact work, described in detail with a proposed approach.
- **P2** — detailed sections: important, smaller-scoped work.
- **P3** — brief entries: acknowledged debt, summarized only.

> ⚠️ **Security note (act first, independent of this plan):** a live Unsplash access key
> is committed in source at `src/app.ts:90` (`UNSPLASH_ACCESS_KEY`), and the
> Sentry/BugSink DSN is hardcoded at `src/sentry.ts:11`. The Unsplash key should be
> **rotated** and both should move to build-time environment configuration
> (`import.meta.env` / Vite `define`). Anything committed to a public repo history must be
> treated as compromised even after removal.

---

## P1-1 · Move compute off the main thread (Web Workers)

### Problem

Every heavy computation runs on the UI thread. There is no `Worker` anywhere in `src/`:

- **DruidJS projections** are the worst offender: `.transform()` is a single synchronous,
  non-yieldable call (`src/app.ts:843-877`). t-SNE/UMAP/Isomap/MDS on thousands of
  768-dim vectors freeze the tab for seconds; only a lone `await yieldMain()` precedes it.
- **Embeddings** (sapiens2 ONNX, nomic Transformers.js, Chrome AI describe) all run on the
  main thread, mitigated only by cooperative yielding.
- `kmeansAsync` and `spreadPointsAsync` (`src/compute.ts`) yield cooperatively via
  `YIELD_BUDGET_MS`, which helps but still competes with rendering.

Secondary main-thread costs found alongside:

- `Matrix.from(vectors.map(v => Float64Array.from(v)))` (`src/app.ts:843`) copies the
  entire dataset f32→f64, doubling memory before every projection.
- `searchByCosine` (`src/compute.ts:237`) full-sorts **all** N indices per query, though
  only top-K are shown — and `selectSmallest` (quickselect) already exists in the same
  file for culling and could be reused.
- k-means (`src/compute.ts:21-37`) uses random shuffle seeding (not k-means++), clusters
  the 2-D **projected** points rather than embeddings, and never re-seeds empty clusters.
- `cachePutBatch` is called once per inference batch (`src/app.ts:1474`); with sapiens2's
  batchSize=1 that is one IndexedDB transaction **per image**. `cacheStats` cursor-scans
  the whole store on every refresh (`src/db.ts:144`).
- Chrome-AI captions write `localStorage` synchronously per image
  (`src/app.ts:1390,1424`).
- Render-loop churn: a fresh `Int32Array(pts.length)` and a `Set` are allocated **every
  frame** (`src/app.ts:677,690`); `new Image()` is constructed inside the draw loop on
  cache miss (`src/app.ts:758`); the O(n) nearest-point-to-center scan is duplicated three
  times (`src/app.ts:694-702`, `:2980`, `:3004`).

### Proposed approach

1. **Projection worker first** (biggest freeze): a `projection.worker.ts` that imports
   DruidJS and receives embeddings as **transferable** Float32Array buffers, posting
   progressive results back as messages (replacing the current in-loop progressive-PCA
   yields). `compute.ts` is already DOM-free and can be imported by the worker as-is.
2. **Embedding worker second**: ONNX Runtime Web and Transformers.js both support workers;
   move `sapiens2.ts` inference and the nomic pipeline behind a worker boundary with a
   small RPC protocol (`embedBatch(files) → Float32Array + shape`). Chrome AI's Prompt API
   is main-thread-only today, so that path keeps its current structure.
3. **Micro-optimizations third** (independent, small PRs each):
   - top-K partial selection in `searchByCosine` (adapt `selectSmallest`, which
     partitions for the *smallest* keys — cosine similarity needs the *largest*, so
     negate the scores or add a select-largest variant);
   - accumulate IDB writes across batches, flush every ~50 items;
   - k-means++ seeding and empty-cluster re-seeding;
   - hoist per-frame allocations out of `render()`; extract a single
     `findNearestToCenter()` helper; move `Image` loading out of the draw path into a
     small request queue;
   - fix the full-res object-URL leak: URLs created at `src/app.ts:756` are only revoked
     in bulk on reset (`src/app.ts:636,1549`) while LRU eviction just sets `img.src = ''`
     (`:748,824,831`), so long sessions accumulate an object URL for every image ever
     viewed at high zoom — revoke on eviction. This can crash tabs on memory-constrained
     devices, which is why it lives here with the performance work rather than in P2-3.

### Files

`src/app.ts` (runProjection, embedAll, render), `src/compute.ts`, `src/db.ts`,
new `src/workers/projection.worker.ts`, new `src/workers/embedding.worker.ts`,
`vite.config.ts` (worker bundling).

---

## P1-2 · Split the `app.ts` monolith

### Problem

`src/app.ts` is 3,104 lines mixing state, rendering, input, modal, navigation, settings,
file walking, and pipeline orchestration:

- Global mutable `state` (`src/app.ts:202-219`) plus a separate module-level `camera`
  (`:222`) and ~15 loose `let` singletons (`:225-246`); consistency is maintained by
  scattered manual `scheduleRender()` calls.
- The `DOMElements` interface (`src/types.ts:73-139`) hand-duplicates the `dom` literal
  (`src/app.ts:94-160`) — 66 entries maintained in two places — while other elements are
  fetched ad hoc with `getElementById` inside functions (`src/app.ts:259, 269-271,
  2310-2312, 3088`), so there are two conventions for DOM access.
- Large copy-paste blocks: the model-ready UI block appears three times
  (`src/app.ts:956-972`, `:996-1012`, `:1078-1091`); the text-model progress callback
  twice (`:931-953` vs `:1052-1074`); clear-search logic three times (`:1994-2023`); the
  datetime-breadcrumb construction is ~90 lines of repeated span/sep/link creation
  (`:2101-2178`); the cache-prefix expression is duplicated (`:1299-1302` vs
  `:2838-2841`); localStorage quota handling is repeated five times.
- `index.html` carries a ~630-line inline `<style>` block (lines 14-644) plus 56 inline
  `style=""` attributes — no stylesheet file.

### Proposed module layout

| Module               | Contents (current locations)                                                                                                     |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `src/renderer.ts`    | `render` (`:669-835`), `resizeCanvas`/`fitCamera` (`:606-626`), camera math, LOD constants                                       |
| `src/input.ts`       | pointer/wheel/pinch (`:1895-1977`), keyboard nav (`:2966-3081`)                                                                  |
| `src/modal.ts`       | `openFileModal`/`closeModal` (`:2027-2372`), EXIF dialog, breadcrumbs                                                            |
| `src/sources.ts`     | `collectImages` (`:348`), `loadDemoImages` (`:405`), `extractVideoFrame`                                                         |
| `src/modelLoader.ts` | `loadModelOnce`/`loadModel`/fallback modal (`:885-1218`) — orchestration on top of the existing `modelFallback.ts`/`sapiens2.ts` |
| `src/settingsUI.ts`  | the ~300 lines of `addEventListener('change') → saveSettings()` wiring (`:2374-2670`), rewritten as a declarative binding table  |
| `src/router.ts`      | `parseURLHash`/`updateURL`/popstate (`:1803-1880`)                                                                               |
| `src/styles.css`     | extracted from the `index.html` inline block                                                                                     |

### Extraction order (lowest-risk first)

1. Settings binding table (pure de-duplication, no behavior change).
2. Renderer (already nearly self-contained; takes `state` + `camera` as parameters).
3. Modal + EXIF dialog.
4. Router / URL state.
5. Pipeline orchestration (`embedAll`, `processFiles`) — last, after P1-1 workers land, so
   the worker boundary and the module boundary are drawn once.

Keep the boot smoke test (`src/boot.test.ts`) green at every step; generate `dom` from a
single source of truth (derive the type from the literal with `satisfies`, deleting the
hand-maintained `DOMElements` interface). Each extraction makes real unit tests possible
for that module — the biggest enabler for the testing debt noted in P3.

---

## P1-3 · Cache correctness & robustness

### Problem

The embedding cache can silently serve stale or corrupt vectors, and hardware/failure
handling differs per model path:

- **No cache versioning.** Cache keys are namespaced by variant prefix only
  (`src/app.ts:1299-1302`), not by model revision or dtype. `searchByCosine` carries an
  explicit "mixed-model cache corruption" guard (`src/compute.ts:251-254`) that treats the
  symptom instead of the cause. `DB_VERSION = 1` (`src/db.ts:11`) has no model-version
  metadata. The fp16 model reuses the `sapiens2-model-v2` Cache-API name and the legacy
  `@sapiens2/` prefix (`src/sapiens2.ts:19`, `src/app.ts:1301`), so re-exported weights
  would not invalidate old embeddings.
- **Zero-vector poisoning.** On inference failure the batch is zero-filled
  (`src/app.ts:1448,1468`) and those `Float32Array(768)` zeros **can be cached**,
  permanently corrupting clusters with no user-visible warning.
- **Inconsistent WebGPU detection.** `sapiens2.ts` checks adapter limits and produces a
  structured `fallbackReason`; the nomic path is a bare try/catch
  (`src/app.ts:1034-1043`); `hardware.ts` does no GPU detection at all and sizes batches
  from Chrome-only `navigator.deviceMemory` / `performance.memory`
  (`src/hardware.ts:10-27`).
- **Fragile failure classification.** `isDownloadError` (`src/modelFallback.ts:106`) is a
  regex over error messages; `navigator.onLine === false` short-circuits genuine compute
  failures into the download-recovery modal.
- **CDN dependency.** onnxruntime WASM is loaded from jsdelivr (`src/sapiens2.ts:202`),
  contradicting the "zero-server, fully local" positioning and bypassing the app's own
  offline-fallback machinery; the COOP/COEP requirement for threading is noted in a
  comment but never verified at runtime.
- Module-level `_tensorBuf`/`_canvas` singletons (`src/sapiens2.ts:146`) are safe only
  while inference stays strictly sequential; no global `unhandledrejection` handler
  exists, and several async paths swallow errors silently
  (`src/app.ts:544, 2038, 2263`).

### Proposed approach

1. **Cache-version stamp**: add a metadata store to the IndexedDB schema holding
   `{modelId, revision, dtype, embeddingDim}` per variant; on mismatch, drop (or migrate)
   that variant's entries. Bump `DB_VERSION` with an upgrade handler.
2. **Never cache failures**: replace zero-fill-and-cache with a `failed` marker per file;
   surface a toast/count of failed embeds and allow retry.
3. **Shared capability probe**: one `probeWebGPU()` (adapter, limits, device) in
   `hardware.ts`, consumed by all three model paths; batch sizing takes the probe result
   into account instead of heap heuristics alone.
4. **Self-host ort WASM**: ship the `.wasm` files as static assets (the existing
   `excludeOrtWasm` Vite plugin already manages ort assets for the Cloudflare size limit —
   extend it rather than fight it).
5. **Structured errors**: have download paths throw a typed `ModelDownloadError` instead
   of classifying by message regex; treat `navigator.onLine` as a hint, not a verdict.
6. Add a global `unhandledrejection` handler that routes to the existing toast + Sentry.

---

## P2-1 · CI & tooling

- `pages.yml` (GitHub Pages) builds and deploys with **no type-check and no tests**;
  `deploy.yml` marks the Cloudflare deploy step `continue-on-error: true` (line 45), so a
  failed deploy still shows green. Every push to `main` double-deploys (Cloudflare + GH
  Pages) — pick one as canonical or make GH Pages explicitly experimental.
- GH Pages ignores `public/_headers`, so that deployment loses COOP/COEP and threaded WASM
  silently degrades there. Either drop the GH Pages target or accept/document
  single-threaded WASM on it. No `Cache-Control` headers exist for hashed assets on
  either target.
- **No linter or formatter** existed before this plan. The PR introducing this document
  also adds a MegaLinter (javascript flavor) CI workflow
  (`.github/workflows/mega-linter.yml` + `.mega-linter.yml`) as a first step.
  Recommendation for local dev remains Biome (one fast tool for lint+format), wired into
  `.husky/pre-commit` (currently runs only `npm test`).
- `tsconfig.json` is `strict` but lacks `noUnusedLocals`, `noUnusedParameters`,
  `noImplicitReturns`, `noUncheckedIndexedAccess`; `declaration: true` emits unused
  `.d.ts` for an app and can be removed.
- Coverage is collected but has no thresholds (`vite.config.ts:49-52`); the bench harness
  (`src/compute.bench.ts`) exists but no CI regression check uses it.
- `@types/node` is `^20` while CI runs Node 24 (`deploy.yml:28`) — align.

## P2-2 · Docs cleanup

- `README.md` says "🆕 v2.3.0 New Features" while `package.json` is `3.0.0`; lists a
  nonexistent `src/similarity.ts` in Project Structure (the logic lives in
  `src/compute.ts`); documents a **Theme setting that does not exist** (no theme field in
  `Settings`, `src/types.ts:36-49` — theming is purely `prefers-color-scheme`); and calls
  EXIF a "placeholder" although exifr is fully wired (`src/app.ts:6`, parse at
  `:2204-2210`, dialog at `:2327`).
- `EXIF_METADATA_PLAN.md` plans work that has shipped (and names `exifreader` while the
  code uses `@modernized/exifr`) — delete it or mark it done.
- `AGENT.md` says "Multimodal Nomic embeddings" but the default model is sapiens2 ONNX
  (`modelVariant: 'sapiens2-fp16'`, `src/app.ts:181`); the Chrome AI describe path is
  undocumented in both docs.
- `__APP_VERSION__` (`vite.config.ts:40`) is never shown in the UI (which renders only
  `__GIT_BRANCH__@__GIT_COMMIT__`), but it **is** used as the Sentry release tag
  (`src/sentry.ts:12`, declared in `src/types.ts:163`) — so it must not be dropped
  without refactoring both files. Recommendation: surface it in the UI/debug overlay so
  the version users see matches the release Sentry tracks.

## P2-3 · UX & accessibility

- The canvas (`index.html:679`) has no `role`, `aria-label`, `tabindex`, or fallback
  content; all keyboard handling is document-level with no focus affordance.
- Breadcrumbs and modal links are `<span onclick>` (`src/app.ts:2062-2178`,
  `index.html:718`) — not focusable, not keyboard-activatable; convert to `<button>`/`<a>`
  with proper roles.
- Only three `aria-label`s exist in the whole page; icon-only buttons (nav arrows, ✕,
  search clear, EXIF) rely on `title=` alone.
- The keydown guard excludes `INPUT` but not `TEXTAREA` (`src/app.ts:3069`) — typing
  w/a/s/d in the Chrome-AI prompt textarea pans the canvas.
- No `prefers-reduced-motion` handling; fixed pixel font sizes in inline styles don't
  respect user font settings.
- (The full-res object-URL leak formerly listed here is a memory/performance issue, not
  a UX/a11y one — it now lives with the P1-1 micro-optimizations above.)

---

## P3 · Brief entries

- **Testing**: `src/app.test.ts` never imports `app.ts` — it re-implements the logic it
  claims to test (debounce redefined at `:140`, search scoring at `:92`), so the 3,104-line
  core has near-zero real coverage; only `boot.test.ts` imports the real module. No tests
  exist for `sapiens2.ts` (the **default** embedder), `chromeAI.ts`, rendering,
  projections, navigation, URL state, or the resume flow. No E2E/Playwright — jsdom
  cannot exercise canvas/WebGPU. This becomes far more tractable after the P1-2 split;
  plan real tests module-by-module as they are extracted.
- **Privacy/telemetry**: the Sentry DSN is hardcoded (`src/sentry.ts:11`) and
  `captureConsoleIntegration` ships every `console.error` — which in this app frequently
  contains file names/paths — unless the user set the do-not-track setting. Move the DSN
  to env config, scrub file paths from breadcrumbs/messages, and make the opt-out
  discoverable. And rotate the committed Unsplash key (see security note at top).

---

## Suggested sequencing

| Step | Work                                         | Why this order                                             |
|------|----------------------------------------------|------------------------------------------------------------|
| 0    | Rotate Unsplash key; docs quick fixes (P2-2) | Security + cheap wins, no code risk                        |
| 1    | Projection worker (P1-1 part 1)              | Biggest user-visible freeze removed                        |
| 2    | Cache versioning + zero-vector fix (P1-3)    | Correctness before more refactoring builds on the cache    |
| 3    | Monolith split in stages (P1-2)              | Enables real tests; do after worker boundary is known      |
| 4    | CI/tooling hardening (P2-1)                  | Lint/coverage gates protect the newly split modules        |
| 5    | Accessibility pass (P2-3)                    | Touches modal/breadcrumb code that step 3 just reorganized |

Testing improvements (P3) ride along with each step rather than being a separate phase.
