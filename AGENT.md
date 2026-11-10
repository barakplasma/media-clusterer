# Agent Guide: Media Clusterer

## Tech Stack

- **Frontend**: Vite (v8+), TypeScript, Vanilla CSS
- **AI**: Three interchangeable embedding backends selected by `settings.modelVariant`:
  - `sapiens2-*` (`sapiens2-fp16`) — `src/sapiens2.ts`, raw onnxruntime-web
  - `nomic` — Transformers.js (v4.2.0) multimodal Nomic embeddings
  - `chrome-ai` — `src/chromeAI.ts`, Chrome Prompt API caption → `nomic-embed-text` vector
  - `embeddinggemma2` (**default**) and `smolvlm2-*` — worker-backed tiers in `src/vlmTiers.ts` /
    `src/vlmWorker.ts`. EmbeddingGemma 2 embeds images, video (as one clip) and text queries into one
    space, so it answers text search itself (query prefix `task: search result | query:`, media unprefixed).

  All three produce 768-dim L2-normalized vectors.
- **Projections**: DruidJS (UMAP, t-SNE, PCA, Isomap, LLE, MDS, Sammon, TriMap)
- **Database**: IndexedDB (for vector caching)
- **Deployment**: Cloudflare Pages

## Documentation

- **Architecture decisions** live in `docs/adr/` — see `docs/adr/README.md` for the convention.
  Write an ADR when a change is hard to reverse or constrains later work.
- **Feature plans** live in root-level `*_PLAN.md` files.
- `AURA_LESSONS.md` holds cross-repository findings from the sibling
  [Aura](https://github.com/barakplasma/Aura) project, which runs the same
  Transformers.js/WebGPU stack against a live camera. Two of its items propose changes to
  decisions already recorded in `docs/adr/`.

## Core Workflows

### Build & Deploy

- Build: `npm run build`
- Deploy: `npm run deploy` (requires `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` in `.env`)
- Type-check: `npm run type-check`

### Testing

End to end only (no unit tests). Needs Google Chrome installed.

- `npm test`: the CPU-only checks (viewer mode; WebGPU-only model falling back with no adapter; the first-visit setup wizard). Pre-commit hook and the deploy job run these.
- `npm run test:gpu`: EmbeddingGemma 2 in the phone (light image budget, the Chromebook/Pixel profile) and desktop profiles. Needs a real GPU. CI runs the desktop profile as `e2e (webgpu)` on GitHub's macOS runner (`.github/workflows/e2e.yml`; the phone profile is for local runs). `SWIFTSHADER=<chrome binary>` runs it on SwiftShader for local debugging.
- `npx playwright test --project=webnn`: the same test on the WebNN path (`src/vlmWebnn.ts`), which the app prefers whenever the browser has WebNN (the project turns the flag on, and the test fails if the app did not end up on WebNN). Downloads the 880 MB fp16 model. CI runs it on the macOS runner as `e2e (webnn)` (Chrome there has a real WebNN backend). `npu-test.html` measures that path on a real phone and copies a JSON report with its JSON Schema.
- Chrome AI: `npx playwright test --project=chrome-ai` (Gemini Nano downloads on first run, about 90 s on the hosted Linux runner; run under `xvfb-run` there). On demand: `.github/workflows/chrome-ai.yml` (Run workflow button; also runs on PRs that change the test).
- Faces: `npx playwright test --project=faces` (on demand; downloads the ~10 MB YuNet/SFace models and OpenCV's reference image from Hugging Face). Checks the match against OpenCV's own result and the People strip end to end.
- Safari: `e2e/safari/` (node:test + safaridriver), one CI job on macOS 26.
- Real phone: `.github/workflows/testlab.yml`, Firebase Test Lab, on demand only (Run workflow button). Setup steps are at the top of that file.

## Coding Standards

- **Types**: Always use strict TypeScript. Define interfaces in `src/types.ts`. Avoid `any` and `unknown` in 99% of circumstances; prefer precise types or generics.
- **Logic**: Use the `IProjection` interface for dimensionality reduction algorithms.
- **State**: Centralized in the `state` object in `src/app.ts`.
- **Storage**: Use `src/db.ts` for IndexedDB operations.
- **Secrets**: API keys never enter `state`, `mc_settings`, a URL, or `console.error` — the last of these
  is shipped to a remote error sink by `captureConsoleIntegration` (`src/sentry.ts:15`). See
  [ADR-0002](docs/adr/0002-openai-compatible-remote-inference.md).
- **Embedding space**: index-time and query-time vectors must come from the same embedder under that
  model's own role-prefix scheme — which for nomic means the prefixes deliberately *differ* by role
  (`search_document:` when indexing, `src/app.ts:1590`; `search_query:` when querying, `src/app.ts:627`),
  and for a model with no such scheme means no prefix on either side. Never mix schemes across the two
  sides. Any change of embedder, model, dimension, or of an input that alters what gets embedded must
  change the IndexedDB cache namespace (`currentCachePrefix`, `src/app.ts:354`).
  The overnight quick video pass is one such change: a video embedded from `DRAFT_VIDEO_FRAMES` frames lives
  under `videoDraftPrefix(variant)` (the tier's prefix plus `-f2`), never under the full one. Opening a folder
  or resuming reads a video's full vector first and its quick one if that is all there is
  (`readCachedEmbeddings`); only overnight worker frames skip that fallback, so the bonus pass can replace them.

## Accelerator choice

EmbeddingGemma 2 can run on WebGPU (q4) or on WebNN (fp16: npu, gpu or cpu). `src/benchStore.ts` keeps every speed
measured in this browser under localStorage `mc_bench`: `npu-test.html` runs (including its hybrid run, two devices
at once: `?hybrid=webgpu,cpu`, where `cpu` is WASM on a worker and WebNN is only touched by a `webnn-*` device or `?diag`) and the app's own embedding runs (`recordAppBench`, median ms per photo).
At start-up the setup wizard asks `fastestAccelerator` which of the accelerators on this device is fastest, by the
median of its runs, and orders WebNN accordingly (WebNN off when WebGPU is fastest). Until two accelerators have
been measured it keeps the default: WebNN first, most native device first. A speed only compares with the same model
file (`dtype`), so the benchmark page's fp16 WebGPU run never stands in for the app's q4 one.

## Architecture

### Application Modes

1. **AI Mode** (default): Loads embeddings, runs projections, enables semantic search
2. **Viewer-Only Mode**: Skips AI, arranges by folder/date grid, no search
3. **Remote AI Mode** (proposed, opt-in): inference runs against a user-configured OpenAI-compatible
   endpoint instead of in the browser — downscaled JPEG thumbnails are uploaded. See
   [ADR-0002](docs/adr/0002-openai-compatible-remote-inference.md) and `REMOTE_INFERENCE_PLAN.md`.

### State Management

- All state in `state` object (phase, files, vectors, points, clusters, thumbnails, settings)
- URL state managed via History API (`#folder:...`, `#dt:...`)
- Session persistence via localStorage (resume capability)

### Resource Management

- Object URLs created lazily via `URL.createObjectURL(file)` in `lazyDecodeThumbnail`
- Resources MUST be cleaned up before processing new files:
  - Close ImageBitmaps: `bmp?.close()`
  - Revoke object URLs: `URL.revokeObjectURL(url)`
  - Clear caches: `thumbDecoding`, `thumbnailLRU`

### Navigation

- **Folder breadcrumbs**: Click path segments → `navigateToFolder(path)`
- **Datetime breadcrumbs**: Click datetime parts → `filterByDateTime(...)`
- **Back/Forward**: Handled via `popstate` event listener
- **n/p keys**: Navigate in datetime order (sorted by `lastModified`)

## Key Functions

### File Processing

- `collectImages(dirHandle, sampleSize, basePath)` - Walk directory tree, apply reservoir sampling if sampleSize > 0
- `processFiles(files)` - Main entry point, handles both AI and viewer modes
- `lazyDecodeThumbnail(idx)` - Create object URL and decode thumbnail lazily

### Navigation

- `run(dirHandle, basePath)` - Load and process files from directory
- `navigateToFolder(targetPath)` - Navigate to subfolder (reuses currentDirHandle)
- `filterByDateTime(granularity, year, month, day, hour, minute)` - Filter and rescan full folder

### Modal

- `openFileModal(index)` - Show media with metadata in footer
- `closeModal()` - Hide modal, pause video
- `navigateModal(direction)` - Arrow buttons for grid navigation

## Deployment Details

- **Platform**: Cloudflare Pages
- **Project Name**: `media-clusterer`
- **Output Dir**: `dist/`
- **Production Branch**: `main`
- **Preview Branches**: PR branches get `https://<branch>.media-clusterer.pages.dev`

## GitHub Actions

- Runs on push to `main` and PRs to `main`
- Type-check, tests, build, deploy to Cloudflare Pages
- Comments on PR with preview URL and commit info
