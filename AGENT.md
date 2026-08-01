# Agent Guide: Media Clusterer

## Tech Stack
- **Frontend**: Vite (v8+), TypeScript, Vanilla CSS
- **AI**: Transformers.js (v4.2.0, Multimodal Nomic embeddings)
- **Projections**: DruidJS (UMAP, t-SNE, PCA, Isomap, LLE, MDS, Sammon, TriMap)
- **Database**: IndexedDB (for vector caching)
- **Deployment**: Cloudflare Pages

## Core Workflows

### Build & Deploy
- Build: `npm run build`
- Deploy: `npm run deploy` (requires `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` in `.env`)
- Type-check: `npm run type-check`

### Testing
- Run all tests: `npm test`
- Coverage: `npm run test:coverage`

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
