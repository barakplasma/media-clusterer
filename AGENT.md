# Agent Guide: Media Clusterer

## Tech Stack
- **Frontend**: Vite (v8+), TypeScript, Vanilla CSS
- **AI**: Transformers.js (Multimodal Nomic embeddings)
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

## Deployment Details
- **Platform**: Cloudflare Pages
- **Project Name**: `media-clusterer`
- **Output Dir**: `dist/`
- **Production Branch**: `main`
