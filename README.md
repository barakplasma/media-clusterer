# Photo Organizer

A zero-server, zero-upload photo organizer that runs entirely in your browser.

Pick a local folder of images → the app embeds them with a local AI vision model on your GPU → projects them into a 2D map where visually similar photos cluster together → explore on an infinite canvas.

**Live**: https://photo-organizer.526462738.xyz  
**Mirror**: https://barakplasma.github.io/photo-organizer/

## How it works

1. **Load Model** — downloads `nomic-ai/nomic-embed-vision-v1.5` (≈380 MB, cached after first load) and runs it on WebGPU (falls back to CPU if unavailable)
2. **Open Folder** — pick any local folder; images are embedded in batches of 8, embeddings cached in IndexedDB so re-opening the same folder is instant
3. **Explore** — a 2D UMAP projection clusters visually similar images; tap any image to view full-size; pan with one finger, pinch to zoom
4. **Resume** — after a successful run, reload the page and tap "Resume last session" to restore the canvas without re-embedding or re-running UMAP

## Privacy

Nothing leaves your device. No server, no upload, no analytics. The AI model runs locally via [transformers.js](https://huggingface.co/docs/transformers.js) on WebGPU.

## Requirements

- Chrome 113+ or Edge 113+ (WebGPU + File System Access API)
- Works best with a GPU; falls back to CPU (slower embedding)

## Stack

| Part | What |
|---|---|
| Language | TypeScript (Strict, no `any`) |
| Build Tool | [Vite](https://vitejs.dev/) |
| Test Runner | [Vitest](https://vitest.dev/) |
| Embeddings | `nomic-ai/nomic-embed-vision-v1.5` and `nomic-ai/nomic-embed-text-v1.5` via `@huggingface/transformers@3` |
| Dimensionality reduction | [umap-js](https://github.com/PAIR-code/umap-js) v1.4.0 |
| Clustering | k-means on 2D UMAP output |
| Rendering | HTML5 Canvas with camera pan/zoom, pointer events |
| Cache | IndexedDB for embeddings, localStorage for session UMAP layout |

## Project Structure

```
index.html       — main entry point
src/
  app.ts         — main application logic
  db.ts          — IndexedDB utilities
  embeddings.ts  — embedding and normalization logic
  similarity.ts  — vector similarity search
  types.ts       — core type definitions
public/
  umap-js.min.js — static asset for dimensionality reduction
```

## Development

This project uses Vite for development and Vitest for testing.

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Type check
npm run type-check

# Build for production
npm run build
```
