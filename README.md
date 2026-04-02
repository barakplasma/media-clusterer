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
| Embeddings | `nomic-ai/nomic-embed-vision-v1.5` via `@huggingface/transformers@4` |
| Dimensionality reduction | [umap-js](https://github.com/PAIR-code/umap-js) v1.4.0, main thread |
| Clustering | k-means on 2D UMAP output, k = min(8, ⌈√(n/2)⌉) |
| Rendering | HTML5 Canvas with camera pan/zoom, pointer events |
| Cache | IndexedDB for embeddings, localStorage for session UMAP layout |
| Hosting | Caddy on k3s + Cloudflare tunnel; GitHub Pages mirror |

## Files

```
index.html       — entire app, single self-contained file
umap-js.min.js   — umap-js v1.4.0 UMD bundle (served as static asset)
```

## Development

Edit `index.html` directly — no build step.

```bash
# Serve locally (requires HTTPS for File System Access API — use ngrok or caddy)
caddy file-server --browse --listen :8080
```

The k8s deployment at `photo-organizer.526462738.xyz` serves directly from this repo folder via a hostPath volume — saving `index.html` here is instantly live.
