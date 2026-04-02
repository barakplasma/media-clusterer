# Photo Organizer — Claude Instructions

## Project layout

```
index.html          — entire frontend app (single file, no build step)
umap-js.min.js      — umap-js v1.4.0 UMD bundle
k8s/                — Kubernetes manifests for the self-hosted deployment
README.md
CLAUDE.md
```

## Editing the app

- **Single file**: all HTML, CSS, and JS is in `index.html`. No bundler, no npm.
- After editing `index.html`, the k8s deployment at `photo-organizer.526462738.xyz` serves it immediately (hostPath volume points here).
- GitHub Pages (`barakplasma.github.io/photo-organizer`) deploys on every push to `main`.
- Always commit and push after changes so both deployments stay in sync.

## Deploying

```bash
# k8s picks up changes instantly (hostPath mount) — no restart needed
# For GitHub Pages:
cd /root/photo-organizer
git add -A && git commit -m "..." && git push
```

## Architecture notes

### umap-js path
`umap-js.min.js` is referenced as a **relative path** (`umap-js.min.js`) so it works on both the k8s subdomain and GitHub Pages subpath.

### Coordinate systems
- **UMAP space**: raw float output from umap-js (arbitrary units, centred near origin)
- **World space**: UMAP after `spreadPoints()` — units are virtual pixels where `THUMB_WORLD = 48` per thumbnail
- **Screen space**: world transformed by `camera` (`x`, `y`, `scale`) → canvas pixels

### Key functions
| Function | File location | Purpose |
|---|---|---|
| `spreadPoints(umapPoints)` | index.html | Scale + repel UMAP output so thumbnails don't overlap |
| `fitCamera()` | index.html | Set initial camera to show all points with padding |
| `render()` | index.html | Draw all thumbnails with cluster-colour borders |
| `embedAll(files)` | index.html | Batch-embed images, check IndexedDB cache first |
| `loadModel()` | index.html | Singleton model init (WebGPU → wasm fallback) |
| `runUMAP(vectors, nNeighbors)` | index.html | Main-thread UMAP projection |

### Storage
| Store | Key prefix | Cleared by |
|---|---|---|
| IndexedDB `photo-organizer-v1` | `name:size:mtime` | "Clear site data" |
| localStorage | `po_fileKeys`, `po_umapPoints`, `po_clusters` | "Clear site data" |

### Canvas interaction
Pointer events (unified touch + mouse):
- 1 pointer drag → pan (`camera.x`, `camera.y`)
- 2 pointer pinch → zoom (`camera.scale`) with pivot at midpoint
- Tap (dragMoved < 8px) → find nearest image in world coords → open modal
- Mouse wheel → zoom at cursor

## k8s deployment

Namespace: `photo-organizer`  
Ingress: `cloudflare-tunnel` → `photo-organizer.526462738.xyz`  
Static server: Caddy 2 alpine  
Volume: hostPath `/root/photo-organizer` mounted at `/srv`

To update k8s resources:
```bash
kubectl apply -f k8s/
```

## Common tasks

**Re-embed from scratch** (user cleared IndexedDB): just re-open folder — embeddings recompute automatically.

**Model won't load**: check browser console for WebGPU errors. The app falls back to wasm automatically.

**UMAP constructor fails**: check `console.log('UMAP export:', ...)` — the fallback chain in `loadUMAPLib()` handles most umap-js export shapes.

**Thumbnails not rendering**: `preloadThumbnails()` creates Image objects but they load async. The canvas re-renders correctly once images decode — no fix needed.
