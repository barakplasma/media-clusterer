/**
 * Main application logic for Media Clusterer
 */

import { pipeline, env, RawImage } from '@huggingface/transformers';
import * as druid from '@saehrimnir/druidjs';
import {
  l2normalize,
  extractVector,
  extractBatchedVectors,
} from './embeddings';
import {
  openDB,
  cacheGet,
  cacheGetBatch,
  cachePutBatch,
  cacheStats,
} from './db';
import type {
  AppState,
  Camera,
  DOMElements,
  PhotoFile,
  Point,
  Pipeline,
  PipelineInstance,
  ProgressEvent,
  Settings,
  ProjectionMethod,
  DirectoryHandle,
  FileSystemHandle,
  PointerState,
  CanvasPointerPos,
  CacheKey,
} from './types';

// Enable caching and local model access for persistent storage
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.cacheDir = 'models';

// ── Constants ────────────────────────────────────────────────────────────────
const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'gif', 'webp', 'avif', 'bmp', 'tiff', 'tif', 'heic', 'heif']);
const VIDEO_EXTS = new Set(['mp4', 'webm']);
const THUMB_WORLD = 48;   // thumbnail size in world units
const FULL_LOD_SIZE = 120;  // screen px at which we switch from thumb to full-res
const IS_MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const BATCH_SIZE = IS_MOBILE ? 4 : 16; // fallback before settings load
const WRITE_BATCH = 20;
const MAX_DRAW_PER_FRAME = IS_MOBILE ? 150 : 400;
const CLUSTER_COLORS = ['#f87171', '#fb923c', '#facc15', '#4ade80', '#38bdf8', '#818cf8', '#f472b6', '#a78bfa'];

// ── DOM Elements ─────────────────────────────────────────────────────────────
const dom: DOMElements = {
  loadModelBtn: document.getElementById('load-model-btn') as HTMLButtonElement,
  resumeBtn: document.getElementById('resume-btn') as HTMLButtonElement,
  openBtn: document.getElementById('open-btn') as HTMLButtonElement,
  recenterBtn: document.getElementById('recenter-btn') as HTMLButtonElement,
  resetBtn: document.getElementById('reset-btn') as HTMLButtonElement,
  progressBar: document.getElementById('progress-bar') as HTMLDivElement,
  statusEl: document.getElementById('status') as HTMLDivElement,
  canvas: document.getElementById('canvas') as HTMLCanvasElement,
  modal: document.getElementById('modal') as HTMLDivElement,
  modalClose: document.getElementById('modal-close') as HTMLButtonElement,
  modalImg: document.getElementById('modal-img') as HTMLImageElement,
  modalVideo: document.getElementById('modal-video') as HTMLVideoElement,
  modalName: document.getElementById('modal-name') as HTMLDivElement,
  searchWrap: document.getElementById('search-wrap') as HTMLDivElement,
  searchInput: document.getElementById('search-input') as HTMLInputElement,
  searchClearBtn: document.getElementById('search-clear-btn') as HTMLButtonElement,
  fileInput: document.getElementById('file-input') as HTMLInputElement,
  aboutBtn: document.getElementById('about-btn') as HTMLButtonElement,
  aboutModal: document.getElementById('about-modal') as HTMLDivElement,
  aboutClose: document.getElementById('about-close') as HTMLButtonElement,
  statsEl: document.getElementById('stats') as HTMLDivElement,
  settingsBtn: document.getElementById('settings-btn') as HTMLButtonElement,
  settingsModal: document.getElementById('settings-modal') as HTMLDivElement,
  settingsClose: document.getElementById('settings-close') as HTMLButtonElement,
  densitySlider: document.getElementById('density-slider') as HTMLInputElement,
  loopToggle: document.getElementById('loop-toggle') as HTMLInputElement,
  themeSelect: document.getElementById('theme-select') as HTMLSelectElement,
  drawBudgetSlider: document.getElementById('draw-budget-slider') as HTMLInputElement,
  enableSearchToggle: document.getElementById('enable-search-toggle') as HTMLInputElement,
  projectionSelect: document.getElementById('projection-select') as HTMLSelectElement,
  batchSizeInput: document.getElementById('batch-size-slider') as HTMLInputElement,
  batchSizeAutoBtn: document.getElementById('batch-size-auto-btn') as HTMLButtonElement,
  bottomPanel: document.getElementById('bottom-panel') as HTMLDivElement,
  headerRecenterBtn: document.getElementById('header-recenter-btn') as HTMLButtonElement,
  headerResetBtn: document.getElementById('header-reset-btn') as HTMLButtonElement,
  };

  // ── Version Info ─────────────────────────────────────────────────────────────
  const versionEl = document.getElementById('version-info');
  if (versionEl) {
  versionEl.textContent = `${__GIT_BRANCH__}@${__GIT_COMMIT__}`;
  }

  // ── Constants ────────────────────────────────────────────────────────────────
// ── Auto batch size ───────────────────────────────────────────────────────────
function computeOptimalBatchSize(avgFileSizeBytes = 2 * 1024 * 1024): number {
  // Estimate available memory headroom
  const deviceMemoryGB: number = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 2;

  // Chrome-only: use live heap headroom if available
  const perfMem = (performance as Performance & { memory?: { jsHeapSizeLimit: number; usedJSHeapSize: number } }).memory;
  const headroomBytes = perfMem
    ? Math.max(0, perfMem.jsHeapSizeLimit - perfMem.usedJSHeapSize)
    : deviceMemoryGB * 0.3 * 1024 * 1024 * 1024; // assume 30% of device RAM usable

  // Vision model resizes to ~224×224 RGBA + ~10× overhead for intermediate tensors
  const bytesPerImageInference = Math.max(avgFileSizeBytes * 0.5, 224 * 224 * 4 * 10);

  const optimal = Math.floor(headroomBytes / bytesPerImageInference);
  return Math.max(1, Math.min(32, optimal));
}

const DEFAULT_SETTINGS: Settings = {
  density: 1.0,
  loopVideos: true,
  theme: 'system',
  drawBudget: IS_MOBILE ? 150 : 400,
  enableTextSearch: false,
  projectionMethod: 'UMAP',
  batchSize: IS_MOBILE ? 4 : 16,
};

const savedSettings = localStorage.getItem('mc_settings');
const settings: Settings = savedSettings ? { ...DEFAULT_SETTINGS, ...JSON.parse(savedSettings) } : DEFAULT_SETTINGS;

const state: AppState = {
  phase: 'idle',
  files: [],
  vectors: [],
  points: [],
  rawPoints: null,
  clusters: null,
  thumbnails: [],
  searchResults: null,
  searchQuery: '',
  searchScores: null,
  settings,
};

// ── Camera (infinite canvas) ─────────────────────────────────────────────────
const camera: Camera = { x: 0, y: 0, scale: 1 };

// ── Model singleton ──────────────────────────────────────────────────────────
let extractor: PipelineInstance | null = null;      // Vision model (for images)
let textExtractor: PipelineInstance | null = null;  // Text model (for search queries)

// ── Helpers ──────────────────────────────────────────────────────────────────
const setStatus = (msg: string) => { dom.statusEl.textContent = msg; };
const setProgress = (pct: number) => { dom.progressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`; };
const yieldMain = () => new Promise(resolve => setTimeout(resolve, 0));

const cacheSizeEl = document.getElementById('cache-size');
async function refreshCacheSize() {
  if (!cacheSizeEl) return;
  try {
    const { count, bytes } = await cacheStats();
    if (count === 0) { cacheSizeEl.textContent = ''; return; }
    const mb = bytes / (1024 * 1024);
    const display = mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
    cacheSizeEl.textContent = `${count} cached · ${display}`;
  } catch {
    cacheSizeEl.textContent = '';
  }
}

// ── Render scheduling (debounce to one frame) ────────────────────────────────
let renderPending = false;
function scheduleRender() {
  if (!renderPending) {
    renderPending = true;
    requestAnimationFrame(() => {
      render();
      renderPending = false;
    });
  }
}

// ── File collection ──────────────────────────────────────────────────────────
async function collectImages(dirHandle: DirectoryHandle): Promise<PhotoFile[]> {
  const files: PhotoFile[] = [];
  async function walk(handle: DirectoryHandle, prefix: string) {
    for await (const [name, entry] of handle.entries()) {
      if (name.startsWith('.')) continue;
      if (entry.kind === 'directory') {
        await walk(entry, `${prefix}${name}/`);
      } else {
        const ext = name.split('.').pop()?.toLowerCase() ?? '';
        if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
          const file = await entry.getFile();
          files.push({
            name: `${prefix}${name}`,
            size: file.size,
            lastModified: file.lastModified,
            file,
            objectURL: null
          });
        }
      }
    }
  }
  await walk(dirHandle, '');
  return files;
}

// ── Media helpers ────────────────────────────────────────────────────────────
async function extractVideoFrame(file: File): Promise<ImageBitmap | null> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;
    const url = URL.createObjectURL(file);
    
    video.onloadedmetadata = () => {
      video.currentTime = Math.min(1.0, video.duration / 2); // Get a frame from near the start
    };
    
    video.onseeked = async () => {
      try {
        const bitmap = await createImageBitmap(video, { resizeWidth: 96, resizeQuality: 'low' });
        resolve(bitmap);
      } catch (e) {
        console.warn('Failed to extract video frame:', e);
        resolve(null);
      } finally {
        URL.revokeObjectURL(url);
        video.remove();
      }
    };
    
    video.onerror = () => {
      console.warn('Video load error');
      URL.revokeObjectURL(url);
      video.remove();
      resolve(null);
    };
    
    video.src = url;
  });
}

// ── Thumbnail preloader ──────────────────────────────────────────────────────
function initThumbnails(files: PhotoFile[]): (ImageBitmap | null)[] {
  // Thumbnails are decoded lazily on first render — no upfront work here
  for (const f of files) {
    if (!f.objectURL) f.objectURL = URL.createObjectURL(f.file);
  }
  return new Array<ImageBitmap | null>(files.length).fill(null);
}

function lazyDecodeThumbnail(idx: number) {
  if (thumbDecoding.has(idx) || state.thumbnails[idx]) return;
  thumbDecoding.add(idx);
  const f = state.files[idx];
  const ext = f.name.split('.').pop()?.toLowerCase() ?? '';

  const done = (bm: ImageBitmap | null) => {
    state.thumbnails[idx] = bm;
    thumbDecoding.delete(idx);
    scheduleRender();
  };

  if (VIDEO_EXTS.has(ext)) {
    extractVideoFrame(f.file).then(done).catch(() => done(null));
  } else {
    createImageBitmap(f.file, { resizeWidth: 96, resizeQuality: 'low' })
      .then(done).catch(() => done(null));
  }
}

// ── Text embedding (uses text model with search_query prefix) ────────────────
async function embedText(text: string): Promise<Float64Array> {
  if (!textExtractor) throw new Error('Text model not loaded');

  // Add required task prefix for nomic-embed-text
  const prefixed = `search_query: ${text}`;
  const output = await textExtractor(prefixed, { pooling: 'mean', normalize: true });

  return Float64Array.from(output.data);
}

// ── Text search using DruidJS HNSW ───────────────────────────────────────────
async function searchImages(query: string) {
  if (!query.trim() || !state.vectors.length) {
    state.searchResults = null;
    state.searchQuery = '';
    state.searchScores = null;
    dom.searchWrap.classList.remove('loading');
    return;
  }

  dom.searchWrap.classList.add('loading');
  setStatus(`Searching for "${query}"…`);

  try {
    const queryVector = await embedText(query);
    state.searchQuery = query;

    if (!state.hnsw && state.vectors.length > 0) {
      state.hnsw = new druid.HNSW(state.vectors, { metric: druid.cosine } as ConstructorParameters<typeof druid.HNSW>[1]);
    }

    if (state.hnsw) {
      const k = state.vectors.length;
      const results = state.hnsw.search(queryVector, k);

      const scores = new Float32Array(state.vectors.length);
      const indices = new Int32Array(results.length);

      for (let i = 0; i < results.length; i++) {
        const { index, distance } = results[i];
        indices[i] = index;
        scores[index] = 1 - distance;
      }

      state.searchScores = scores;
      state.searchResults = indices;

      const topScore = scores[indices[0]] ?? 0;
      const statusMsg = `${state.vectors.length} media files · top match: ${(topScore * 100).toFixed(0)}% similar`;
      setStatus(statusMsg);
      if (dom.statsEl) dom.statsEl.textContent = statusMsg;
    }
  } catch (err) {
    console.error('Search failed:', err);
    setStatus('Search failed. Check console.');
  } finally {
    dom.searchWrap.classList.remove('loading');
  }
}

// ── k-means (runs on 2D UMAP output) ─────────────────────────────────────────
async function kmeansAsync(points: number[][], k: number, maxIter = 60): Promise<Int32Array> {
  const n = points.length;
  if (n === 0) return new Int32Array(0);
  k = Math.min(k, n);

  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const centroids = idx.slice(0, k).map(i => [points[i][0], points[i][1]]);
  const labels = new Int32Array(n);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      const px = points[i][0], py = points[i][1];
      for (let j = 0; j < k; j++) {
        const dx = px - centroids[j][0], dy = py - centroids[j][1];
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = j; }
      }
      if (labels[i] !== best) { labels[i] = best; changed = true; }
    }

    if (iter % 10 === 0) await yieldMain();
    if (!changed) break;

    const sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      sx[labels[i]] += points[i][0];
      sy[labels[i]] += points[i][1];
      cnt[labels[i]]++;
    }
    for (let j = 0; j < k; j++) {
      if (cnt[j]) { centroids[j][0] = sx[j] / cnt[j]; centroids[j][1] = sy[j] / cnt[j]; }
    }
  }
  return labels;
}

// ── Canvas ───────────────────────────────────────────────────────────────────
const fullImages = new Map<number, HTMLImageElement>(); // index → HTMLImageElement
const thumbDecoding = new Set<number>();               // indices currently being decoded

function resizeCanvas() {
  const wrap = dom.canvas.parentElement;
  if (!wrap) return;
  dom.canvas.width = wrap.clientWidth || 800;
  dom.canvas.height = wrap.clientHeight || 600;
}

async function spreadPointsAsync(projectedPoints: number[][]): Promise<Point[]> {
  const n = projectedPoints.length;
  if (n === 0) return [];

  // Normalize to zero-centered unit space
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of projectedPoints) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const r = Math.max(maxX - cx, maxY - cy) || 1;

  const vsize = Math.sqrt(n) * THUMB_WORLD * 1.4 * state.settings.density;
  const pts: [number, number][] = projectedPoints.map(([x, y]) => [(x - cx) / r * vsize, (y - cy) / r * vsize]);

  const CELL = THUMB_WORLD * 2 * state.settings.density;
  for (let iter = 0; iter < 60; iter++) {
    const grid = new Map<string, number[]>();
    for (let i = 0; i < n; i++) {
      const gx = Math.floor(pts[i][0] / CELL), gy = Math.floor(pts[i][1] / CELL);
      const key = `${gx},${gy}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key)!.push(i);
    }
    let moved = false;
    for (let i = 0; i < n; i++) {
      const gx = Math.floor(pts[i][0] / CELL), gy = Math.floor(pts[i][1] / CELL);
      for (let dgx = -1; dgx <= 1; dgx++) for (let dgy = -1; dgy <= 1; dgy++) {
        for (const j of (grid.get(`${gx + dgx},${gy + dgy}`) ?? [])) {
          if (j <= i) continue;
          const dx = pts[j][0] - pts[i][0], dy = pts[j][1] - pts[i][1];
          const d2 = dx * dx + dy * dy;
          if (d2 < THUMB_WORLD * THUMB_WORLD && d2 > 0) {
            const dist = Math.sqrt(d2);
            const push = (THUMB_WORLD - dist) / 2 + 0.1;
            const nx = dx / dist, ny = dy / dist;
            pts[i][0] -= nx * push; pts[i][1] -= ny * push;
            pts[j][0] += nx * push; pts[j][1] += ny * push;
            moved = true;
          }
        }
      }
    }
    if (iter % 5 === 0) await yieldMain();
    if (!moved) break;
  }
  return pts as Point[];
}

function fitCamera() {
  const pts = state.points;
  if (!pts.length) return;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const pad = THUMB_WORLD + 16;
  const scX = (dom.canvas.width - pad * 2) / ((maxX - minX) || 1);
  const scY = (dom.canvas.height - pad * 2) / ((maxY - minY) || 1);
  camera.x = (minX + maxX) / 2;
  camera.y = (minY + maxY) / 2;
  camera.scale = Math.min(scX, scY);
}

function resetAll() {
  for (const bmp of state.thumbnails) { if (bmp?.close) bmp.close(); }
  for (const img of fullImages.values()) img.src = '';
  fullImages.clear();
  thumbDecoding.clear();
  for (const f of state.files) {
    if (f.objectURL) { URL.revokeObjectURL(f.objectURL); f.objectURL = null; }
  }
  state.phase = 'idle'; state.files = []; state.vectors = [];
  state.points = []; state.clusters = null; state.thumbnails = [];
  state.searchResults = null;
  state.searchQuery = '';
  state.searchScores = null;
  state.hnsw = undefined;
  localStorage.removeItem('po_fileKeys');
  localStorage.removeItem('po_umapPoints');
  localStorage.removeItem('po_clusters');
  camera.x = 0; camera.y = 0; camera.scale = 1;
  const ctx = dom.canvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
  dom.recenterBtn.disabled = true;
  dom.resetBtn.disabled = true;
  dom.headerRecenterBtn.disabled = true;
  dom.headerResetBtn.disabled = true;
  dom.searchInput.disabled = true;
  dom.searchInput.value = '';
  dom.searchClearBtn.hidden = true;
  dom.resumeBtn.hidden = true;
  dom.resumeBtn.disabled = true;
  dom.resumeBtn.classList.remove('primary');
  dom.openBtn.disabled = false;
  setProgress(0);
  setStatus('Cleared. Open a folder to start.');
  refreshCacheSize();
}

function render() {
  const ctx = dom.canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height);
  const pts = state.points;
  const thumbs = state.thumbnails;
  if (!pts.length) return;

  const rank = new Int32Array(pts.length).fill(pts.length);
  if (state.searchResults) {
    for (let i = 0; i < state.searchResults.length; i++) {
      rank[state.searchResults[i]] = i;
    }
  }

  const s = camera.scale;
  const cxW = dom.canvas.width / 2;
  const cyW = dom.canvas.height / 2;
  const drawSize = THUMB_WORLD * s;
  const half = drawSize / 2;
  const useFull = drawSize > FULL_LOD_SIZE;
  const drawnFull = useFull ? new Set<number>() : null;

  // Frustum culling and visible list
  const visibleIndices: number[] = [];
  for (let i = 0; i < pts.length; i++) {
    const sx = (pts[i][0] - camera.x) * s + cxW;
    const sy = (pts[i][1] - camera.y) * s + cyW;

    if (sx + half >= 0 && sx - half <= dom.canvas.width &&
      sy + half >= 0 && sy - half <= dom.canvas.height) {
      visibleIndices.push(i);
    }
  }

  // Prioritize drawing: search results first, then by distance to center
  const budget = state.settings.drawBudget;
  if (visibleIndices.length > budget) {
    visibleIndices.sort((a, b) => {
      // Search results always first
      if (state.searchResults) {
        const ra = rank[a], rb = rank[b];
        if (ra < 20 || rb < 20) return ra - rb;
      }
      // Then by distance to camera center
      const da = (pts[a][0] - camera.x)**2 + (pts[a][1] - camera.y)**2;
      const db = (pts[b][0] - camera.x)**2 + (pts[b][1] - camera.y)**2;
      return da - db;
    });
    visibleIndices.length = budget;
  }

  for (const i of visibleIndices) {
    const sx = (pts[i][0] - camera.x) * s + cxW;
    const sy = (pts[i][1] - camera.y) * s + cyW;

    let alphaMultiplier = 1.0;
    let highlightBorder = null;
    if (state.searchResults) {
      const r = rank[i];
      if (r < 20) {
        highlightBorder = '#facc15'; // Bright yellow for top results
        alphaMultiplier = 1.0;
      } else if (r < 100) {
        alphaMultiplier = 0.4;
      } else {
        alphaMultiplier = 0.1;
      }
    }

    const clr = highlightBorder ?? (state.clusters?.length ? CLUSTER_COLORS[state.clusters[i] % CLUSTER_COLORS.length] : '#6b7280');
    let drawn = false;

    if (useFull && drawnFull) {
      drawnFull.add(i);
      let fullImg = fullImages.get(i);
      if (!fullImg) {
        fullImg = new Image();
        fullImg.onload = () => {
          if (fullImg) {
            fullImg.decode().then(() => scheduleRender()).catch(() => scheduleRender());
          }
        };
        fullImg.src = state.files[i].objectURL || '';
        fullImages.set(i, fullImg);
      }
      if (fullImg.complete && fullImg.naturalWidth > 0) {
        const ratio = fullImg.naturalWidth / fullImg.naturalHeight;
        const dw = ratio >= 1 ? drawSize : drawSize * ratio;
        const dh = ratio >= 1 ? drawSize / ratio : drawSize;
        ctx.globalAlpha = 0.9 * alphaMultiplier;
        ctx.drawImage(fullImg, sx - dw / 2, sy - dh / 2, dw, dh);
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = clr;
        ctx.lineWidth = Math.max(1.5, 2 * Math.min(s, 1));
        ctx.strokeRect(sx - dw / 2, sy - dh / 2, dw, dh);
        drawn = true;
      }
    }

    if (!drawn) {
      const thumb = thumbs[i];
      if (!thumb) lazyDecodeThumbnail(i);
      if (thumb && thumb.width > 0) {
        const ratio = thumb.width / thumb.height;
        const dw = ratio >= 1 ? drawSize : drawSize * ratio;
        const dh = ratio >= 1 ? drawSize / ratio : drawSize;
        ctx.globalAlpha = 0.9 * alphaMultiplier;
        ctx.drawImage(thumb, sx - dw / 2, sy - dh / 2, dw, dh);
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = clr;
        ctx.lineWidth = Math.max(1.5, 2 * Math.min(s, 1));
        ctx.strokeRect(sx - dw / 2, sy - dh / 2, dw, dh);
      } else {
        const r = Math.max(3, half * 0.3);
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fillStyle = clr;
        ctx.globalAlpha = 0.8;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
  }

  if (useFull && drawnFull) {
    for (const [idx, img] of fullImages) {
      if (!drawnFull.has(idx)) { img.src = ''; fullImages.delete(idx); }
    }
  } else {
    for (const img of fullImages.values()) img.src = '';
    fullImages.clear();
  }
}

// ── Projection ───────────────────────────────────────────────────────────────
async function runProjection(vectors: Float64Array[], method: ProjectionMethod, nNeighbors: number): Promise<number[][]> {
  try {
    setStatus(`Projecting with ${method}…`);
    // Druid expects data as Array of Arrays or a Matrix
    const data = vectors.map(v => Array.from(v));
    const matrix = druid.Matrix.from(data);
    let result: druid.Matrix;

    // Small yield to allow UI update
    await yieldMain();

    switch (method) {
      case 'TSNE':
        result = new druid.TSNE(matrix, { d: 2, perplexity: Math.min(30, Math.floor(vectors.length / 3)) }).transform();
        break;
      case 'PCA':
        result = new druid.PCA(matrix, { d: 2 }).transform();
        break;
      case 'ISOMAP':
        result = new druid.ISOMAP(matrix, { d: 2, neighbors: nNeighbors }).transform();
        break;
      case 'LLE':
        result = new druid.LLE(matrix, { d: 2, neighbors: nNeighbors }).transform();
        break;
      case 'MDS':
        result = new druid.MDS(matrix, { d: 2 }).transform();
        break;
      case 'SAMMON':
        result = new druid.SAMMON(matrix, { d: 2 }).transform();
        break;
      case 'TriMap':
        result = new druid.TriMap(matrix, { d: 2 }).transform();
        break;
      case 'UMAP':
      default:
        result = new druid.UMAP(matrix, { d: 2, n_neighbors: nNeighbors, local_connectivity: 1 }).transform();
        break;
    }

    return result.to2dArray().map(row => Array.from(row));
  } catch (err) {
    setStatus(`${method} projection failed: ${(err as Error).message}`);
    throw err;
  }
}

// ── Model singleton ──────────────────────────────────────────────────────────
async function loadModel() {
  if (extractor) return;
  state.phase = 'loading_model';
  setStatus('Loading model…');
  setProgress(0);

  const MODEL_SIZE_BYTES = 380 * 1024 * 1024 + 134 * 1024 * 1024;
  const loaded = new Map<string, number>();

  const progressCb = (e: ProgressEvent) => {
    if (e.status === 'progress') {
      loaded.set(e.file, e.loaded ?? 0);
      const total = [...loaded.values()].reduce((a, b) => a + b, 0);
      const pct = Math.min(99, (total / MODEL_SIZE_BYTES) * 100);
      setProgress(pct);
      setStatus(`Loading model… ${pct.toFixed(0)}%`);
    }
  };

  const tryLoad = (device: 'webgpu' | 'wasm') => (pipeline as Pipeline)(
    'image-feature-extraction',
    'nomic-ai/nomic-embed-vision-v1.5',
    { device, dtype: 'fp32', progress_callback: progressCb, pooling: 'mean', normalize: true }
  ) as Promise<PipelineInstance>;

  try {
    extractor = await tryLoad('webgpu');
  } catch (gpuErr) {
    console.warn('WebGPU init failed, falling back to wasm:', gpuErr);
    setStatus('WebGPU unavailable — using CPU (slower)…');
    loaded.clear();
    try {
      extractor = await tryLoad('wasm');
    } catch (wasmErr) {
      setStatus('Failed to load model. Your browser may not support WebGPU or WASM.');
      dom.loadModelBtn.hidden = false;
      throw wasmErr;
    }
  }
  setProgress(100);

  if (state.settings.enableTextSearch) {
    setStatus('Loading text model for search…');
    const TEXT_MODEL_SIZE_BYTES = 134 * 1024 * 1024;
    const textLoaded = new Map<string, number>();

    const textProgressCb = (e: ProgressEvent) => {
      if (e.status === 'progress') {
        textLoaded.set(e.file, e.loaded ?? 0);
        const total = [...textLoaded.values()].reduce((a, b) => a + b, 0);
        const pct = Math.min(99, (total / TEXT_MODEL_SIZE_BYTES) * 100);
        setProgress(pct);
        setStatus(`Loading text model… ${pct.toFixed(0)}%`);
      }
    };

    const tryLoadText = (device: 'webgpu' | 'wasm') => (pipeline as Pipeline)(
      'feature-extraction',
      'nomic-ai/nomic-embed-text-v1.5',
      { device, dtype: 'fp32', progress_callback: textProgressCb }
    ) as Promise<PipelineInstance>;

    try {
      textExtractor = await tryLoadText('webgpu');
    } catch (gpuErr) {
      console.warn('Text model WebGPU failed, using wasm:', gpuErr);
      textLoaded.clear();
      textExtractor = await tryLoadText('wasm');
    }
    setProgress(100);
  }

  state.phase = 'model_ready';
  setProgress(0);
  dom.loadModelBtn.hidden = true;
  dom.openBtn.disabled = false;
  dom.openBtn.classList.add('primary');
  if (!dom.resumeBtn.hidden) {
    dom.resumeBtn.disabled = false;
    dom.resumeBtn.classList.add('primary');
    setStatus('Model ready — resume or open a folder.');
  } else {
    setStatus('Model ready — open a folder to start.');
  }
}

// ── Embedding loop ───────────────────────────────────────────────────────────
async function embedAll(files: PhotoFile[]) {
  state.phase = 'embedding';
  const vectors = new Array<Float64Array>(files.length);
  let cacheHits = 0;
  const writeQueue: [CacheKey, Float64Array][] = [];

  const batchSize = state.settings.batchSize;
  for (let i = 0; i < files.length; i += batchSize) {
    const batch = files.slice(i, Math.min(i + batchSize, files.length));
    const keys = batch.map(f => `${f.name}:${f.size}:${f.lastModified}` as CacheKey);

    // One IDB transaction for the whole batch
    const cached = await cacheGetBatch(keys);

    // Resolve inputs for cache misses (convert video thumbnails to RawImage in parallel)
    const missIndices: number[] = [];
    const missInputs: (File | RawImage)[] = [];

    await Promise.all(batch.map(async (f, bi) => {
      if (cached[bi]) return; // cache hit — handled below
      const idx = i + bi;
      const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
      if (VIDEO_EXTS.has(ext)) {
        // Use cached thumbnail if available, otherwise extract frame now
        const thumb = state.thumbnails[idx] ?? await extractVideoFrame(f.file);
        if (thumb) {
          // Also store it so the render loop doesn't re-extract
          if (!state.thumbnails[idx]) {
            state.thumbnails[idx] = thumb;
            thumbDecoding.delete(idx);
          }
          const canvas = document.createElement('canvas');
          canvas.width = thumb.width;
          canvas.height = thumb.height;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(thumb, 0, 0);
            missInputs.push(await RawImage.fromCanvas(canvas));
          } else {
            missInputs.push(f.file);
          }
        } else {
          missInputs.push(f.file);
        }
      } else {
        missInputs.push(f.file);
      }
      missIndices.push(bi);
    }));

    // Apply cache hits
    for (let bi = 0; bi < batch.length; bi++) {
      if (cached[bi]) { vectors[i + bi] = cached[bi]!; cacheHits++; }
    }

    // Run true batch inference for all cache misses at once
    if (missInputs.length > 0) {
      if (!extractor) throw new Error('Extractor not loaded');
      try {
        const output = await extractor(missInputs.length === 1 ? missInputs[0] : missInputs, { pooling: 'mean', normalize: true });
        const extracted = missInputs.length === 1
          ? [extractVector(output)]
          : extractBatchedVectors(output, missInputs.length);

        for (let m = 0; m < missIndices.length; m++) {
          const bi = missIndices[m];
          const idx = i + bi;
          vectors[idx] = Float64Array.from(extracted[m]);
          writeQueue.push([keys[bi], vectors[idx]]);
        }
      } catch (err) {
        console.warn('Batch inference failed, filling zeros:', (err as Error).message);
        for (const bi of missIndices) {
          vectors[i + bi] = new Float64Array(768);
        }
      }
    }

    let didFlush = false;
    if (writeQueue.length >= WRITE_BATCH) {
      await cachePutBatch(writeQueue);
      writeQueue.length = 0;
      didFlush = true;
    }

    const done = Math.min(i + batchSize, files.length);

    if (done === batchSize || didFlush) {
      resizeCanvas();
      const cols = Math.ceil(Math.sqrt(files.length));
      const spacing = 60;
      state.points = Array.from({ length: done }, (_, j) => [
        (j % cols) * spacing,
        Math.floor(j / cols) * spacing
      ]) as Point[];
      fitCamera();
      scheduleRender();
    }
    const fromCache = cacheHits > 0 ? ` (${cacheHits} cached)` : '';
    setStatus(`Embedding ${done} / ${files.length} images…${fromCache}`);
    setProgress(10 + (done / files.length) * 80); // 10% to 90%
    await yieldMain();
  }

  if (writeQueue.length > 0) {
    await cachePutBatch(writeQueue);
  }

  refreshCacheSize();
  return vectors;
}

// ── Main run ─────────────────────────────────────────────────────────────────
async function processFiles(files: PhotoFile[]) {
  if (files.length === 0) {
    setStatus('No images found.');
    return;
  }
  state.files = files;
  setStatus(`Found ${files.length} media files. Loading thumbnails…`);

  try {
    state.thumbnails = initThumbnails(files);

    const vectors = await embedAll(files);
    state.vectors = vectors;
    
    setStatus('Building search index…');
    state.hnsw = new druid.HNSW(vectors, { metric: druid.cosine } as ConstructorParameters<typeof druid.HNSW>[1]);

    state.phase = 'projecting';
    setStatus(`Projecting with ${state.settings.projectionMethod}…`);
    setProgress(90);
    const nNeighbors = Math.max(2, Math.min(15, files.length - 1));
    const rawPoints = await runProjection(vectors, state.settings.projectionMethod, nNeighbors);
    state.rawPoints = rawPoints;

    const k = Math.min(8, Math.max(2, Math.ceil(Math.sqrt(files.length / 2))));
    state.clusters = await kmeansAsync(rawPoints, k);

    state.phase = 'done';
    resizeCanvas();
    state.points = await spreadPointsAsync(rawPoints);
    fitCamera();
    scheduleRender();
    setProgress(100);
    const finalMsg = `${files.length} media files · ${k} clusters`;
    setStatus(`${files.length} media files — tap to view · ${k} clusters`);
    if (dom.statsEl) dom.statsEl.textContent = finalMsg;
    dom.recenterBtn.disabled = false;
    dom.resetBtn.disabled = false;
    dom.headerRecenterBtn.disabled = false;
    dom.headerResetBtn.disabled = false;
    dom.searchInput.disabled = false;

    state.fileKeys = files.map(f => `${f.name}:${f.size}:${f.lastModified}`);
    try {
      localStorage.setItem('po_fileKeys', JSON.stringify(state.fileKeys));
      localStorage.setItem('po_umapPoints', JSON.stringify(state.points));
      localStorage.setItem('po_clusters', JSON.stringify(Array.from(state.clusters)));
    } catch (_) { /* quota exceeded */ }

  } catch (err) {
    console.error(err);
    setStatus(`Error: ${(err as Error).message}`);
  } finally {
    dom.openBtn.disabled = false;
  }
}

async function run(dirHandle: DirectoryHandle) {
  dom.openBtn.disabled = true;
  setProgress(0);

  try {
    setStatus('Scanning folder…');
    const files = await collectImages(dirHandle);
    await processFiles(files);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${(err as Error).message}`);
    dom.openBtn.disabled = false;
  }
}

// ── Interaction ──────────────────────────────────────────────────────────────
const pointers = new Map<number, PointerState>();
let lastPinchDist = 0;
let dragMoved = 0;

function pointerPos(e: PointerEvent): CanvasPointerPos {
  const rect = dom.canvas.getBoundingClientRect();
  return {
    cx: (e.clientX - rect.left) * (dom.canvas.width / rect.width),
    cy: (e.clientY - rect.top) * (dom.canvas.height / rect.height),
  };
}

dom.canvas.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  dom.canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  dragMoved = 0;
});

dom.canvas.addEventListener('pointermove', (e) => {
  e.preventDefault();
  const prev = pointers.get(e.pointerId);
  if (!prev) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (pointers.size === 1) {
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    dragMoved += Math.abs(dx) + Math.abs(dy);
    camera.x -= dx / camera.scale;
    camera.y -= dy / camera.scale;
    scheduleRender();
  } else if (pointers.size === 2) {
    const pts = [...pointers.values()];
    const dx = pts[0].x - pts[1].x, dy = pts[0].y - pts[1].y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (lastPinchDist > 0) {
      const midX = (pts[0].x + pts[1].x) / 2;
      const midY = (pts[0].y + pts[1].y) / 2;
      const rect = dom.canvas.getBoundingClientRect();
      const scaleRatio = dom.canvas.width / rect.width;
      const px = (midX - rect.left) * scaleRatio - dom.canvas.width / 2;
      const py = (midY - rect.top) * scaleRatio - dom.canvas.height / 2;
      const ratio = dist / lastPinchDist;
      camera.x += px / camera.scale - px / (camera.scale * ratio);
      camera.y += py / camera.scale - py / (camera.scale * ratio);
      camera.scale = Math.max(0.05, Math.min(20, camera.scale * ratio));
      scheduleRender();
    }
    lastPinchDist = dist;
  }
});

dom.canvas.addEventListener('pointerup', (e) => {
  e.preventDefault();
  const wasSingleTap = pointers.size === 1 && dragMoved < 15;
  pointers.delete(e.pointerId);
  if (pointers.size < 2) lastPinchDist = 0;

  if (wasSingleTap && state.phase === 'done' && state.points.length) {
    const { cx, cy } = pointerPos(e);
    const wx = (cx - dom.canvas.width / 2) / camera.scale + camera.x;
    const wy = (cy - dom.canvas.height / 2) / camera.scale + camera.y;
    const hitRadius = (THUMB_WORLD / 2) ** 2;
    let closest = -1, minD = hitRadius * 4;
    for (let i = 0; i < state.points.length; i++) {
      const ddx = state.points[i][0] - wx, ddy = state.points[i][1] - wy;
      const d = ddx * ddx + ddy * ddy;
      if (d < minD) { minD = d; closest = i; }
    }
    if (closest >= 0) {
      const f = state.files[closest];
      const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
      
      if (VIDEO_EXTS.has(ext)) {
        dom.modalImg.style.display = 'none';
        dom.modalVideo.style.display = 'block';
        dom.modalVideo.loop = state.settings.loopVideos;
        dom.modalVideo.src = f.objectURL || '';
        dom.modalVideo.play().catch(() => {}); // Autoplay when opened
      } else {
        dom.modalVideo.style.display = 'none';
        dom.modalVideo.pause();
        dom.modalVideo.src = '';
        dom.modalImg.style.display = 'block';
        dom.modalImg.src = f.objectURL || '';
      }
      
      dom.modalName.textContent = f.name.split('/').pop() || '';
      dom.modal.classList.add('open');
    }
  }
});

dom.canvas.addEventListener('pointercancel', (e) => {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) lastPinchDist = 0;
});

dom.canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.88 : 1.14;
  const rect = dom.canvas.getBoundingClientRect();
  const px = (e.clientX - rect.left) * (dom.canvas.width / rect.width) - dom.canvas.width / 2;
  const py = (e.clientY - rect.top) * (dom.canvas.height / rect.height) - dom.canvas.height / 2;
  camera.x += px / camera.scale - px / (camera.scale * factor);
  camera.y += py / camera.scale - py / (camera.scale * factor);
  camera.scale = Math.max(0.05, Math.min(20, camera.scale * factor));
  scheduleRender();
}, { passive: false });

dom.recenterBtn.addEventListener('click', () => { fitCamera(); scheduleRender(); });
dom.resetBtn.addEventListener('click', resetAll);
dom.headerRecenterBtn.addEventListener('click', () => { fitCamera(); scheduleRender(); });
dom.headerResetBtn.addEventListener('click', resetAll);

// ── Search input ─────────────────────────────────────────────────────────────
let searchDebounce: ReturnType<typeof setTimeout> | null = null;

dom.searchInput.addEventListener('input', () => {
  dom.searchClearBtn.hidden = !dom.searchInput.value.trim();

  if (searchDebounce) clearTimeout(searchDebounce);
  searchDebounce = setTimeout(async () => {
    if (dom.searchInput.value.trim()) {
      await searchImages(dom.searchInput.value);
    } else {
      state.searchResults = null;
      state.searchQuery = '';
      state.searchScores = null;
      dom.searchClearBtn.hidden = true;
      setStatus(`${state.files.length} images · tap to view`);
    }
    scheduleRender();
  }, 300);
});

dom.searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    dom.searchInput.value = '';
    state.searchResults = null;
    state.searchQuery = '';
    state.searchScores = null;
    dom.searchClearBtn.hidden = true;
    setStatus(`${state.files.length} images · tap to view`);
    scheduleRender();
  }
});

dom.searchClearBtn.addEventListener('click', () => {
  dom.searchInput.value = '';
  state.searchResults = null;
  state.searchQuery = '';
  state.searchScores = null;
  dom.searchClearBtn.hidden = true;
  dom.searchInput.focus();
  setStatus(`${state.files.length} images · tap to view`);
  scheduleRender();
});

const closeModal = () => {
  dom.modal.classList.remove('open');
  dom.modalVideo.pause();
  dom.modalVideo.src = '';
};

dom.modalClose.addEventListener('click', closeModal);

dom.modal.addEventListener('click', (e) => {
  if (e.target === dom.modal) closeModal();
});

// ── Settings ────────────────────────────────────────────────────────────────
const applyTheme = (theme: 'dark' | 'light' | 'system') => {
  const isLight = theme === 'light' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: light)').matches);
  document.documentElement.classList.toggle('light-mode', isLight);
};

const saveSettings = () => {
  localStorage.setItem('mc_settings', JSON.stringify(state.settings));
};

// Sync UI with initial settings
dom.densitySlider.value = state.settings.density.toString();
dom.drawBudgetSlider.value = state.settings.drawBudget.toString();
dom.loopToggle.checked = state.settings.loopVideos;
dom.themeSelect.value = state.settings.theme;
dom.enableSearchToggle.checked = state.settings.enableTextSearch;
if (dom.projectionSelect) dom.projectionSelect.value = state.settings.projectionMethod;
dom.batchSizeInput.value = state.settings.batchSize.toString();
applyTheme(state.settings.theme);

const updateSearchUI = () => {
  if (state.settings.enableTextSearch) {
    dom.bottomPanel.style.display = 'flex';
    dom.headerRecenterBtn.parentElement!.style.display = 'none';
  } else {
    dom.bottomPanel.style.display = 'none';
    dom.headerRecenterBtn.parentElement!.style.display = 'flex';
  }
  dom.searchInput.disabled = !state.settings.enableTextSearch || state.phase !== 'done';
};
updateSearchUI();
refreshCacheSize();

dom.settingsBtn.addEventListener('click', () => {
  dom.settingsModal.classList.add('open');
});

dom.settingsClose.addEventListener('click', () => {
  dom.settingsModal.classList.remove('open');
});

dom.settingsModal.addEventListener('click', (e) => {
  if (e.target === dom.settingsModal) dom.settingsModal.classList.remove('open');
});

dom.enableSearchToggle.addEventListener('change', async () => {
  state.settings.enableTextSearch = dom.enableSearchToggle.checked;
  saveSettings();
  updateSearchUI();

  // If enabled and models are already loaded, load the text model now
  if (state.settings.enableTextSearch && state.phase !== 'idle' && state.phase !== 'loading_model' && !textExtractor) {
    // We duplicate the text loading logic here for dynamic loading
    dom.settingsModal.classList.remove('open');
    setStatus('Loading text model for search…');
    const TEXT_MODEL_SIZE_BYTES = 134 * 1024 * 1024;
    const textLoaded = new Map<string, number>();

    const textProgressCb = (e: ProgressEvent) => {
      if (e.status === 'progress') {
        textLoaded.set(e.file, e.loaded ?? 0);
        const total = [...textLoaded.values()].reduce((a, b) => a + b, 0);
        const pct = Math.min(99, (total / TEXT_MODEL_SIZE_BYTES) * 100);
        setProgress(pct);
        setStatus(`Loading text model… ${pct.toFixed(0)}%`);
      }
    };

    const tryLoadText = (device: 'webgpu' | 'wasm') => (pipeline as Pipeline)(
      'feature-extraction',
      'nomic-ai/nomic-embed-text-v1.5',
      { device, dtype: 'fp32', progress_callback: textProgressCb }
    ) as Promise<PipelineInstance>;

    try {
      textExtractor = await tryLoadText('webgpu');
    } catch (gpuErr) {
      console.warn('Text model WebGPU failed, using wasm:', gpuErr);
      textLoaded.clear();
      textExtractor = await tryLoadText('wasm');
    }
    setProgress(100);
    setTimeout(() => setProgress(0), 500);
    setStatus('Text model loaded.');
  }
});

dom.densitySlider.addEventListener('input', async () => {
  state.settings.density = parseFloat(dom.densitySlider.value);
  saveSettings();
  if (state.phase === 'done' && state.rawPoints && state.files.length) {
    state.points = await spreadPointsAsync(state.rawPoints);
    scheduleRender();
  }
});

dom.drawBudgetSlider.addEventListener('input', () => {
  state.settings.drawBudget = parseInt(dom.drawBudgetSlider.value);
  saveSettings();
  scheduleRender();
});

dom.batchSizeInput.addEventListener('input', () => {
  const v = Math.max(1, Math.min(32, parseInt(dom.batchSizeInput.value) || 1));
  state.settings.batchSize = v;
  saveSettings();
});

const hasMemoryAPI = 'deviceMemory' in navigator || 'memory' in performance;
dom.batchSizeAutoBtn.hidden = !hasMemoryAPI;

dom.batchSizeAutoBtn.addEventListener('click', () => {
  const avgFileSize = state.files.length > 0
    ? state.files.reduce((s, f) => s + f.size, 0) / state.files.length
    : 2 * 1024 * 1024;
  const optimal = computeOptimalBatchSize(avgFileSize);
  state.settings.batchSize = optimal;
  dom.batchSizeInput.value = optimal.toString();
  saveSettings();
});

dom.loopToggle.addEventListener('change', () => {
  state.settings.loopVideos = dom.loopToggle.checked;
  saveSettings();
});

dom.themeSelect.addEventListener('change', () => {
  state.settings.theme = dom.themeSelect.value as 'dark' | 'light' | 'system';
  saveSettings();
  applyTheme(state.settings.theme);
});

if (dom.projectionSelect) {
  dom.projectionSelect.addEventListener('change', async (e) => {
    const select = e.target as HTMLSelectElement;
    state.settings.projectionMethod = select.value as ProjectionMethod;
    saveSettings();
    
    if (state.vectors.length > 0 && state.phase === 'done') {
      try {
        state.phase = 'projecting';
        dom.recenterBtn.disabled = true;
        if (dom.headerRecenterBtn) dom.headerRecenterBtn.disabled = true;
        
        const nNeighbors = Math.max(2, Math.min(15, state.files.length - 1));
        const rawPoints = await runProjection(state.vectors, state.settings.projectionMethod, nNeighbors);
        state.rawPoints = rawPoints;
        
        const k = Math.min(8, Math.max(2, Math.ceil(Math.sqrt(state.files.length / 2))));
        state.clusters = await kmeansAsync(rawPoints, k);
        
        state.points = await spreadPointsAsync(rawPoints);
        
        try {
          localStorage.setItem('po_projectedPoints', JSON.stringify(state.points));
          localStorage.setItem('po_clusters', JSON.stringify(Array.from(state.clusters)));
        } catch (_) {}

        fitCamera();
        scheduleRender();
        
        const finalMsg = `${state.files.length} media files · ${k} clusters`;
        setStatus(`${state.files.length} media files — tap to view · ${k} clusters`);
        if (dom.statsEl) dom.statsEl.textContent = finalMsg;
      } catch (err) {
        console.error('Reprojection error:', err);
        setStatus(`Reprojection failed: ${(err as Error).message}`);
      } finally {
        state.phase = 'done';
        dom.recenterBtn.disabled = false;
        if (dom.headerRecenterBtn) dom.headerRecenterBtn.disabled = false;
      }
    }
  });
}

dom.aboutBtn.addEventListener('click', () => {
  dom.aboutModal.classList.add('open');
});

dom.aboutClose.addEventListener('click', () => {
  dom.aboutModal.classList.remove('open');
});

dom.aboutModal.addEventListener('click', (e) => {
  if (e.target === dom.aboutModal) dom.aboutModal.classList.remove('open');
});

window.addEventListener('resize', () => {
  resizeCanvas();
  if (state.phase === 'done' && state.points.length) scheduleRender();
});

// ── Init ─────────────────────────────────────────────────────────────────────
resizeCanvas();

const _savedKeys = localStorage.getItem('po_fileKeys');
if (_savedKeys) {
  try {
    const n = JSON.parse(_savedKeys).length;
    dom.resumeBtn.hidden = false;
    dom.resumeBtn.textContent = `Resume last session (${n} images)`;
  } catch (_) { localStorage.clear(); }
}

dom.loadModelBtn.addEventListener('click', async () => {
  if (extractor) return;
  dom.loadModelBtn.disabled = true;
  await loadModel();
});

dom.openBtn.addEventListener('click', async () => {
  if (window.showDirectoryPicker) {
    try {
      const dir = await window.showDirectoryPicker({ mode: 'read' });
      await run(dir);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setStatus(`Error: ${(err as Error).message}`);
    }
  } else {
    // Fallback for Safari/iOS
    dom.fileInput.click();
  }
});

dom.fileInput.addEventListener('change', async () => {
  const fileList = dom.fileInput.files;
  if (!fileList || fileList.length === 0) return;

  dom.openBtn.disabled = true;
  setProgress(0);
  setStatus('Processing files…');

  const files: PhotoFile[] = [];
  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i];
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
      files.push({
        name: file.webkitRelativePath || file.name,
        size: file.size,
        lastModified: file.lastModified,
        file,
        objectURL: null
      });
    }
  }

  await processFiles(files);
});

dom.resumeBtn.addEventListener('click', async () => {
  const savedPoints = JSON.parse(localStorage.getItem('po_projectedPoints') || 'null');
  const savedClusters = JSON.parse(localStorage.getItem('po_clusters') || 'null');
  const savedKeys = JSON.parse(localStorage.getItem('po_fileKeys') || 'null');
  if (!savedPoints || !savedKeys) { await dom.openBtn.click(); return; }

  try {
    let files: PhotoFile[] = [];
    if (window.showDirectoryPicker) {
      const dir = await window.showDirectoryPicker({ mode: 'read' });
      setStatus('Matching files…');
      files = await collectImages(dir);
    } else {
      // Safari/iOS fallback
      setStatus('Please re-select the folder to resume.');
      const fileList = await new Promise<FileList | null>((resolve) => {
        const handler = () => {
          dom.fileInput.removeEventListener('change', handler);
          resolve(dom.fileInput.files);
        };
        dom.fileInput.addEventListener('change', handler);
        dom.fileInput.click();
      });

      if (!fileList || fileList.length === 0) return;
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
        if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
          files.push({

            name: file.webkitRelativePath || file.name,
            size: file.size,
            lastModified: file.lastModified,
            file,
            objectURL: null
          });
        }
      }
    }

    const keyToFile = new Map(files.map(f => [`${f.name}:${f.size}:${f.lastModified}`, f]));
    const matched = (savedKeys as string[]).map(k => keyToFile.get(k)).filter((f): f is PhotoFile => !!f);

    if (matched.length < (savedKeys as string[]).length * 0.8) {
      await processFiles(files); return;
    }

    state.files = matched;
    state.rawPoints = savedPoints;
    state.points = savedPoints.slice(0, matched.length);
    state.clusters = new Int32Array(savedClusters.slice(0, matched.length));

    setStatus('Restoring search index…');
    const keys = matched.map(f => `${f.name}:${f.size}:${f.lastModified}` as CacheKey);
    const cachedVectors = await cacheGetBatch(keys);
    state.vectors = cachedVectors.map(v => v || new Float64Array(768));
    if (state.vectors.length > 0) {
      state.hnsw = new druid.HNSW(state.vectors, { metric: druid.cosine } as ConstructorParameters<typeof druid.HNSW>[1]);
    }

    state.thumbnails = initThumbnails(matched);
    state.phase = 'done';
    resizeCanvas();
    fitCamera();
    scheduleRender();
    setProgress(100);
    const finalMsg = `${matched.length} media files · restored`;
    setStatus(`${matched.length} media files — resumed from session`);
    if (dom.statsEl) dom.statsEl.textContent = finalMsg;
    dom.recenterBtn.disabled = false;
    dom.resetBtn.disabled = false;
    dom.headerRecenterBtn.disabled = false;
    dom.headerResetBtn.disabled = false;
    dom.searchInput.disabled = false;
  } catch (err) {
    if ((err as Error).name !== 'AbortError') setStatus(`Error: ${(err as Error).message}`);
  }
});

// Auto-start model loading
dom.loadModelBtn.disabled = true;
(async () => {
  try {
    await loadModel();
  } catch (err) {
    dom.loadModelBtn.hidden = false;
    dom.loadModelBtn.disabled = false;
    setStatus(`Model failed: ${(err as Error).message}. Tap "Load Model" to retry.`);
  }
})();
