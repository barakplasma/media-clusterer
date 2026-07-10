/**
 * Pure compute functions: clustering, layout, overlap relaxation, and
 * viewport culling. No DOM or app-state access — everything needed comes in
 * as parameters, so these can be unit-tested and benchmarked headlessly.
 */

import type { PhotoFile, Point } from './types';

export const THUMB_WORLD = 48; // thumbnail size in world units

/** Yield to the event loop so long computations don't block rendering/input. */
const defaultYield = () => new Promise<void>(resolve => setTimeout(resolve, 0));

// How long a compute loop may hold the main thread before yielding. ~12ms
// keeps a 60fps frame budget breathable while yielding far less often than a
// fixed every-N-iterations rule on small inputs.
const YIELD_BUDGET_MS = 12;
const now: () => number =
  typeof performance !== 'undefined' ? () => performance.now() : () => Date.now();

export async function kmeansAsync(
  points: number[][],
  k: number,
  maxIter = 60,
  yieldFn: () => Promise<void> = defaultYield,
): Promise<Int32Array> {
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

  let lastYield = now();
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

    if (now() - lastYield > YIELD_BUDGET_MS) {
      await yieldFn();
      lastYield = now();
    }
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

/**
 * Normalize projected points into world space and relax overlaps with a
 * grid-accelerated repulsion pass.
 */
export async function spreadPointsAsync(
  projectedPoints: number[][],
  density: number,
  yieldFn: () => Promise<void> = defaultYield,
): Promise<Point[]> {
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

  const vsize = Math.sqrt(n) * THUMB_WORLD * 1.4 * density;
  // Flat typed arrays instead of an array of [x, y] tuples: the relaxation
  // loop touches every coordinate 60 times, and tuple arrays were the main
  // source of GC pressure here.
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = (projectedPoints[i][0] - cx) / r * vsize;
    ys[i] = (projectedPoints[i][1] - cy) / r * vsize;
  }

  const CELL = THUMB_WORLD * 2 * density;
  const invCell = 1 / CELL;
  // Numeric grid keys: cell coordinates stay well under 1e5, so gx*200003+gy
  // is collision-free and avoids allocating a string per point per iteration.
  const GRID_STRIDE = 200003;
  const grid = new Map<number, number[]>();
  let lastYield = now();
  for (let iter = 0; iter < 60; iter++) {
    grid.clear();
    for (let i = 0; i < n; i++) {
      const key = Math.floor(xs[i] * invCell) * GRID_STRIDE + Math.floor(ys[i] * invCell);
      const bucket = grid.get(key);
      if (bucket) bucket.push(i);
      else grid.set(key, [i]);
    }
    let moved = false;
    for (let i = 0; i < n; i++) {
      const gx = Math.floor(xs[i] * invCell), gy = Math.floor(ys[i] * invCell);
      for (let dgx = -1; dgx <= 1; dgx++) for (let dgy = -1; dgy <= 1; dgy++) {
        const bucket = grid.get((gx + dgx) * GRID_STRIDE + (gy + dgy));
        if (!bucket) continue;
        for (const j of bucket) {
          if (j <= i) continue;
          const dx = xs[j] - xs[i], dy = ys[j] - ys[i];
          const d2 = dx * dx + dy * dy;
          if (d2 < THUMB_WORLD * THUMB_WORLD && d2 > 0) {
            const dist = Math.sqrt(d2);
            const push = (THUMB_WORLD - dist) / 2 + 0.1;
            const nx = dx / dist, ny = dy / dist;
            xs[i] -= nx * push; ys[i] -= ny * push;
            xs[j] += nx * push; ys[j] += ny * push;
            moved = true;
          }
        }
      }
    }
    if (now() - lastYield > YIELD_BUDGET_MS) {
      await yieldFn();
      lastYield = now();
    }
    if (!moved) break;
  }
  const out: Point[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = [xs[i], ys[i]];
  return out;
}

/**
 * Generate grid-based 2D coordinates from folder structure and datetime.
 * Creates a "folder clusters" layout:
 * - Folders arranged in a horizontal grid
 * - Within each folder, photos arranged by date (vertical time flow)
 * - Almost grid-like for predictable navigation
 */
export function generateMetadataBasedLayout(
  files: Pick<PhotoFile, 'name' | 'lastModified'>[],
): Point[] {
  if (files.length === 0) return [];

  // Group files by folder path
  const folderGroups = new Map<string, Array<{ index: number; lastModified: number }>>();
  for (let i = 0; i < files.length; i++) {
    const pathParts = files[i].name.split('/');
    const folder = pathParts.slice(0, -1).join('/') || '(root)';
    if (!folderGroups.has(folder)) {
      folderGroups.set(folder, []);
    }
    folderGroups.get(folder)!.push({ index: i, lastModified: files[i].lastModified });
  }

  // Sort each group by date and collect folders in sorted order
  const sortedFolders: Array<{ folder: string; files: Array<{ index: number; lastModified: number }> }> = [];
  for (const [folder, fileGroup] of folderGroups) {
    fileGroup.sort((a, b) => a.lastModified - b.lastModified);
    sortedFolders.push({ folder, files: fileGroup });
  }
  sortedFolders.sort((a, b) => a.folder.localeCompare(b.folder));

  // Calculate grid dimensions - first find max files per column across all folders
  const numFolders = sortedFolders.length;
  const foldersPerRow = Math.ceil(Math.sqrt(numFolders * 1.5)); // Slightly wider grid
  let maxFilesPerCol = 0;
  for (const { files: folderFiles } of sortedFolders) {
    const filesPerCol = Math.ceil(Math.sqrt(folderFiles.length));
    if (filesPerCol > maxFilesPerCol) maxFilesPerCol = filesPerCol;
  }
  const folderGridWidth = foldersPerRow * THUMB_WORLD * (maxFilesPerCol + 1); // Space based on largest folder
  const fileGridSize = THUMB_WORLD * 1.5; // Space between files in a folder

  const points: Point[] = new Array(files.length) as Point[];

  for (let folderIdx = 0; folderIdx < sortedFolders.length; folderIdx++) {
    const { files: folderFiles } = sortedFolders[folderIdx];

    // Folder position in the grid
    const folderCol = folderIdx % foldersPerRow;
    const folderRow = Math.floor(folderIdx / foldersPerRow);
    const folderOffsetX = folderCol * folderGridWidth;
    const folderOffsetY = folderRow * folderGridWidth;

    // Files within this folder - also in a grid
    const numFiles = folderFiles.length;
    const filesPerCol = Math.ceil(Math.sqrt(numFiles));

    for (let i = 0; i < folderFiles.length; i++) {
      const { index } = folderFiles[i];

      // Position within folder grid (time flows downward)
      const fileCol = i % filesPerCol;
      const fileRow = Math.floor(i / filesPerCol);

      const x = folderOffsetX + fileCol * fileGridSize;
      const y = folderOffsetY + fileRow * fileGridSize;

      points[index] = [x, y];
    }
  }

  return points;
}

/**
 * Exact cosine-similarity search over L2-normalized vectors (cosine = dot
 * product for unit vectors). Returns indices sorted best-first plus a
 * per-index score array. Brute force is O(n·dims) — a few ms even at 20k×768 —
 * and needs no index build or extra copies of the vectors.
 */
export function searchByCosine(
  query: ArrayLike<number>,
  vectors: Float32Array[],
): { indices: Int32Array; scores: Float32Array } {
  const n = vectors.length;
  const dims = query.length;
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = vectors[i];
    let dot = 0;
    for (let j = 0; j < dims; j++) dot += v[j] * query[j];
    scores[i] = dot;
  }
  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  indices.sort((a, b) => scores[b] - scores[a]);
  return { indices, scores };
}

/**
 * Hoare quickselect: partition `order` so its first `k` entries are the ones
 * with the smallest `keys` values (unordered within the partition).
 */
function selectSmallest(order: Int32Array, keys: Float64Array, k: number): void {
  let left = 0;
  let right = order.length - 1;
  while (right > left) {
    const pivot = keys[order[(left + right) >> 1]];
    let i = left, j = right;
    while (i <= j) {
      while (keys[order[i]] < pivot) i++;
      while (keys[order[j]] > pivot) j--;
      if (i <= j) {
        const t = order[i]; order[i] = order[j]; order[j] = t;
        i++; j--;
      }
    }
    if (k - 1 <= j) right = j;
    else if (k - 1 >= i) left = i;
    else break;
  }
}

export interface ViewCamera {
  x: number;
  y: number;
  scale: number;
}

/**
 * Frustum-cull points against the viewport and, when more are visible than
 * the draw budget, prioritize: search results first, then distance to the
 * camera center. Returns the indices to draw, at most `budget` of them.
 * `rank` is the search-result rank per index (or null when no search active).
 */
export function cullAndPrioritize(
  pts: readonly Point[],
  camera: ViewCamera,
  viewportW: number,
  viewportH: number,
  half: number,
  rank: Int32Array | null,
  budget: number,
): number[] {
  const s = camera.scale;
  const cxW = viewportW / 2;
  const cyW = viewportH / 2;

  const visibleIndices: number[] = [];
  for (let i = 0; i < pts.length; i++) {
    const sx = (pts[i][0] - camera.x) * s + cxW;
    const sy = (pts[i][1] - camera.y) * s + cyW;

    if (sx + half >= 0 && sx - half <= viewportW &&
      sy + half >= 0 && sy - half <= viewportH) {
      visibleIndices.push(i);
    }
  }

  if (visibleIndices.length > budget) {
    // Precompute one key per visible point: top-20 search results first (by
    // rank, shifted below any possible distance), everything else by squared
    // distance to the camera center. Then quickselect the best `budget`
    // instead of sorting all visible points — O(m) instead of O(m log m).
    const m = visibleIndices.length;
    const keys = new Float64Array(m);
    for (let p = 0; p < m; p++) {
      const i = visibleIndices[p];
      if (rank && rank[i] < 20) {
        // -1e15 sorts below any real squared distance while staying small
        // enough that integer ranks survive f64 rounding (ULP at 1e15 is 0.125)
        keys[p] = rank[i] - 1e15;
      } else {
        const dx = pts[i][0] - camera.x;
        const dy = pts[i][1] - camera.y;
        keys[p] = dx * dx + dy * dy;
      }
    }
    const order = new Int32Array(m);
    for (let p = 0; p < m; p++) order[p] = p;
    selectSmallest(order, keys, budget);
    const head = Array.from(order.subarray(0, budget));
    head.sort((a, b) => keys[a] - keys[b]);
    const out = new Array<number>(budget);
    for (let p = 0; p < budget; p++) out[p] = visibleIndices[head[p]];
    return out;
  }

  return visibleIndices;
}
