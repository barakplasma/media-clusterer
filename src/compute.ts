/**
 * Pure compute functions: clustering, layout, overlap relaxation, and
 * viewport culling. No DOM or app-state access — everything needed comes in
 * as parameters, so these can be unit-tested and benchmarked headlessly.
 */

import type { PhotoFile, Point } from './types';

export const THUMB_WORLD = 48; // thumbnail size in world units

/** Yield to the event loop so long computations don't block rendering/input. */
const defaultYield = () => new Promise<void>(resolve => setTimeout(resolve, 0));

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

    if (iter % 10 === 0) await yieldFn();
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
  const pts: [number, number][] = projectedPoints.map(([x, y]) => [(x - cx) / r * vsize, (y - cy) / r * vsize]);

  const CELL = THUMB_WORLD * 2 * density;
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
    if (iter % 5 === 0) await yieldFn();
    if (!moved) break;
  }
  return pts as Point[];
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
    visibleIndices.sort((a, b) => {
      // Search results always first
      if (rank) {
        const ra = rank[a], rb = rank[b];
        if (ra < 20 || rb < 20) return ra - rb;
      }
      // Then by distance to camera center
      const da = (pts[a][0] - camera.x)**2 + (pts[a][1] - camera.y)**2;
      const db = (pts[b][1] - camera.y)**2 + (pts[b][1] - camera.y)**2;
      return da - db;
    });
    visibleIndices.length = budget;
  }

  return visibleIndices;
}
