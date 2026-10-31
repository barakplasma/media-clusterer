/**
 * Pure compute functions: clustering, layout, overlap relaxation, and
 * viewport culling. No DOM or app-state access — everything needed comes in
 * as parameters, so these can be unit-tested and benchmarked headlessly.
 */

import type { PhotoFile, Point } from './types'

export const THUMB_WORLD = 48 // thumbnail size in world units

/** Yield to the event loop so long computations don't block rendering/input. */
const defaultYield = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

// How long a compute loop may hold the main thread before yielding. ~12ms
// keeps a 60fps frame budget breathable while yielding far less often than a
// fixed every-N-iterations rule on small inputs.
const YIELD_BUDGET_MS = 12
const now: () => number =
  typeof performance !== 'undefined' ? () => performance.now() : () => Date.now()

export async function kmeansAsync(
  points: number[][],
  k: number,
  maxIter = 60,
  yieldFn: () => Promise<void> = defaultYield,
  onProgress?: (fraction: number) => void
): Promise<Int32Array> {
  const n = points.length
  if (n === 0) return new Int32Array(0)
  k = Math.max(1, Math.min(k, n))

  const idx = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  const centroids = idx.slice(0, k).map((i) => [points[i][0], points[i][1]])
  const labels = new Int32Array(n)

  let lastYield = now()
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false
    for (let i = 0; i < n; i++) {
      let best = 0,
        bestD = Infinity
      const px = points[i][0],
        py = points[i][1]
      for (let j = 0; j < k; j++) {
        const dx = px - centroids[j][0],
          dy = py - centroids[j][1]
        const d = dx * dx + dy * dy
        if (d < bestD) {
          bestD = d
          best = j
        }
      }
      if (labels[i] !== best) {
        labels[i] = best
        changed = true
      }
    }

    if (now() - lastYield > YIELD_BUDGET_MS) {
      onProgress?.(iter / maxIter)
      await yieldFn()
      lastYield = now()
    }
    if (!changed) break

    const sx = new Float64Array(k),
      sy = new Float64Array(k),
      cnt = new Int32Array(k)
    for (let i = 0; i < n; i++) {
      sx[labels[i]] += points[i][0]
      sy[labels[i]] += points[i][1]
      cnt[labels[i]]++
    }
    for (let j = 0; j < k; j++) {
      if (cnt[j]) {
        centroids[j][0] = sx[j] / cnt[j]
        centroids[j][1] = sy[j] / cnt[j]
      }
    }
  }
  onProgress?.(1)
  return labels
}

/**
 * Normalize projected points into world space and relax overlaps with a
 * grid-accelerated repulsion pass.
 */
export async function spreadPointsAsync(
  projectedPoints: number[][],
  density: number,
  yieldFn: () => Promise<void> = defaultYield,
  onProgress?: (fraction: number) => void
): Promise<Point[]> {
  const n = projectedPoints.length
  if (n === 0) return []

  // Normalize to zero-centered unit space. Median center and a 95th-percentile
  // radius instead of min/max: one outlier (a failed file's zero vector, an
  // odd photo) used to squash every other point into a clump in the middle.
  const sortedX = projectedPoints.map((p) => p[0]).sort((a, b) => a - b)
  const sortedY = projectedPoints.map((p) => p[1]).sort((a, b) => a - b)
  const cx = sortedX[n >> 1],
    cy = sortedY[n >> 1]
  const radii = projectedPoints
    .map(([x, y]) => Math.max(Math.abs(x - cx), Math.abs(y - cy)))
    .sort((a, b) => a - b)
  const r = radii[Math.floor((n - 1) * 0.95)] || radii[n - 1] || 1

  const vsize = Math.sqrt(n) * THUMB_WORLD * 1.4 * density
  // Flat typed arrays instead of an array of [x, y] tuples: the relaxation
  // loop touches every coordinate 60 times, and tuple arrays were the main
  // source of GC pressure here.
  const xs = new Float64Array(n)
  const ys = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    xs[i] = ((projectedPoints[i][0] - cx) / r) * vsize
    ys[i] = ((projectedPoints[i][1] - cy) / r) * vsize
  }

  const CELL = THUMB_WORLD * 2 * density
  const invCell = 1 / CELL
  // Numeric grid keys: cell coordinates stay well under 1e5, so gx*200003+gy
  // is collision-free and avoids allocating a string per point per iteration.
  const GRID_STRIDE = 200003
  const grid = new Map<number, number[]>()
  let lastYield = now()
  for (let iter = 0; iter < 60; iter++) {
    grid.clear()
    for (let i = 0; i < n; i++) {
      const key = Math.floor(xs[i] * invCell) * GRID_STRIDE + Math.floor(ys[i] * invCell)
      const bucket = grid.get(key)
      if (bucket) bucket.push(i)
      else grid.set(key, [i])
    }
    let moved = false
    for (let i = 0; i < n; i++) {
      const gx = Math.floor(xs[i] * invCell),
        gy = Math.floor(ys[i] * invCell)
      for (let dgx = -1; dgx <= 1; dgx++)
        for (let dgy = -1; dgy <= 1; dgy++) {
          const bucket = grid.get((gx + dgx) * GRID_STRIDE + (gy + dgy))
          if (!bucket) continue
          for (const j of bucket) {
            if (j <= i) continue
            const dx = xs[j] - xs[i],
              dy = ys[j] - ys[i]
            const d2 = dx * dx + dy * dy
            if (d2 < THUMB_WORLD * THUMB_WORLD) {
              const dist = Math.sqrt(d2)
              const push = (THUMB_WORLD - dist) / 2 + 0.1
              // Identical points (duplicate photos, zero vectors) have no
              // direction to push along; pick a stable one per pair.
              const angle = i * 7919 + j * 104729 // radians; any value works
              const nx = dist > 0 ? dx / dist : Math.cos(angle),
                ny = dist > 0 ? dy / dist : Math.sin(angle)
              xs[i] -= nx * push
              ys[i] -= ny * push
              xs[j] += nx * push
              ys[j] += ny * push
              moved = true
            }
          }
        }
    }
    if (now() - lastYield > YIELD_BUDGET_MS) {
      onProgress?.(iter / 60)
      await yieldFn()
      lastYield = now()
    }
    if (!moved) break
  }
  onProgress?.(1)
  const out: Point[] = new Array(n)
  for (let i = 0; i < n; i++) out[i] = [xs[i], ys[i]]
  return out
}

/**
 * Generate grid-based 2D coordinates from folder structure and datetime.
 * Creates a "folder clusters" layout:
 * - Folders arranged in a horizontal grid
 * - Within each folder, photos arranged by date (vertical time flow)
 * - Almost grid-like for predictable navigation
 */
export function generateMetadataBasedLayout(
  files: Pick<PhotoFile, 'name' | 'lastModified'>[]
): Point[] {
  if (files.length === 0) return []

  // Group files by folder path
  const folderGroups = new Map<string, Array<{ index: number; lastModified: number }>>()
  for (let i = 0; i < files.length; i++) {
    const pathParts = files[i].name.split('/')
    const folder = pathParts.slice(0, -1).join('/') || '(root)'
    if (!folderGroups.has(folder)) {
      folderGroups.set(folder, [])
    }
    folderGroups.get(folder)!.push({ index: i, lastModified: files[i].lastModified })
  }

  // Sort each group by date and collect folders in sorted order
  const sortedFolders: Array<{
    folder: string
    files: Array<{ index: number; lastModified: number }>
  }> = []
  for (const [folder, fileGroup] of folderGroups) {
    fileGroup.sort((a, b) => a.lastModified - b.lastModified)
    sortedFolders.push({ folder, files: fileGroup })
  }
  sortedFolders.sort((a, b) => a.folder.localeCompare(b.folder))

  // Calculate grid dimensions - first find max files per column across all folders
  const numFolders = sortedFolders.length
  const foldersPerRow = Math.ceil(Math.sqrt(numFolders * 1.5)) // Slightly wider grid
  let maxFilesPerCol = 0
  for (const { files: folderFiles } of sortedFolders) {
    const filesPerCol = Math.ceil(Math.sqrt(folderFiles.length))
    if (filesPerCol > maxFilesPerCol) maxFilesPerCol = filesPerCol
  }
  const folderGridWidth = foldersPerRow * THUMB_WORLD * (maxFilesPerCol + 1) // Space based on largest folder
  const fileGridSize = THUMB_WORLD * 1.5 // Space between files in a folder

  const points: Point[] = new Array(files.length) as Point[]

  for (let folderIdx = 0; folderIdx < sortedFolders.length; folderIdx++) {
    const { files: folderFiles } = sortedFolders[folderIdx]

    // Folder position in the grid
    const folderCol = folderIdx % foldersPerRow
    const folderRow = Math.floor(folderIdx / foldersPerRow)
    const folderOffsetX = folderCol * folderGridWidth
    const folderOffsetY = folderRow * folderGridWidth

    // Files within this folder - also in a grid
    const numFiles = folderFiles.length
    const filesPerCol = Math.ceil(Math.sqrt(numFiles))

    for (let i = 0; i < folderFiles.length; i++) {
      const { index } = folderFiles[i]

      // Position within folder grid (time flows downward)
      const fileCol = i % filesPerCol
      const fileRow = Math.floor(i / filesPerCol)

      const x = folderOffsetX + fileCol * fileGridSize
      const y = folderOffsetY + fileRow * fileGridSize

      points[index] = [x, y]
    }
  }

  return points
}

/**
 * Exact cosine-similarity search over L2-normalized vectors (cosine = dot
 * product for unit vectors). Returns indices sorted best-first plus a
 * per-index score array. Brute force is O(n·dims) — a few ms even at 20k×768 —
 * and needs no index build or extra copies of the vectors.
 */
export function searchByCosine(
  query: ArrayLike<number>,
  vectors: Float32Array[]
): { indices: Int32Array; scores: Float32Array } {
  const n = vectors.length
  const dims = query.length
  const scores = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const v = vectors[i]
    if (!v) {
      // Missing entry ranks last instead of poisoning the sort with NaN
      scores[i] = -1
      continue
    }
    // Clamp to the shorter vector so a dimension mismatch (mixed-model cache
    // corruption) degrades to a weaker score instead of NaN everywhere
    const limit = Math.min(v.length, dims)
    let dot = 0
    for (let j = 0; j < limit; j++) dot += v[j] * query[j]
    scores[i] = dot
  }
  const indices = new Int32Array(n)
  for (let i = 0; i < n; i++) indices[i] = i
  indices.sort((a, b) => scores[b] - scores[a])
  return { indices, scores }
}

/** Human-readable ETA from a seconds estimate, e.g. "~3m 20s left". */
export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return ''
  // Round the total first so 119.7s becomes 2m 0s, not 1m 60s
  const total = Math.max(1, Math.round(seconds))
  if (total < 90) return `~${total}s left`
  const m = Math.floor(total / 60)
  const s = total % 60
  if (m < 60) return `~${m}m ${s}s left`
  return `~${Math.floor(m / 60)}h ${m % 60}m left`
}

/**
 * Hoare quickselect: partition `order` so its first `k` entries are the ones
 * with the smallest `keys` values (unordered within the partition).
 */
export function selectSmallest(order: Int32Array, keys: Float64Array, k: number): void {
  let left = 0
  let right = order.length - 1
  while (right > left) {
    const pivot = keys[order[(left + right) >> 1]]
    let i = left,
      j = right
    while (i <= j) {
      while (keys[order[i]] < pivot) i++
      while (keys[order[j]] > pivot) j--
      if (i <= j) {
        const t = order[i]
        order[i] = order[j]
        order[j] = t
        i++
        j--
      }
    }
    if (k - 1 <= j) right = j
    else if (k - 1 >= i) left = i
    else break
  }
}

export interface ViewCamera {
  x: number
  y: number
  scale: number
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
  budget: number
): number[] {
  const s = camera.scale
  const cxW = viewportW / 2
  const cyW = viewportH / 2

  const visibleIndices: number[] = []
  for (let i = 0; i < pts.length; i++) {
    const sx = (pts[i][0] - camera.x) * s + cxW
    const sy = (pts[i][1] - camera.y) * s + cyW

    if (sx + half >= 0 && sx - half <= viewportW && sy + half >= 0 && sy - half <= viewportH) {
      visibleIndices.push(i)
    }
  }

  if (visibleIndices.length > budget) {
    // Precompute one key per visible point: top-20 search results first (by
    // rank, shifted below any possible distance), everything else by squared
    // distance to the camera center. Then quickselect the best `budget`
    // instead of sorting all visible points — O(m) instead of O(m log m).
    const m = visibleIndices.length
    const keys = new Float64Array(m)
    for (let p = 0; p < m; p++) {
      const i = visibleIndices[p]
      if (rank && rank[i] < 20) {
        // -1e15 sorts below any real squared distance while staying small
        // enough that integer ranks survive f64 rounding (ULP at 1e15 is 0.125)
        keys[p] = rank[i] - 1e15
      } else {
        const dx = pts[i][0] - camera.x
        const dy = pts[i][1] - camera.y
        keys[p] = dx * dx + dy * dy
      }
    }
    const order = new Int32Array(m)
    for (let p = 0; p < m; p++) order[p] = p
    selectSmallest(order, keys, budget)
    const head = Array.from(order.subarray(0, budget))
    head.sort((a, b) => keys[a] - keys[b])
    const out = new Array<number>(budget)
    for (let p = 0; p < budget; p++) out[p] = visibleIndices[head[p]]
    return out
  }

  return visibleIndices
}

/**
 * Index of the vector most similar (dot product; vectors are L2-normalized)
 * to vectors[from], ignoring indices `skip` rejects. -1 when none is left.
 */
export function mostSimilarIndex(
  vectors: Float32Array[],
  from: number,
  skip: (index: number) => boolean
): number {
  const a = vectors[from]
  let best = -1,
    bestScore = -Infinity
  for (let i = 0; i < vectors.length; i++) {
    if (i === from || skip(i)) continue
    const b = vectors[i]
    let dot = 0
    for (let d = 0; d < a.length; d++) dot += a[d] * b[d]
    if (dot > bestScore) {
      bestScore = dot
      best = i
    }
  }
  return best
}

// Words that say nothing about a group: English filler plus caption boilerplate.
const LABEL_STOPWORDS = new Set(
  (
    'the and with for from that this there their they are was were has have had its ' +
    'into onto over under near next some several many few one two three four five ' +
    'his her him she he who what which while where when then than very also just ' +
    'image photo picture photograph shows showing shown depicts depicting features ' +
    'featuring appears seems seen visible view scene background foreground front ' +
    'side top bottom left right center middle area close closeup up out off on of ' +
    'in at by an a is be been being it as or but not no can could would should may ' +
    'other another each both all any more most such like small large big little ' +
    'white black dark light bright colorful various different looking'
  ).split(' ')
)

/**
 * A short label per cluster from the captions its photos already have: the
 * one or two words that appear in the most captions (at least two). Clusters
 * with fewer than `minCaptions` captions get no label.
 */
export function clusterCaptionLabels(
  clusters: ArrayLike<number>,
  captions: ArrayLike<string | null | undefined>,
  minCaptions = 2
): Map<number, string> {
  const df = new Map<number, Map<string, number>>()
  const captioned = new Map<number, number>()
  for (let i = 0; i < clusters.length; i++) {
    const caption = captions[i]
    if (!caption) continue
    const c = clusters[i]
    captioned.set(c, (captioned.get(c) ?? 0) + 1)
    let counts = df.get(c)
    if (!counts) df.set(c, (counts = new Map()))
    // Count each word once per caption, so one wordy caption can't dominate.
    for (const w of new Set(caption.toLowerCase().match(/[a-z]{3,}/g) ?? [])) {
      if (!LABEL_STOPWORDS.has(w)) counts.set(w, (counts.get(w) ?? 0) + 1)
    }
  }
  const labels = new Map<number, string>()
  for (const [c, counts] of df) {
    if ((captioned.get(c) ?? 0) < minCaptions) continue
    const top = [...counts]
      .filter(([, n]) => n >= 2)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 2)
      .map(([w]) => w)
    if (top.length) labels.set(c, top.join(' · '))
  }
  return labels
}
