/**
 * Lightweight vector similarity search for browser.
 * Optimized for Float32Array embeddings (768-dim, L2-normalized).
 *
 * For 1,000-10,000 vectors, brute-force O(n) search is ~5-20ms —
 * no indexing needed for this scale.
 *
 * @module similarity
 */

export type Vector = Float32Array;
export type SearchResult = readonly [index: number, score: number];

/**
 * Dot product (cosine similarity for L2-normalized vectors).
 * For normalized vectors, this equals cosine similarity.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Dot product score in range [-1, 1], typically [0, 1] for normalized embeddings
 */
export function dotProduct(a: Vector, b: Vector): number {
  const n = a.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * Compute similarity scores between query and all vectors.
 * Uses dot product (cosine similarity for L2-normalized vectors).
 *
 * @param query - Query vector to compare against all vectors
 * @param vectors - Array of vectors to compare
 * @returns Float32Array of similarity scores in same order as input vectors
 */
export function allSimilarities(query: Vector, vectors: readonly Vector[]): Float32Array {
  const result = new Float32Array(vectors.length);
  for (let i = 0; i < vectors.length; i++) {
    result[i] = dotProduct(query, vectors[i]);
  }
  return result;
}

/**
 * Sort indices by similarity score descending.
 *
 * @param scores - Float32Array of similarity scores
 * @returns Int32Array of indices sorted by score (highest first)
 */
export function sortBySimilarity(scores: Float32Array): Int32Array {
  const indices = new Int32Array(scores.length);
  for (let i = 0; i < scores.length; i++) indices[i] = i;
  indices.sort((a, b) => scores[b] - scores[a]);
  return indices;
}

/**
 * Find top-k most similar vectors to query.
 * Uses min-heap for O(n log k) performance.
 *
 * @param query - Query vector
 * @param vectors - Array of vectors to search
 * @param k - Number of top results to return
 * @returns Array of [index, score] tuples sorted by score descending
 */
export function topK(
  query: Vector,
  vectors: readonly Vector[],
  k: number
): SearchResult[] {
  const n = vectors.length;
  k = Math.min(k, n);

  // Min-heap of size k: [score, index]
  const heap: [number, number][] = [];

  for (let i = 0; i < n; i++) {
    const score = dotProduct(query, vectors[i]);

    if (heap.length < k) {
      heapPush(heap, [score, i]);
    } else if (score > heap[0][0]) {
      heapReplaceMin(heap, [score, i]);
    }
  }

  // Sort by score descending
  heap.sort((a, b) => b[0] - a[0]);

  // Return as [index, score] format
  return heap.map(([score, index]) => [index, score]);
}

/** Min-heap push operation */
function heapPush(heap: [number, number][], item: [number, number]): void {
  heap.push(item);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (heap[p][0] <= heap[i][0]) break;
    [heap[p], heap[i]] = [heap[i], heap[p]];
    i = p;
  }
}

/** Min-heap replace minimum operation */
function heapReplaceMin(heap: [number, number][], item: [number, number]): void {
  heap[0] = item;
  let i = 0;
  while (true) {
    const left = (i << 1) + 1;
    const right = left + 1;
    let smallest = i;

    if (left < heap.length && heap[left][0] < heap[smallest][0]) smallest = left;
    if (right < heap.length && heap[right][0] < heap[smallest][0]) smallest = right;

    if (smallest === i) break;
    [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
    i = smallest;
  }
}
