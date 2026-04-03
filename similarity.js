/**
 * Lightweight vector similarity search for browser.
 * Optimized for Float32Array embeddings (768-dim, L2-normalized).
 *
 * For 1,000-10,000 vectors, brute-force O(n) search is ~5-20ms —
 * no indexing needed for this scale.
 *
 * @module similarity
 */

// ── Dot product (cosine similarity for L2-normalized vectors) ───────────────
export function dotProduct(a, b) {
  const n = a.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

// ── Full similarity computation ────────────────────────────────────────────────
export function allSimilarities(query, vectors) {
  const result = new Float32Array(vectors.length);
  for (let i = 0; i < vectors.length; i++) {
    result[i] = dotProduct(query, vectors[i]);
  }
  return result;
}

// ── Sort indices by similarity ────────────────────────────────────────────────
export function sortBySimilarity(scores) {
  const indices = new Int32Array(scores.length);
  for (let i = 0; i < scores.length; i++) indices[i] = i;
  indices.sort((a, b) => scores[b] - scores[a]);
  return indices;
}
