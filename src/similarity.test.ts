import { describe, it, expect } from 'vitest';
import { dotProduct, allSimilarities, sortBySimilarity, topK } from './similarity';

describe('similarity', () => {
  describe('dotProduct', () => {
    it('calculates dot product correctly', () => {
      const a = new Float32Array([1, 2, 3]);
      const b = new Float32Array([4, 5, 6]);
      expect(dotProduct(a, b)).toBe(32); // 1*4 + 2*5 + 3*6 = 32
    });

    it('handles zero vectors', () => {
      const a = new Float32Array([0, 0, 0]);
      const b = new Float32Array([1, 2, 3]);
      expect(dotProduct(a, b)).toBe(0);
    });

    it('handles negative values', () => {
      const a = new Float32Array([-1, 2, -3]);
      const b = new Float32Array([4, -5, 6]);
      expect(dotProduct(a, b)).toBe(-32); // -1*4 + 2*-5 + -3*6 = -32
    });

    it('returns 1 for identical normalized vectors', () => {
      const a = new Float32Array([0.577, 0.577, 0.577]); // Normalized [1,1,1]
      expect(dotProduct(a, a)).toBeCloseTo(1.0, 2);
    });
  });

  describe('allSimilarities', () => {
    const vectors = [
      new Float32Array([1, 0, 0]),
      new Float32Array([0, 1, 0]),
      new Float32Array([0, 0, 1]),
      new Float32Array([1, 1, 0]),
    ] as const;

    it('computes similarities for all vectors', () => {
      const query = new Float32Array([1, 0, 0]);
      const scores = allSimilarities(query, vectors);

      expect(scores).toBeInstanceOf(Float32Array);
      expect(scores.length).toBe(4);
      expect(scores[0]).toBe(1.0); // Exact match
      expect(scores[1]).toBe(0.0); // Orthogonal
      expect(scores[2]).toBe(0.0); // Orthogonal
      expect(scores[3]).toBeCloseTo(0.707, 2); // ~45 degrees
    });

    it('handles empty array', () => {
      const query = new Float32Array([1, 0, 0]);
      const scores = allSimilarities(query, []);
      expect(scores.length).toBe(0);
    });
  });

  describe('sortBySimilarity', () => {
    it('sorts indices by score descending', () => {
      const scores = new Float32Array([0.3, 0.9, 0.1, 0.7]);
      const sorted = sortBySimilarity(scores);

      expect(sorted).toBeInstanceOf(Int32Array);
      expect(Array.from(sorted)).toEqual([1, 3, 0, 2]); // Indices sorted by score
    });

    it('handles empty array', () => {
      const scores = new Float32Array([]);
      const sorted = sortBySimilarity(scores);
      expect(sorted.length).toBe(0);
    });

    it('handles single element', () => {
      const scores = new Float32Array([0.5]);
      const sorted = sortBySimilarity(scores);
      expect(Array.from(sorted)).toEqual([0]);
    });
  });

  describe('topK', () => {
    const vectors = [
      new Float32Array([1, 0, 0]),
      new Float32Array([0, 1, 0]),
      new Float32Array([0, 0, 1]),
      new Float32Array([1, 1, 0]),
      new Float32Array([0, 1, 1]),
    ] as const;

    it('returns top k results', () => {
      const query = new Float32Array([1, 0, 0]);
      const results = topK(query, vectors, 3);

      expect(results.length).toBe(3);
      expect(results[0][0]).toBe(0); // Index 0 has highest score
      expect(results[0][1]).toBe(1.0); // Score = 1.0
      expect(results[1][0]).toBe(3); // Index 3 has second highest
      expect(results[1][1]).toBeCloseTo(0.707, 2);
    });

    it('handles k larger than array length', () => {
      const query = new Float32Array([1, 0, 0]);
      const results = topK(query, vectors, 100);

      expect(results.length).toBe(5); // All vectors returned
    });

    it('handles k=1', () => {
      const query = new Float32Array([1, 0, 0]);
      const results = topK(query, vectors, 1);

      expect(results.length).toBe(1);
      expect(results[0][0]).toBe(0);
    });

    it('returns results sorted by score descending', () => {
      const query = new Float32Array([1, 0, 0]);
      const results = topK(query, vectors, 3);

      for (let i = 1; i < results.length; i++) {
        expect(results[i-1][1]).toBeGreaterThanOrEqual(results[i][1]);
      }
    });
  });

  describe('cosine similarity behavior', () => {
    it('higher similarity for closer vectors', () => {
      const query = new Float32Array([1, 0]);

      const vectors = [
        new Float32Array([1, 0]),      // Identical: score = 1.0
        new Float32Array([0.707, 0.707]), // 45 deg: score ≈ 0.707
        new Float32Array([0, 1]),      // 90 deg: score = 0.0
        new Float32Array([-1, 0]),     // 180 deg: score = -1.0
      ] as const;

      const scores = allSimilarities(query, vectors);

      expect(scores[0]).toBeCloseTo(1.0, 2);
      expect(scores[1]).toBeCloseTo(0.707, 2);
      expect(scores[2]).toBe(0.0);
      expect(scores[3]).toBe(-1.0);
    });
  });
});
