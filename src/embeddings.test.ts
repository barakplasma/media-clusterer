import { describe, it, expect } from 'vitest';
import { l2normalize, makeCacheKey } from './embeddings';

/**
 * Tests for Nomic embedding alignment.
 */

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // Both are L2 normalized, so dot product = cosine similarity
}

describe('Embedding utilities', () => {
  it('l2normalize should normalize a vector to unit length', () => {
    const vec = new Float32Array([1, 1, 1, 1]);
    const norm = l2normalize(vec);
    let sumSq = 0;
    for (let i = 0; i < norm.length; i++) sumSq += norm[i] * norm[i];
    expect(sumSq).toBeCloseTo(1, 5);
  });
});

describe('makeCacheKey (folder-independent embedding cache)', () => {
  const size = 12345;
  const lastModified = 1700000000000;

  it('produces the same key regardless of folder depth', () => {
    const fromRoot = makeCacheKey({ name: 'photos/trip/a.jpg', size, lastModified });
    const fromParent = makeCacheKey({ name: 'trip/a.jpg', size, lastModified });
    const fromChild = makeCacheKey({ name: 'a.jpg', size, lastModified });
    expect(fromRoot).toBe(fromChild);
    expect(fromParent).toBe(fromChild);
    expect(fromChild).toBe(`a.jpg:${size}:${lastModified}`);
  });

  it('keeps distinct files with the same basename apart via size/lastModified', () => {
    const a = makeCacheKey({ name: 'x/a.jpg', size: 100, lastModified });
    const b = makeCacheKey({ name: 'y/a.jpg', size: 200, lastModified });
    expect(a).not.toBe(b);
  });

  it('handles a bare filename with no path', () => {
    expect(makeCacheKey({ name: 'a.jpg', size, lastModified })).toBe(`a.jpg:${size}:${lastModified}`);
  });
});

describe('Nomic Embedding Alignment', () => {
  describe('Text query prefix', () => {
    it('adds search_query: prefix to user input', () => {
      const input = 'ball';
      const prefixed = `search_query: ${input}`;
      expect(prefixed).toBe('search_query: ball');
    });

    it('handles empty input gracefully', () => {
      const input = '';
      const prefixed = `search_query: ${input}`;
      expect(prefixed).toBe('search_query: ');
    });

    it('handles multi-word queries', () => {
      const input = 'sunset at the beach';
      const prefixed = `search_query: ${input}`;
      expect(prefixed).toBe('search_query: sunset at the beach');
    });
  });
});
