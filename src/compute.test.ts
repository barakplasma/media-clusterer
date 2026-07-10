import { describe, it, expect } from 'vitest';
import { searchByCosine } from './compute';

describe('searchByCosine', () => {
  const rows = [
    Float32Array.from([1, 0, 0, 0]),
    Float32Array.from([0, 1, 0, 0]),
    Float32Array.from([0, 0, 1, 0]),
  ];

  it('returns indices sorted best-first with per-index scores', () => {
    const query = Float32Array.from([0.9, 0.1, 0, 0]);
    const { indices, scores } = searchByCosine(query, rows);
    expect(indices.length).toBe(3);
    expect(indices[0]).toBe(0); // most aligned with [1,0,0,0]
    expect(scores[0]).toBeCloseTo(0.9, 5);
    expect(scores[indices[0]]).toBeGreaterThanOrEqual(scores[indices[1]]);
    expect(scores[indices[1]]).toBeGreaterThanOrEqual(scores[indices[2]]);
  });

  it('handles zero vectors (failed embeddings) without NaN', () => {
    const withZero = [...rows, new Float32Array(4)];
    const { indices, scores } = searchByCosine(Float32Array.from([0, 1, 0, 0]), withZero);
    expect(indices[0]).toBe(1);
    expect(scores[3]).toBe(0);
    expect(Array.from(scores).some(Number.isNaN)).toBe(false);
  });

  it('handles empty vector list', () => {
    const { indices, scores } = searchByCosine(Float32Array.from([1, 0]), []);
    expect(indices.length).toBe(0);
    expect(scores.length).toBe(0);
  });
});
