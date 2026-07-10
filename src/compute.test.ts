import { describe, it, expect } from 'vitest';
import { searchByCosine, cullAndPrioritize } from './compute';
import type { Point } from './types';

describe('cullAndPrioritize', () => {
  const camera = { x: 0, y: 0, scale: 1 };

  it('keeps all visible points when under budget', () => {
    const pts: Point[] = [[0, 0], [10, 10], [5000, 5000]]; // last one offscreen
    const out = cullAndPrioritize(pts, camera, 200, 200, 5, null, 10);
    expect(out).toEqual([0, 1]);
  });

  it('prioritizes by true distance to camera center when over budget', () => {
    // (50, 0) is closer than (60, 60) but farther than (10, 10).
    // The old comparator computed the second distance from pts[b][1] twice,
    // so any point with y≈0 unfairly won regardless of its x distance.
    const pts: Point[] = [[60, 60], [50, 0], [10, 10]];
    const out = cullAndPrioritize(pts, camera, 200, 200, 5, null, 2);
    expect(out).toEqual([2, 1]); // nearest two, nearest first
  });

  it('matches a full-sort reference on random data (quickselect check)', () => {
    let seed = 1234;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
    const pts: Point[] = Array.from({ length: 500 }, () => [(rnd() - 0.5) * 150, (rnd() - 0.5) * 150]);
    const budget = 40;
    const out = cullAndPrioritize(pts, camera, 200, 200, 5, null, budget);

    const d2 = (i: number) => pts[i][0] ** 2 + pts[i][1] ** 2;
    const reference = pts.map((_, i) => i).sort((a, b) => d2(a) - d2(b)).slice(0, budget);
    expect(out).toEqual(reference);
  });

  it('puts top-20 search results first, in rank order', () => {
    const pts: Point[] = [[90, 90], [1, 1], [80, 80]];
    const rank = Int32Array.from([3, 5, 999]); // idx0 and idx1 are top results
    const out = cullAndPrioritize(pts, camera, 200, 200, 5, rank, 2);
    expect(out).toEqual([0, 1]); // by rank, even though idx1 is nearest
  });
});

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
