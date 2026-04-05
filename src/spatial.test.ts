import { describe, it, expect } from 'vitest';
import { getNextImageInDirection } from './spatial';
import type { Point } from './types';

describe('Spatial Navigation: getNextImageInDirection', () => {
  it('returns currentIndex if there is only 1 point', () => {
    const points: Point[] = [[0, 0]];
    expect(getNextImageInDirection(0, points, 'right')).toBe(0);
  });

  it('finds the nearest neighbor to the right in a simple grid', () => {
    const points: Point[] = [
      [0, 0],   [10, 0],  [20, 0],
      [0, 10],  [10, 10], [20, 10]
    ];
    // from [0, 0] -> right should be [10, 0] (index 1)
    expect(getNextImageInDirection(0, points, 'right')).toBe(1);
    // from [10, 0] -> right should be [20, 0] (index 2)
    expect(getNextImageInDirection(1, points, 'right')).toBe(2);
  });

  it('finds the nearest neighbor down in a simple grid', () => {
    const points: Point[] = [
      [0, 0],   [10, 0],  [20, 0],
      [0, 10],  [10, 10], [20, 10]
    ];
    // from [10, 0] (index 1) -> down should be [10, 10] (index 4)
    expect(getNextImageInDirection(1, points, 'down')).toBe(4);
  });

  it('heavily penalizes diagonal drift', () => {
    const points: Point[] = [
      [0, 0],
      [10, 10], // Closer by euclidean distance (14.14), but very diagonal
      [20, 0]   // Further (20) but perfectly straight to the right
    ];
    // Moving right from [0, 0], it should prefer [20, 0] due to the orthogonal penalty
    expect(getNextImageInDirection(0, points, 'right')).toBe(2);
  });

  it('wraps around to the furthest point in the opposite direction', () => {
    const points: Point[] = [
      [0, 0],   [10, 0],  [20, 0],
      [0, 10],  [10, 10], [20, 10]
    ];
    // from [20, 0] (index 2) -> right should wrap to [0, 0] (index 0)
    expect(getNextImageInDirection(2, points, 'right')).toBe(0);

    // from [0, 10] (index 3) -> left should wrap to [20, 10] (index 5)
    expect(getNextImageInDirection(3, points, 'left')).toBe(5);

    // from [10, 10] (index 4) -> down should wrap to [10, 0] (index 1)
    expect(getNextImageInDirection(4, points, 'down')).toBe(1);
  });

  it('finds best wrap around even with slight orthogonal misalignment', () => {
    const points: Point[] = [
      [0, 2],    // "Row 0"
      [10, 0],   // "Row 0"
      [20, 1],   // "Row 0"
      [0, 100],  // Way down
      [20, 100], // Way down
    ];
    // from [20, 1] (index 2) -> right should wrap to [0, 2] (index 0) 
    // instead of wrapping to [0, 100] (index 3) because of orthogonal penalty
    expect(getNextImageInDirection(2, points, 'right')).toBe(0);
  });
});