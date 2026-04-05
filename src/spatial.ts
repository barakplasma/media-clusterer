import type { Point } from './types';

/**
 * Finds the index of the next image in a given direction based on spatial layout.
 * Uses a directional nearest-neighbor algorithm.
 */
export function getNextImageInDirection(
  currentIndex: number,
  points: Point[],
  direction: 'left' | 'right' | 'up' | 'down'
): number {
  if (points.length <= 1) return currentIndex;

  const cur = points[currentIndex];
  let dx = 0, dy = 0;
  if (direction === 'left') dx = -1;
  else if (direction === 'right') dx = 1;
  else if (direction === 'up') dy = -1;
  else if (direction === 'down') dy = 1;

  let bestIdx = -1;
  let minCost = Infinity;

  // 1. Find the nearest neighbor that lies strictly in the requested direction
  for (let i = 0; i < points.length; i++) {
    if (i === currentIndex) continue;
    const p = points[i];
    const vx = p[0] - cur[0];
    const vy = p[1] - cur[1];
    
    // Projection of V onto the direction vector D
    const proj = vx * dx + vy * dy;
    // Orthogonal distance from the direction vector D
    const orth = Math.abs(vx * dy - vy * dx);

    // Only consider points strictly in the forward half-plane
    if (proj > 0) {
      // Cost: squared distance, but heavily penalize orthogonal drift
      const cost = proj * proj + orth * orth * 5;
      if (cost < minCost) {
        minCost = cost;
        bestIdx = i;
      }
    }
  }

  if (bestIdx !== -1) {
    return bestIdx;
  }

  // 2. Wrap around: if at the edge, find the point furthest in the OPPOSITE direction
  let wrapIdx = -1;
  let minWrapScore = Infinity;
  for (let i = 0; i < points.length; i++) {
    if (i === currentIndex) continue;
    const p = points[i];
    const vx = p[0] - cur[0];
    const vy = p[1] - cur[1];
    
    const proj = vx * dx + vy * dy;
    const orth = Math.abs(vx * dy - vy * dx);

    // We want the most negative projection (furthest backward). 
    // We add an orthogonal penalty so it wraps to the same visual "row/column"
    const score = proj + orth * 2; 
    if (score < minWrapScore) {
      minWrapScore = score;
      wrapIdx = i;
    }
  }

  return wrapIdx !== -1 ? wrapIdx : currentIndex;
}