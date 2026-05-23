import { describe, it, expect } from 'vitest';
import { computeOptimalBatchSize } from './hardware';

describe('Hardware Utilities: computeOptimalBatchSize', () => {
  it('uses deviceMemoryGB when performance.memory is unavailable', () => {
    // 8GB device — headroom = 8 * 0.3 * 0.8 GB = ~1.9 GB
    // vitActivationsPerImage ≈ 28 MB → optimal ~71, capped at 32
    const batchSize = computeOptimalBatchSize(8, undefined);
    expect(batchSize).toBe(32);
  });

  it('uses performance.memory when available', () => {
    const perfMem = {
      jsHeapSizeLimit: 4 * 1024 * 1024 * 1024, // 4 GB
      usedJSHeapSize:  1 * 1024 * 1024 * 1024, // 1 GB used → 3 GB headroom
    };
    // safe = 3 GB * 0.8 = 2.4 GB; 2.4 GB / 28 MB ≈ 85 → capped at 32
    const batchSize = computeOptimalBatchSize(undefined, perfMem);
    expect(batchSize).toBe(32);
  });

  it('returns 1 under severe memory constraints', () => {
    const perfMem = {
      jsHeapSizeLimit: 100 * 1024 * 1024, // 100 MB
      usedJSHeapSize:   95 * 1024 * 1024, // 95 MB used → 5 MB headroom
    };
    // safe = 4 MB; 4 MB / 28 MB < 1 → floor(0.14) = 0 → max(1, 0) = 1
    const batchSize = computeOptimalBatchSize(undefined, perfMem);
    expect(batchSize).toBe(1);
  });

  it('returns a small batch on a 2 GB device', () => {
    // 2 GB device — headroom = 2 * 0.3 * 0.8 GB = ~480 MB
    // 480 MB / 28 MB ≈ 17
    const batchSize = computeOptimalBatchSize(2, undefined);
    expect(batchSize).toBeGreaterThanOrEqual(16);
    expect(batchSize).toBeLessThanOrEqual(20);
  });
});
