import { describe, it, expect, vi, afterEach } from 'vitest';
import { computeOptimalBatchSize, getMemoryPressure } from './hardware';

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

  it('adjusts batch size for very large files', () => {
    // 1000MB average file size!
    // bytesPerImageInference = max(500MB, ~87MB) = 500MB
    const batchSize = computeOptimalBatchSize(1000 * 1024 * 1024, 16, undefined); // 16GB device

    // 16 * 0.3 * 0.8 = 3.84GB safe headroom = ~4,123,168,604 bytes
    // 4,123,168,604 / (500 * 1024 * 1024) = ~7.86 -> 7
    expect(batchSize).toBe(7);
  });
});

describe('Hardware Utilities: getMemoryPressure', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns null when performance.memory is unavailable', () => {
    vi.spyOn(performance as any, 'memory', 'get').mockReturnValue(undefined);
    expect(getMemoryPressure()).toBeNull();
  });

  it('returns freeRatio when performance.memory is available', () => {
    vi.spyOn(performance as any, 'memory', 'get').mockReturnValue({
      jsHeapSizeLimit: 1000,
      usedJSHeapSize: 600,
      totalJSHeapSize: 700,
    });
    const result = getMemoryPressure();
    expect(result).not.toBeNull();
    expect(result!.freeRatio).toBeCloseTo(0.4);
  });

  it('clamps freeRatio to 0 when used exceeds limit', () => {
    vi.spyOn(performance as any, 'memory', 'get').mockReturnValue({
      jsHeapSizeLimit: 1000,
      usedJSHeapSize: 1100,
      totalJSHeapSize: 1100,
    });
    const result = getMemoryPressure();
    expect(result!.freeRatio).toBe(0);
  });

  it('returns freeRatio < 0.20 when memory is low', () => {
    vi.spyOn(performance as any, 'memory', 'get').mockReturnValue({
      jsHeapSizeLimit: 1000,
      usedJSHeapSize: 850,
      totalJSHeapSize: 900,
    });
    const result = getMemoryPressure();
    expect(result!.freeRatio).toBeCloseTo(0.15);
    expect(result!.freeRatio).toBeLessThan(0.20);
  });
});
