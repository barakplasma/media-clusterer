import { describe, it, expect } from 'vitest';
import { computeOptimalBatchSize } from './hardware';

describe('Hardware Utilities: computeOptimalBatchSize', () => {
  it('calculates optimal batch size based on deviceMemoryGB when performance.memory is missing', () => {
    // 8GB device memory, 2MB average file size
    const batchSize = computeOptimalBatchSize(2 * 1024 * 1024, 8, undefined);
    expect(batchSize).toBe(23);
  });

  it('calculates optimal batch size using performance.memory when available', () => {
    // Exact simulation of Chrome's performance.memory
    const perfMem = {
      jsHeapSizeLimit: 4 * 1024 * 1024 * 1024, // 4GB limit
      usedJSHeapSize: 1 * 1024 * 1024 * 1024,  // 1GB used
    };
    
    const batchSize = computeOptimalBatchSize(2 * 1024 * 1024, undefined, perfMem);
    expect(batchSize).toBe(29);
  });

  it('returns at least 1 even under severe memory constraints', () => {
    const perfMem = {
      jsHeapSizeLimit: 100 * 1024 * 1024, // 100MB limit
      usedJSHeapSize: 95 * 1024 * 1024,   // 95MB used -> 5MB headroom
    };
    
    // safe headroom = 4MB. That is less than the ~87MB needed per image.
    // The function should return Math.max(1, optimal) -> 1
    const batchSize = computeOptimalBatchSize(2 * 1024 * 1024, undefined, perfMem);
    expect(batchSize).toBe(1);
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