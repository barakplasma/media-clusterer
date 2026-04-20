/**
 * Hardware and environment utility functions.
 */

export function computeOptimalBatchSize(
  avgFileSizeBytes: number = 2 * 1024 * 1024,
  deviceMemoryGB?: number,
  perfMem?: { jsHeapSizeLimit: number; usedJSHeapSize: number }
): number {
  // Use passed in values or fall back to browser globals
  const memGB = deviceMemoryGB ?? (typeof navigator !== 'undefined' ? (navigator as any).deviceMemory : undefined) ?? 2;
  const memory = perfMem ?? (typeof performance !== 'undefined' ? (performance as any).memory : undefined);

  const headroomBytes = memory
    ? Math.max(0, memory.jsHeapSizeLimit - memory.usedJSHeapSize)
    : memGB * 0.3 * 1024 * 1024 * 1024; // assume 30% of device RAM usable

  // Apply a 20% safety margin to the available headroom
  const safeHeadroomBytes = headroomBytes * 0.8;

  // ViT-B/16: 197 patches × 768 dims × 12 layers × 4 attention tensors × 4 bytes × 3× safety margin
  const vitActivationsPerImage = 197 * 768 * 12 * 4 * 4 * 3; // ~87MB
  const bytesPerImageInference = Math.max(avgFileSizeBytes * 0.5, vitActivationsPerImage);

  const optimal = Math.floor(safeHeadroomBytes / bytesPerImageInference);
  return Math.max(1, optimal);
}

export function getMemoryPressure(): { freeRatio: number } | null {
  const memory = typeof performance !== 'undefined' ? (performance as any).memory : undefined;
  if (!memory?.jsHeapSizeLimit) return null;
  const freeRatio = Math.max(0, (memory.jsHeapSizeLimit - memory.usedJSHeapSize) / memory.jsHeapSizeLimit);
  return { freeRatio };
}
