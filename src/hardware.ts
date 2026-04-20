/**
 * Hardware and environment utility functions.
 */

export function computeOptimalBatchSize(
  deviceMemoryGB?: number,
  perfMem?: { jsHeapSizeLimit: number; usedJSHeapSize: number }
): number {
  // Use passed in values or fall back to browser globals
  const memGB = deviceMemoryGB ?? (typeof navigator !== 'undefined' ? (navigator as any).deviceMemory : undefined) ?? 2;
  const memory = perfMem ?? (typeof performance !== 'undefined' ? (performance as any).memory : undefined);

  // JS heap headroom is a proxy for total available RAM; GPU memory is not
  // directly queryable from JS. Images are pre-resized to 256px before GPU
  // upload so input size is constant (~200 KB each) and no longer a factor.
  const headroomBytes = memory
    ? Math.max(0, memory.jsHeapSizeLimit - memory.usedJSHeapSize)
    : memGB * 0.3 * 1024 * 1024 * 1024; // assume 30% of device RAM usable

  const safeHeadroomBytes = headroomBytes * 0.8;

  // ViT-B/16: 197 tokens × 768 dims × 12 layers × 4 attention tensors × 4 bytes
  const vitActivationsPerImage = 197 * 768 * 12 * 4 * 4; // ~28 MB

  const optimal = Math.floor(safeHeadroomBytes / vitActivationsPerImage);
  return Math.min(32, Math.max(1, optimal));
}

export function getMemoryPressure(): { freeRatio: number } | null {
  const memory = typeof performance !== 'undefined' ? (performance as any).memory : undefined;
  if (!memory?.jsHeapSizeLimit) return null;
  const freeRatio = Math.max(0, (memory.jsHeapSizeLimit - memory.usedJSHeapSize) / memory.jsHeapSizeLimit);
  return { freeRatio };
}
