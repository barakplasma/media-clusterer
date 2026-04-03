/**
 * Test utilities for photo organizer tests
 */

/**
 * Create mock vectors for testing
 */
export function createMockVectors(count: number, dim = 768): Float32Array[] {
  const vectors: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const v = new Float32Array(dim);
    // Fill with predictable pattern based on index
    for (let j = 0; j < dim; j++) {
      v[j] = (i * j) / (count * dim);
    }
    // L2 normalize
    const norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    for (let j = 0; j < dim; j++) {
      v[j] = v[j] / (norm || 1);
    }
    vectors.push(v);
  }
  return vectors;
}

/**
 * Create a mock file object
 */
export function createMockFile(name: string, size = 1024, lastModified = Date.now()): File {
  const content = new Uint8Array(size);
  return new File([content], name, { type: 'image/jpeg', lastModified });
}

/**
 * Wait for async operations (debounce, setTimeout, etc.)
 */
export function wait(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Create a minimal DOM environment for testing
 */
export function setupTestDOM(): void {
  document.body.innerHTML = `
    <input id="search-input" disabled placeholder="Search...">
    <button id="search-clear-btn" hidden>✕</button>
    <button id="recenter-btn" disabled>⌖ Recenter</button>
    <button id="reset-btn" disabled>Reset</button>
    <div id="status">Ready</div>
    <canvas id="canvas"></canvas>
  `;
}

/**
 * Clean up test DOM
 */
export function cleanupTestDOM(): void {
  document.body.innerHTML = '';
}
