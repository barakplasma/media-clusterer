/**
 * Sapiens2 ONNX embedder for browser inference via onnxruntime-web.
 * Model: barakplasma/sapiens2-onnx (facebook/sapiens2-pretrain-0.1b, int8)
 * Output: 768-dim L2-normalized float32 vector per image.
 */

import * as ort from 'onnxruntime-web';
import { l2normalize } from './embeddings';

const MODEL_URL =
  'https://huggingface.co/barakplasma/sapiens2-onnx/resolve/main/sapiens2_0.1b_int8.onnx';
const MODEL_CACHE_NAME = 'sapiens2-model-v1';

// Input resolution expected by the model
const MODEL_H = 1024;
const MODEL_W = 768;

// ImageNet normalization — precomputed as scale+offset so the hot loop uses
// multiply instead of divide (3× cheaper per pixel, 786k pixels per image).
const R_SCALE = 1 / (255 * 0.229), R_OFF = 0.485 / 0.229;
const G_SCALE = 1 / (255 * 0.224), G_OFF = 0.456 / 0.224;
const B_SCALE = 1 / (255 * 0.225), B_OFF = 0.406 / 0.225;

export type Sapiens2Session = ort.InferenceSession;

// Progress callback receives current percent and whether bytes came from cache.
export type Sapiens2ProgressCallback = (pct: number, fromCache: boolean) => void;

async function downloadWithProgress(
  url: string,
  onProgress?: (pct: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Sapiens2 download failed: ${res.status}`);

  const total = parseInt(res.headers.get('Content-Length') ?? '0', 10);
  if (!total || !res.body) return res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(Math.min(99, (received / total) * 100));
  }

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out.buffer;
}

async function loadModelBuffer(
  onProgress?: Sapiens2ProgressCallback,
): Promise<{ buffer: ArrayBuffer; fromCache: boolean }> {
  // Try Cache API first — avoids re-downloading 116 MB on every visit.
  // Keyed on the canonical HuggingFace URL (stable), not any signed redirect.
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const cached = await cache.match(MODEL_URL);
      if (cached) {
        onProgress?.(100, true);
        return { buffer: await cached.arrayBuffer(), fromCache: true };
      }
    } catch { /* Cache API unavailable (private browsing, etc.) — fall through */ }
  }

  const buffer = await downloadWithProgress(MODEL_URL, (pct) => onProgress?.(pct, false));

  // Persist to cache in the background — don't block session creation.
  if (typeof caches !== 'undefined') {
    caches.open(MODEL_CACHE_NAME)
      .then(cache => cache.put(MODEL_URL, new Response(buffer.slice(0), {
        headers: { 'Content-Type': 'application/octet-stream' },
      })))
      .catch(() => {}); // best-effort; failure just means next visit re-downloads
  }

  return { buffer, fromCache: false };
}

// Reused across all imageToTensor calls — avoids per-image canvas alloc
let _canvas: HTMLCanvasElement | null = null;
let _ctx: CanvasRenderingContext2D | null = null;

// Reused tensor buffer — avoids allocating 9.4 MB on every image and the
// resulting GC pressure. Safe to reuse because session.run() is awaited
// before the buffer is overwritten for the next image.
const _pixels = MODEL_H * MODEL_W;
const _tensorBuf = new Float32Array(3 * _pixels);

function getCanvas(): CanvasRenderingContext2D {
  if (!_canvas || !_ctx) {
    _canvas = document.createElement('canvas');
    _canvas.width = MODEL_W;
    _canvas.height = MODEL_H;
    _ctx = _canvas.getContext('2d', { willReadFrequently: true })!;
  }
  return _ctx;
}

function imageToTensor(source: ImageBitmap): ort.Tensor {
  const ctx = getCanvas();
  // source is already MODEL_W × MODEL_H — omit dest dimensions to skip the
  // redundant scaling pass inside drawImage.
  ctx.drawImage(source, 0, 0);
  const { data } = ctx.getImageData(0, 0, MODEL_W, MODEL_H);

  for (let i = 0; i < _pixels; i++) {
    _tensorBuf[i]               = data[i * 4]     * R_SCALE - R_OFF;
    _tensorBuf[_pixels + i]     = data[i * 4 + 1] * G_SCALE - G_OFF;
    _tensorBuf[2 * _pixels + i] = data[i * 4 + 2] * B_SCALE - B_OFF;
  }
  return new ort.Tensor('float32', _tensorBuf, [1, 3, MODEL_H, MODEL_W]);
}

async function decodeBitmap(src: File | ImageBitmap): Promise<{ bmp: ImageBitmap; owned: boolean }> {
  if (src instanceof File) {
    const bmp = await createImageBitmap(src, {
      resizeWidth: MODEL_W,
      resizeHeight: MODEL_H,
      resizeQuality: 'medium',
    });
    return { bmp, owned: true };
  }
  return { bmp: src, owned: false };
}

/**
 * Load the Sapiens2 ONNX model. Tries WebGPU first, falls back to WASM.
 * The model is cached in the browser's Cache API after the first download.
 */
export async function loadSapiens2(
  onProgress?: Sapiens2ProgressCallback,
): Promise<{ session: Sapiens2Session; device: 'webgpu' | 'wasm'; fromCache: boolean }> {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  ort.env.logLevel = 'error';
  // Use as many threads as the browser allows.
  // SharedArrayBuffer (required for multithreading) is available when the page
  // is served with COOP: same-origin + COEP: credentialless headers.
  // Without those headers ORT silently falls back to a single thread.
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

  const { buffer: modelBuffer, fromCache } = await loadModelBuffer(onProgress);

  // int8-quantized ONNX models have very limited WebGPU kernel coverage in
  // onnxruntime-web: the session appears to load on WebGPU but most ops fall
  // back to single-threaded WASM, giving worse throughput than multithreaded
  // WASM. Skip the WebGPU attempt for this model.
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return { session, device: 'wasm', fromCache };
}

/**
 * Embed a list of images using the Sapiens2 ONNX session (one at a time).
 * Accepts File or ImageBitmap; File objects are decoded internally.
 * Returns L2-normalized Float32Array of length 768 per image.
 *
 * Pipeline: decoding of image i+1 starts before inference on image i completes.
 * createImageBitmap runs off the main thread, so its ~200ms overlaps with the
 * ~450ms WASM inference, hiding most of the decode latency.
 */
export async function embedWithSapiens2(
  session: Sapiens2Session,
  sources: (File | ImageBitmap)[],
): Promise<Float32Array[]> {
  if (sources.length === 0) return [];

  const results: Float32Array[] = [];

  // Kick off decode for the first image immediately
  let nextDecode = decodeBitmap(sources[0]);

  for (let i = 0; i < sources.length; i++) {
    const { bmp, owned } = await nextDecode;

    // Start decoding image i+1 before running inference on image i.
    // createImageBitmap is off-thread, so this overlaps with the ~450ms
    // WASM session.run below and hides most of the decode cost.
    if (i + 1 < sources.length) {
      nextDecode = decodeBitmap(sources[i + 1]);
    }

    try {
      const tensor = imageToTensor(bmp);
      const output = await session.run({ pixel_values: tensor });
      const raw = output['embedding'].data as Float32Array;
      // l2normalize returns a new Float32Array (copy), so _tensorBuf is safe to reuse
      results.push(l2normalize(raw));
    } finally {
      if (owned) bmp.close();
    }
  }

  return results;
}
