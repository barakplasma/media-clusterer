/**
 * Sapiens2 ONNX embedder for browser inference via onnxruntime-web.
 * Model: barakplasma/sapiens2-onnx (facebook/sapiens2-pretrain-0.1b)
 * Output: 768-dim L2-normalized float32 vector per image.
 */

import * as ort from 'onnxruntime-web';
import { l2normalize } from './embeddings';

export type Sapiens2Variant = 'int8' | 'fp16' | 'fp32';

const HF_HOST = 'https://huggingface.co';
const REPO_PATH = 'barakplasma/sapiens2-onnx/resolve/main';

// Per-variant filename and Cache API bucket.
// fp16 keeps the v2 cache name so existing downloads aren't re-fetched.
const VARIANT_CONFIG: Record<Sapiens2Variant, { file: string; cacheName: string; sizeMB: number }> = {
  int8: { file: 'sapiens2_0.1b_int8.onnx', cacheName: 'sapiens2-model-int8-v1', sizeMB: 116 },
  fp16: { file: 'sapiens2_0.1b_fp16.onnx', cacheName: 'sapiens2-model-v2',      sizeMB: 229 },
  fp32: { file: 'sapiens2_0.1b_fp32.onnx', cacheName: 'sapiens2-model-fp32-v1', sizeMB: 458 },
};

/**
 * Full download URL for a Sapiens2 variant. `host` overrides the HuggingFace
 * host (e.g. a corporate mirror); falsy `host` falls back to huggingface.co.
 */
export function sapiens2Url(variant: Sapiens2Variant, host?: string): string {
  const base = (host && host.trim() ? host.trim() : HF_HOST).replace(/\/+$/, '');
  return `${base}/${REPO_PATH}/${VARIANT_CONFIG[variant].file}`;
}

/** Per-load overrides for offline / proxy fallback. */
export interface Sapiens2LoadOptions {
  host?: string;               // alternative HuggingFace-compatible host
  uploadedBuffer?: ArrayBuffer; // pre-supplied model bytes (skips download)
}

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
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal });
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

// Persist a model buffer to the Cache API in the background. Keyed on the
// canonical HuggingFace URL (stable) so downloads via a proxy host or local
// uploads still hit the cache on the next visit.
function persistToCache(cacheName: string, cacheKey: string, buffer: ArrayBuffer): void {
  if (typeof caches === 'undefined') return;
  caches.open(cacheName)
    .then(cache => cache.put(cacheKey, new Response(buffer.slice(0), {
      headers: { 'Content-Type': 'application/octet-stream' },
    })))
    .catch((err) => {
      console.warn('Sapiens2 model cache write failed (quota?):', err);
    });
}

async function loadModelBuffer(
  cfg: typeof VARIANT_CONFIG[Sapiens2Variant],
  cacheKey: string,
  downloadUrl: string,
  opts: Sapiens2LoadOptions | undefined,
  onProgress?: Sapiens2ProgressCallback,
  signal?: AbortSignal,
): Promise<{ buffer: ArrayBuffer; fromCache: boolean }> {
  // A locally-uploaded buffer bypasses the network entirely (offline fallback).
  if (opts?.uploadedBuffer) {
    onProgress?.(100, false);
    persistToCache(cfg.cacheName, cacheKey, opts.uploadedBuffer);
    return { buffer: opts.uploadedBuffer, fromCache: false };
  }

  // Try Cache API first — avoids re-downloading on every visit.
  // Keyed on the canonical HuggingFace URL (stable), not any signed redirect.
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(cfg.cacheName);
      const cached = await cache.match(cacheKey);
      if (cached) {
        onProgress?.(100, true);
        return { buffer: await cached.arrayBuffer(), fromCache: true };
      }
    } catch { /* Cache API unavailable (private browsing, etc.) — fall through */ }
  }

  // Request durable storage before writing a large model — on Android Chrome
  // without this the browser may evict Cache Storage entries freely.
  if (navigator.storage?.persist) {
    navigator.storage.persist().catch(() => {});
  }

  const buffer = await downloadWithProgress(downloadUrl, (pct) => onProgress?.(pct, false), signal);

  // Persist to cache in the background — don't block session creation.
  persistToCache(cfg.cacheName, cacheKey, buffer);

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
 * Load a Sapiens2 ONNX model. Tries WebGPU first, falls back to WASM.
 * The model is cached in the browser's Cache API after the first download.
 */
export type Sapiens2FallbackReason =
  | 'no-webgpu'       // navigator.gpu absent
  | 'no-adapter'      // requestAdapter() returned null
  | 'vram-limit'      // adapter maxStorageBufferBindingSize < model size
  | 'device-error'    // requestDevice() threw
  | 'session-error';  // InferenceSession.create() threw on WebGPU

export async function loadSapiens2(
  variant: Sapiens2Variant = 'fp16',
  onProgress?: Sapiens2ProgressCallback,
  signal?: AbortSignal,
  opts?: Sapiens2LoadOptions,
): Promise<{ session: Sapiens2Session; device: 'webgpu' | 'wasm'; fromCache: boolean; fallbackReason?: Sapiens2FallbackReason }> {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  ort.env.logLevel = 'error';
  // Use as many threads as the browser allows.
  // SharedArrayBuffer (required for multithreading) is available when the page
  // is served with COOP: same-origin + COEP: credentialless headers.
  // Without those headers ORT silently falls back to a single thread.
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

  const cfg = VARIANT_CONFIG[variant];
  // Cache is always keyed on the canonical Hub URL so a proxy host or local
  // upload still reuses (and populates) the same cache entry.
  const cacheKey = sapiens2Url(variant);
  const downloadUrl = sapiens2Url(variant, opts?.host);
  const { buffer: modelBuffer, fromCache } = await loadModelBuffer(
    cfg, cacheKey, downloadUrl, opts, onProgress, signal,
  );
  signal?.throwIfAborted();

  // fp16 ONNX has full WebGPU kernel coverage — try WebGPU first.
  // Falls back to multithreaded WASM (numThreads set above) if WebGPU is
  // unavailable or fails.
  // logSeverityLevel: 3 (error) suppresses benign WASM-level [W:] warnings:
  // - constant_folding Tile node (fp16 has no CPU kernel for constant folding)
  // - VerifyEachNodeIsAssignedToAnEp (shape ops intentionally stay on CPU)
  const sessionOpts: ort.InferenceSession.SessionOptions = {
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };

  // Try to pre-create a WebGPU device with the adapter's actual hardware limits.
  // ORT's default device uses the WebGPU spec minimum (128 MB storage buffer
  // binding size), which is too small for the fp16 model (~228 MB). By
  // requesting the adapter's own maxStorageBufferBindingSize we get whatever
  // the GPU actually supports (often 2 GB+ on desktop), then hand it to ORT
  // before the first session is created.
  let fallbackReason: Sapiens2FallbackReason | undefined;

  const gpuDevice = await (async (): Promise<GPUDevice | null> => {
    if (!navigator.gpu) { fallbackReason = 'no-webgpu'; return null; }
    let adapter: GPUAdapter | null;
    try {
      adapter = await navigator.gpu.requestAdapter();
    } catch { fallbackReason = 'no-adapter'; return null; }
    if (!adapter) { fallbackReason = 'no-adapter'; return null; }
    const maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    if (maxStorageBufferBindingSize < modelBuffer.byteLength) {
      fallbackReason = 'vram-limit';
      return null;
    }
    try {
      return await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize } });
    } catch { fallbackReason = 'device-error'; return null; }
  })();

  if (gpuDevice) {
    try {
      // ort.env.webgpu.device must be set before the first WebGPU session.
      ort.env.webgpu.device = gpuDevice;
      const session = await ort.InferenceSession.create(modelBuffer, {
        ...sessionOpts,
        executionProviders: ['webgpu'],
      });
      return { session, device: 'webgpu', fromCache };
    } catch { fallbackReason = 'session-error'; }
  }

  const session = await ort.InferenceSession.create(modelBuffer, {
    ...sessionOpts,
    executionProviders: ['wasm'],
  });
  return { session, device: 'wasm', fromCache, fallbackReason };
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
