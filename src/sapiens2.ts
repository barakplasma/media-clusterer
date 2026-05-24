/**
 * Sapiens2 ONNX embedder for browser inference via onnxruntime-web.
 * Model: barakplasma/sapiens2-onnx (facebook/sapiens2-pretrain-0.1b, int8)
 * Output: 768-dim L2-normalized float32 vector per image.
 */

import * as ort from 'onnxruntime-web';
import { l2normalize } from './embeddings';

const MODEL_URL =
  'https://huggingface.co/barakplasma/sapiens2-onnx/resolve/main/sapiens2_0.1b_int8.onnx';

// Input resolution expected by the model
const MODEL_H = 1024;
const MODEL_W = 768;

// ImageNet normalization
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

export type Sapiens2Session = ort.InferenceSession;

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

function imageToTensor(source: ImageBitmap): ort.Tensor {
  const canvas = document.createElement('canvas');
  canvas.width = MODEL_W;
  canvas.height = MODEL_H;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, MODEL_W, MODEL_H);
  const { data } = ctx.getImageData(0, 0, MODEL_W, MODEL_H);

  const pixels = MODEL_H * MODEL_W;
  const t = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) {
    t[i]             = (data[i * 4]     / 255 - MEAN[0]) / STD[0];
    t[pixels + i]    = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    t[2 * pixels + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return new ort.Tensor('float32', t, [1, 3, MODEL_H, MODEL_W]);
}

/**
 * Load the Sapiens2 ONNX model. Tries WebGPU first, falls back to WASM.
 * Returns the session and the device that was used.
 */
export async function loadSapiens2(
  onProgress?: (pct: number) => void,
): Promise<{ session: Sapiens2Session; device: 'webgpu' | 'wasm' }> {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

  const modelBuffer = await downloadWithProgress(MODEL_URL, onProgress);

  try {
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    return { session, device: 'webgpu' };
  } catch {
    const session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return { session, device: 'wasm' };
  }
}

/**
 * Embed a list of images using the Sapiens2 ONNX session (one at a time).
 * Accepts File or ImageBitmap; File objects are decoded internally.
 * Returns L2-normalized Float32Array of length 768 per image.
 */
export async function embedWithSapiens2(
  session: Sapiens2Session,
  sources: (File | ImageBitmap)[],
): Promise<Float32Array[]> {
  const results: Float32Array[] = [];

  for (const src of sources) {
    let bmp: ImageBitmap;
    let owned = false;

    if (src instanceof File) {
      bmp = await createImageBitmap(src);
      owned = true;
    } else {
      bmp = src;
    }

    try {
      const tensor = imageToTensor(bmp);
      const output = await session.run({ pixel_values: tensor });
      const raw = output['embedding'].data as Float32Array;
      results.push(l2normalize(raw));
    } finally {
      if (owned) bmp.close();
    }
  }

  return results;
}
