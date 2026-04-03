/**
 * Embedding utilities for vision and text models
 */

import type { PipelineOutput, PipelineInstance } from './types';

/**
 * L2 normalize array to unit vector
 */
export function l2normalize(arr: number[] | Float32Array): Float32Array {
  let norm = 0;
  const len = arr.length;

  for (let i = 0; i < len; i++) {
    norm += arr[i] * arr[i];
  }

  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    out[i] = arr[i] / norm;
  }

  return out;
}

/**
 * Layer normalization followed by L2 normalization
 * (matches nomic-embed-text model card)
 */
export function layerNormAndNormalize(
  data: number[] | Float32Array,
  hiddenSize: number
): Float32Array {
  const len = data.length;

  // Layer norm
  const mean = Array.from(data).reduce((a, b) => a + b, 0) / hiddenSize;
  const variance = Array.from(data).reduce((a, b) => a + (b - mean) ** 2, 0) / hiddenSize;
  const epsilon = 1e-5;

  const normalized = new Float32Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    normalized[i] = (data[i] - mean) / Math.sqrt(variance + epsilon);
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < hiddenSize; i++) {
    norm += normalized[i] * normalized[i];
  }

  norm = Math.sqrt(norm) || 1;
  const vec = new Float32Array(hiddenSize);

  for (let i = 0; i < hiddenSize; i++) {
    vec[i] = normalized[i] / norm;
  }

  return vec;
}

/**
 * Extract vector from pipeline output
 * Handles Tensor, array of Tensors, or named objects
 */
export function extractVector(output: PipelineOutput): Float32Array {
  // Unwrap pipeline output: Tensor, array of Tensors, or named object
  let tensor: PipelineOutput | undefined = Array.isArray(output) ? output[0] : output;

  if (tensor && typeof tensor === 'object' && !('dims' in tensor)) {
    const obj = tensor as Record<string, unknown>;
    tensor = (obj.last_hidden_state ?? obj.pooler_output ?? Object.values(obj)[0]) as PipelineOutput;
  }

  if (!tensor) {
    throw new Error('Failed to extract tensor from pipeline output');
  }

  const dims = tensor.dims;
  const data = tensor.data;
  const hiddenSize = dims[dims.length - 1];

  // CLS token: first hiddenSize values in the flat data array
  const vec = new Float32Array(hiddenSize);
  for (let i = 0; i < hiddenSize; i++) {
    vec[i] = data[i];
  }

  // Use layerNormAndNormalize as required by Nomic-v1.5 to match the text/vision space
  return layerNormAndNormalize(vec, hiddenSize);
}

/**
 * Create cache key from file metadata
 */
export function makeCacheKey(file: { name: string; size: number; lastModified: number }): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}
