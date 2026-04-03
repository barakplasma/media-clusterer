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
 * Extract vector from pipeline output
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

  // If we have a sequence [batch, seq, hidden], average across the sequence
  if (dims.length === 3) {
    const seqLen = dims[1];
    const pooled = new Float32Array(hiddenSize);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < hiddenSize; j++) {
        pooled[j] += data[i * hiddenSize + j];
      }
    }
    for (let j = 0; j < hiddenSize; j++) {
      pooled[j] /= seqLen;
    }
    return l2normalize(pooled);
  }

  // Already pooled [batch, hidden] or [hidden]
  return l2normalize(data);
}

/**
 * Create cache key from file metadata
 */
export function makeCacheKey(file: { name: string; size: number; lastModified: number }): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}
