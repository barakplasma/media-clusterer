/**
 * Adaptive batch inference. GPU inference can fail (typically out-of-memory)
 * for a whole batch even though every image in it is fine at a smaller batch
 * size. Instead of zero-filling the entire batch (permanent data loss for
 * those files), bisect and retry until only genuinely bad inputs remain.
 */

/**
 * Run `embed` on `inputs`; on failure split the batch in half and retry each
 * half recursively. Only an input that fails alone gets `makeFallback`.
 * Results are returned in input order. `onFailure` fires once per failed
 * embed call (useful to shrink the working batch size).
 */
export async function embedBatchAdaptive<I, V>(
  inputs: I[],
  embed: (batch: I[]) => Promise<V[]>,
  makeFallback: (input: I, error: unknown) => V,
  onFailure?: (batchLength: number, error: unknown) => void,
): Promise<V[]> {
  if (inputs.length === 0) return [];
  try {
    return await embed(inputs);
  } catch (err) {
    onFailure?.(inputs.length, err);
    if (inputs.length === 1) {
      return [makeFallback(inputs[0], err)];
    }
    const mid = Math.ceil(inputs.length / 2);
    // Sequential, not parallel: after an OOM the last thing we want is two
    // half-size batches hitting the GPU at the same time.
    const left = await embedBatchAdaptive(inputs.slice(0, mid), embed, makeFallback, onFailure);
    const right = await embedBatchAdaptive(inputs.slice(mid), embed, makeFallback, onFailure);
    return left.concat(right);
  }
}

export interface AdaptiveBatcher {
  /** Current working batch size. */
  readonly size: number;
  recordSuccess(): void;
  recordFailure(): void;
}

/**
 * Tracks a working batch size: halve on failure, and after `growAfter`
 * consecutive successes grow by 25% back toward (never past) the initial size.
 */
export function createAdaptiveBatcher(
  initialSize: number,
  { growAfter = 5 }: { growAfter?: number } = {},
): AdaptiveBatcher {
  let size = Math.max(1, initialSize);
  let streak = 0;
  return {
    get size() { return size; },
    recordSuccess() {
      if (size >= initialSize) return;
      streak++;
      if (streak >= growAfter) {
        size = Math.min(initialSize, Math.ceil(size * 1.25));
        streak = 0;
      }
    },
    recordFailure() {
      size = Math.max(1, Math.floor(size / 2));
      streak = 0;
    },
  };
}
