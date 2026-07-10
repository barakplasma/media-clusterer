import { describe, it, expect } from 'vitest';
import { embedBatchAdaptive, createAdaptiveBatcher } from './batching';

// Embedder that fails whenever the batch is larger than `limit`, or contains
// an input marked poison (fails even alone, like a corrupt image).
function makeEmbedder(limit: number) {
  const calls: number[] = [];
  const embed = async (batch: string[]): Promise<string[]> => {
    calls.push(batch.length);
    if (batch.length > limit) throw new Error(`OOM at ${batch.length}`);
    if (batch.some(s => s === 'poison')) throw new Error('corrupt input');
    return batch.map(s => `vec:${s}`);
  };
  return { embed, calls };
}

describe('embedBatchAdaptive', () => {
  it('embeds a healthy batch in a single call', async () => {
    const { embed, calls } = makeEmbedder(100);
    const out = await embedBatchAdaptive(['a', 'b', 'c'], embed, () => 'zero');
    expect(out).toEqual(['vec:a', 'vec:b', 'vec:c']);
    expect(calls).toEqual([3]);
  });

  it('bisects on failure and recovers every input without data loss', async () => {
    const { embed, calls } = makeEmbedder(2); // "GPU" only survives batches of <= 2
    const inputs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const out = await embedBatchAdaptive(inputs, embed, () => 'zero');
    expect(out).toEqual(inputs.map(s => `vec:${s}`)); // order preserved, no zeros
    expect(calls[0]).toBe(8); // first attempt was the full batch
    expect(Math.max(...calls.slice(1))).toBeLessThanOrEqual(4);
  });

  it('zero-fills only the truly failing input, keeps the rest', async () => {
    const { embed } = makeEmbedder(100);
    const out = await embedBatchAdaptive(['a', 'poison', 'c'], embed, input => `zero:${input}`);
    expect(out).toEqual(['vec:a', 'zero:poison', 'vec:c']);
  });

  it('returns [] for empty input without calling embed', async () => {
    const { embed, calls } = makeEmbedder(100);
    expect(await embedBatchAdaptive([], embed, () => 'zero')).toEqual([]);
    expect(calls).toEqual([]);
  });

  it('reports failures via onFailure', async () => {
    const { embed } = makeEmbedder(1);
    const failures: number[] = [];
    await embedBatchAdaptive(['a', 'b'], embed, () => 'zero', len => failures.push(len));
    expect(failures).toEqual([2]);
  });
});

describe('createAdaptiveBatcher', () => {
  it('halves the working size on failure, never below 1', () => {
    const b = createAdaptiveBatcher(16);
    expect(b.size).toBe(16);
    b.recordFailure();
    expect(b.size).toBe(8);
    for (let i = 0; i < 10; i++) b.recordFailure();
    expect(b.size).toBe(1);
  });

  it('grows back slowly after consecutive successes, capped at the initial size', () => {
    const b = createAdaptiveBatcher(16, { growAfter: 3 });
    b.recordFailure(); // 8
    b.recordFailure(); // 4
    b.recordSuccess();
    b.recordSuccess();
    expect(b.size).toBe(4); // not yet
    b.recordSuccess(); // 3rd consecutive → grow
    expect(b.size).toBe(5); // ceil(4 * 1.25)
    for (let i = 0; i < 100; i++) b.recordSuccess();
    expect(b.size).toBe(16); // capped at initial
  });

  it('a failure resets the success streak', () => {
    const b = createAdaptiveBatcher(16, { growAfter: 2 });
    b.recordFailure(); // 8
    b.recordSuccess();
    b.recordFailure(); // 4, streak reset
    b.recordSuccess();
    expect(b.size).toBe(4);
    b.recordSuccess();
    expect(b.size).toBe(5);
  });
});
