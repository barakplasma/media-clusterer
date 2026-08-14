import { describe, it, expect } from 'vitest'
import {
  poolVision,
  assertVisionDim,
  getVlmTier,
  isVlmVariant,
  allVlmTiers,
  selectableVlmTiers,
  supportsTextSearch,
  poolVectors,
  frameTimestamps,
  THUMB_FRAME_PX,
  VLM_FRAME_PX
} from './vlmTiers'
import type { VlmTier } from './types'

describe('poolVision', () => {
  it('mean-pools over the token axis and L2-normalizes', () => {
    // Two tokens, three dims. Means are [2.5, 3.5, 4.5] before normalization.
    const data = Float32Array.from([1, 2, 3, 4, 5, 6])
    const out = poolVision(data, [1, 2, 3])

    expect(out).toHaveLength(3)
    const norm = Math.hypot(...out)
    expect(norm).toBeCloseTo(1, 6)

    // Direction must match the un-normalized mean, or the pooling is wrong in
    // a way a magnitude check alone would not catch.
    const mean = [2.5, 3.5, 4.5]
    const meanNorm = Math.hypot(...mean)
    for (let d = 0; d < 3; d++) expect(out[d]).toBeCloseTo(mean[d] / meanNorm, 6)
  })

  it('reads rows with the right stride', () => {
    // Token 0 is all 1s, token 1 all 3s → mean is all 2s → normalized all equal.
    // A stride bug (e.g. reading columns) would give an uneven vector here.
    const data = Float32Array.from([1, 1, 1, 1, 3, 3, 3, 3])
    const out = poolVision(data, [1, 2, 4])
    for (let d = 0; d < 4; d++) expect(out[d]).toBeCloseTo(0.5, 6)
  })

  it('handles a single token', () => {
    const out = poolVision(Float32Array.from([3, 4]), [1, 1, 2])
    expect(out[0]).toBeCloseTo(0.6, 6)
    expect(out[1]).toBeCloseTo(0.8, 6)
  })

  it('does not divide by zero on an all-zero tensor', () => {
    const out = poolVision(new Float32Array(6), [1, 2, 3])
    expect([...out]).toEqual([0, 0, 0])
  })

  it('rejects a tensor that is not rank 3', () => {
    expect(() => poolVision(Float32Array.from([1, 2]), [1, 2])).toThrow(/rank 3/)
  })

  it('rejects a batch larger than 1', () => {
    expect(() => poolVision(new Float32Array(12), [2, 2, 3])).toThrow(/batch of 1/)
  })

  it('rejects a buffer shorter than the declared shape', () => {
    // Guards against silently reading past the end and pooling garbage.
    expect(() => poolVision(new Float32Array(5), [1, 2, 3])).toThrow(/expected 6/)
  })
})

describe('assertVisionDim', () => {
  const tier = getVlmTier('smolvlm2-vision') as VlmTier

  it('accepts the declared width', () => {
    expect(() => assertVisionDim(tier, tier.visionDim)).not.toThrow()
  })

  it('names the observed width and the fix when the ONNX export disagrees', () => {
    // ADR-0001 flags 576 as the plausible alternative (post-connector tensor).
    // The message has to carry the real number, since acting on it means
    // editing the tier table.
    expect(() => assertVisionDim(tier, 576)).toThrow(/576/)
    expect(() => assertVisionDim(tier, 576)).toThrow(/visionDim: 576/)
    expect(() => assertVisionDim(tier, 576)).toThrow(/cachePrefix/)
  })
})

describe('tier table', () => {
  it('recognizes SmolVLM2 variants and rejects the others', () => {
    expect(isVlmVariant('smolvlm2-vision')).toBe(true)
    expect(isVlmVariant('smolvlm2-256m')).toBe(true)
    expect(isVlmVariant('smolvlm2-500m')).toBe(true)
    expect(isVlmVariant('sapiens2-fp16')).toBe(false)
    expect(isVlmVariant('chrome-ai')).toBe(false)
    expect(isVlmVariant('openai')).toBe(false)
    expect(isVlmVariant('nomic')).toBe(false)
  })

  it('returns null for a non-VLM variant', () => {
    expect(getVlmTier('nomic')).toBeNull()
  })

  it('gives every tier its own cache prefix', () => {
    // Vectors from different models are not comparable; a shared prefix would
    // silently mix them in IndexedDB.
    const prefixes = allVlmTiers().map((t) => t.cachePrefix)
    expect(new Set(prefixes).size).toBe(prefixes.length)
    for (const p of prefixes) expect(p).toMatch(/^@smolvlm2-.+\/$/)
  })

  it('orders tiers by download size', () => {
    const sizes = allVlmTiers().map((t) => t.downloadMB)
    expect([...sizes]).toEqual([...sizes].sort((a, b) => a - b))
  })

  it('keeps only tier A vision-only', () => {
    expect(getVlmTier('smolvlm2-vision')?.visionOnly).toBe(true)
    expect(getVlmTier('smolvlm2-256m')?.visionOnly).toBe(false)
    expect(getVlmTier('smolvlm2-500m')?.visionOnly).toBe(false)
  })

  it('stays under the 1B-parameter intent: no tier exceeds the 500M download', () => {
    for (const t of allVlmTiers()) expect(t.downloadMB).toBeLessThanOrEqual(358)
  })

  it('keeps thumbnail and VLM frame sizes distinct', () => {
    // Collapsing these is the regression flagged in review: it would promote
    // every cached video thumbnail to the VLM size.
    expect(THUMB_FRAME_PX).toBe(224)
    expect(VLM_FRAME_PX).toBe(512)
    expect(THUMB_FRAME_PX).toBeLessThan(VLM_FRAME_PX)
  })
})

describe('frameTimestamps', () => {
  it('reproduces the pre-M3 single-frame behaviour exactly', () => {
    // The thumbnail path still calls this with n=1 and must not shift.
    expect(frameTimestamps(10, 1)).toEqual([1.0]) // min(1.0, 10/2)
    expect(frameTimestamps(1, 1)).toEqual([0.5]) // min(1.0, 1/2)
  })

  it('spreads n frames over the interior, avoiding both endpoints', () => {
    // First and last frames are usually black or a title card.
    const ts = frameTimestamps(10, 4)
    expect(ts).toEqual([2, 4, 6, 8])
    expect(ts[0]).toBeGreaterThan(0)
    expect(ts[ts.length - 1]).toBeLessThan(10)
  })

  it('returns exactly n timestamps, ascending', () => {
    for (const n of [1, 2, 3, 4, 8, 64]) {
      const ts = frameTimestamps(37.5, n)
      expect(ts).toHaveLength(n)
      for (let i = 1; i < ts.length; i++) expect(ts[i]).toBeGreaterThan(ts[i - 1])
    }
  })

  it('keeps every timestamp inside the clip', () => {
    const duration = 3.3
    for (const t of frameTimestamps(duration, 8)) {
      expect(t).toBeGreaterThan(0)
      expect(t).toBeLessThan(duration)
    }
  })

  it('falls back to 0 when the duration is unknown or degenerate', () => {
    // Some WebM streams report NaN/Infinity duration until fully buffered.
    expect(frameTimestamps(NaN, 3)).toEqual([0, 0, 0])
    expect(frameTimestamps(Infinity, 2)).toEqual([0, 0])
    expect(frameTimestamps(0, 2)).toEqual([0, 0])
    expect(frameTimestamps(-5, 1)).toEqual([0])
  })

  it('never returns an empty list', () => {
    // Callers index straight into the result; [] would be a silent no-frame bug.
    expect(frameTimestamps(10, 0)).toHaveLength(1)
    expect(frameTimestamps(10, -3)).toHaveLength(1)
  })
})

describe('selectableVlmTiers', () => {
  it('offers only tiers whose label the code can honour', () => {
    // The worker loads vision_encoder only, so the captioning tiers would
    // deliver tier-A behaviour under a label promising captions.
    const ids = selectableVlmTiers().map((t) => t.id)
    expect(ids).toEqual(['smolvlm2-vision'])
  })

  it('still declares the unreleased tiers', () => {
    // They stay in the table so M4 is a flag flip, not a re-derivation.
    expect(allVlmTiers().map((t) => t.id)).toContain('smolvlm2-256m')
    expect(allVlmTiers().map((t) => t.id)).toContain('smolvlm2-500m')
  })

  it('never offers a tier that claims captions before M4', () => {
    for (const t of selectableVlmTiers()) expect(t.visionOnly).toBe(true)
  })
})

describe('supportsTextSearch', () => {
  it('is false for every VLM tier', () => {
    // Pooled SigLIP vectors and nomic-embed-text live in unrelated spaces;
    // equal width is not compatibility.
    for (const t of allVlmTiers()) expect(supportsTextSearch(t.id)).toBe(false)
  })

  it('leaves the existing embedders alone', () => {
    expect(supportsTextSearch('nomic')).toBe(true)
    expect(supportsTextSearch('sapiens2-fp16')).toBe(true)
    expect(supportsTextSearch('chrome-ai')).toBe(true)
    expect(supportsTextSearch('openai')).toBe(true)
  })
})

describe('poolVectors', () => {
  it('returns a unit vector', () => {
    const out = poolVectors([
      Float32Array.from([1, 0, 0]),
      Float32Array.from([0, 1, 0])
    ])
    expect(Math.hypot(...out)).toBeCloseTo(1, 6)
  })

  it('averages frames rather than favouring one', () => {
    // Two orthogonal frames should land halfway between them, not on either.
    const out = poolVectors([
      Float32Array.from([1, 0]),
      Float32Array.from([0, 1])
    ])
    expect(out[0]).toBeCloseTo(Math.SQRT1_2, 6)
    expect(out[1]).toBeCloseTo(Math.SQRT1_2, 6)
  })

  it('passes a single frame through untouched', () => {
    const only = Float32Array.from([0.6, 0.8])
    expect(poolVectors([only])).toBe(only)
  })

  it('is order-independent', () => {
    const a = Float32Array.from([1, 0, 0])
    const b = Float32Array.from([0, 1, 0])
    const c = Float32Array.from([0, 0, 1])
    const x = poolVectors([a, b, c])
    const y = poolVectors([c, a, b])
    for (let i = 0; i < x.length; i++) expect(x[i]).toBeCloseTo(y[i], 6)
  })

  it('survives frames that cancel out', () => {
    // Opposed frames average to zero; must not divide by zero.
    const out = poolVectors([Float32Array.from([1, 0]), Float32Array.from([-1, 0])])
    expect([...out]).toEqual([0, 0])
  })

  it('rejects an empty group and mixed widths', () => {
    expect(() => poolVectors([])).toThrow(/no vectors/)
    expect(() =>
      poolVectors([Float32Array.from([1, 0]), Float32Array.from([1, 0, 0])])
    ).toThrow(/mixed widths/)
  })
})
