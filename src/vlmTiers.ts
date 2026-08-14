/**
 * SmolVLM2 tier descriptors — see docs/adr/0001-small-video-language-model.md.
 *
 * The tier is *data*, not control flow. Adding a future sub-1B video model is a
 * new row here, not a new `variant.startsWith(...)` branch in app.ts. Anything
 * that needs to know "which model, how quantised, how many frames" reads it off
 * the descriptor rather than re-deriving it from the variant string.
 */

import type { ModelVariant, VlmTier } from './types'

// Longest-edge pixel sizes. These are deliberately different values, not one
// constant reused: the display thumbnail and the VLM input serve different
// purposes, and collapsing them silently multiplies thumbnail memory. See
// "Two resolutions, two purposes" in VIDEO_LM_PLAN.md.
export const THUMB_FRAME_PX = 224
export const VLM_FRAME_PX = 512

/**
 * `vision_config.hidden_size` for both SmolVLM2 checkpoints, and — not by
 * coincidence — the width every other embedder in this app already produces,
 * which is what lets pooled vision features drop straight into the existing
 * IndexedDB cache, DruidJS projection and k-means.
 *
 * Treated as a per-tier field rather than a global constant because it is
 * pending confirmation against the ONNX graph: if `vision_encoder.onnx` turns
 * out to export the post-connector tensor it is 576, and only this number
 * changes. `assertVisionDim()` below is what catches that at runtime.
 */
const SIGLIP_HIDDEN_SIZE = 768

const TIERS: Record<string, VlmTier> = {
  'smolvlm2-vision': {
    id: 'smolvlm2-vision',
    repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: true,
    visionDim: SIGLIP_HIDDEN_SIZE,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-vision/',
    downloadMB: 55,
    // The only tier the worker can actually honour today.
    selectable: true
  },
  'smolvlm2-256m': {
    id: 'smolvlm2-256m',
    repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: false,
    visionDim: SIGLIP_HIDDEN_SIZE,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-256m/',
    downloadMB: 189,
    // Not offered until M4 builds the decoder path. The worker loads only
    // vision_encoder, so selecting this today would deliver tier A behaviour
    // under a label promising captions.
    selectable: false
  },
  'smolvlm2-500m': {
    id: 'smolvlm2-500m',
    repo: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: false,
    visionDim: SIGLIP_HIDDEN_SIZE,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-500m/',
    downloadMB: 358,
    selectable: false // as above — needs M4
  }
}

/** True when `variant` is one of the SmolVLM2 tiers. */
export function isVlmVariant(variant: ModelVariant): boolean {
  return variant in TIERS
}

/** Tier descriptor for a variant, or null when the variant is not a VLM tier. */
export function getVlmTier(variant: ModelVariant): VlmTier | null {
  return TIERS[variant] ?? null
}

/** Every declared tier, including ones not yet offered to users. */
export function allVlmTiers(): VlmTier[] {
  return Object.values(TIERS).sort((a, b) => a.downloadMB - b.downloadMB)
}

/**
 * Tiers a user may actually pick, ascending by download size.
 *
 * A tier is listed only once the code can deliver what its label claims.
 * Advertising "captions + embeddings" while the worker loads nothing but the
 * vision encoder would be a worse outcome than not offering it at all.
 */
export function selectableVlmTiers(): VlmTier[] {
  return allVlmTiers().filter((t) => t.selectable)
}

/**
 * Whether a variant can answer a typed text query.
 *
 * False for every VLM tier: `state.vectors` hold pooled SigLIP features, and
 * the only text encoder in the app is nomic-embed-text, whose output lives in
 * an unrelated space. Equal width (768) is not compatibility — cosine over the
 * two is well-defined and meaningless, which is worse than an error because it
 * looks like it worked. A compatible path arrives with M6.
 */
export function supportsTextSearch(variant: ModelVariant): boolean {
  return !isVlmVariant(variant)
}

/**
 * Combine per-frame vectors into the single vector stored for an item.
 *
 * Mean-then-renormalize: each frame already contributes a unit vector, so the
 * mean is the centroid of the clip's visual content and renormalizing keeps the
 * result comparable with single-image vectors under cosine.
 */
export function poolVectors(vectors: Float32Array[]): Float32Array {
  if (vectors.length === 0) throw new Error('poolVectors: no vectors to pool')
  if (vectors.length === 1) return vectors[0]

  const dim = vectors[0].length
  for (const v of vectors) {
    if (v.length !== dim) {
      throw new Error(`poolVectors: mixed widths (${dim} vs ${v.length})`)
    }
  }

  const out = new Float32Array(dim)
  for (const v of vectors) for (let d = 0; d < dim; d++) out[d] += v[d]
  for (let d = 0; d < dim; d++) out[d] /= vectors.length

  let norm = 0
  for (let d = 0; d < dim; d++) norm += out[d] * out[d]
  norm = Math.sqrt(norm) || 1
  for (let d = 0; d < dim; d++) out[d] /= norm
  return out
}

/**
 * Mean-pool a `[1, tokens, dim]` vision-encoder output down to one `dim`-wide
 * L2-normalized vector.
 *
 * Kept separate from the worker so it is unit-testable without a model: this is
 * the one piece of real arithmetic on the tier-A path, and getting the stride
 * wrong would produce plausible-looking garbage rather than an error.
 */
export function poolVision(data: Float32Array, dims: readonly number[]): Float32Array {
  if (dims.length !== 3) {
    throw new Error(`Expected vision features of rank 3 [batch, tokens, dim], got [${dims.join(', ')}]`)
  }
  const [batch, tokens, dim] = dims
  if (batch !== 1) {
    throw new Error(`Expected batch of 1 from the vision encoder, got ${batch}`)
  }
  if (tokens < 1 || dim < 1) {
    throw new Error(`Degenerate vision feature shape [${dims.join(', ')}]`)
  }
  if (data.length < tokens * dim) {
    throw new Error(`Vision feature buffer holds ${data.length} values, expected ${tokens * dim}`)
  }

  const sums = new Float32Array(dim)
  for (let t = 0; t < tokens; t++) {
    const base = t * dim
    for (let d = 0; d < dim; d++) sums[d] += data[base + d]
  }
  for (let d = 0; d < dim; d++) sums[d] /= tokens

  // l2normalize is imported lazily by the caller to keep this module free of
  // side effects; do the normalisation inline so poolVision is self-contained.
  let norm = 0
  for (let d = 0; d < dim; d++) norm += sums[d] * sums[d]
  norm = Math.sqrt(norm) || 1
  for (let d = 0; d < dim; d++) sums[d] /= norm

  return sums
}

/**
 * Timestamps to seek to when sampling `n` frames from a clip of `duration`.
 *
 * Frames are spread uniformly over the *interior* of the clip, avoiding both
 * endpoints: the first frame of a video is frequently black or a title card,
 * and the last is often a fade-out, so sampling at 0 and `duration` reliably
 * wastes two of a small budget on the least informative frames.
 *
 * n=1 reproduces the pre-M3 behaviour exactly — `min(1.0, duration / 2)` — so
 * the thumbnail path is unchanged by the move to multi-frame.
 */
export function frameTimestamps(duration: number, n: number): number[] {
  const count = Math.max(1, Math.floor(n))
  if (!Number.isFinite(duration) || duration <= 0) {
    // Unknown duration (some WebM streams): seek to 0 and take what we get.
    return new Array(count).fill(0)
  }
  if (count === 1) return [Math.min(1.0, duration / 2)]

  // n interior points: duration * i/(n+1) for i in 1..n.
  const out: number[] = []
  for (let i = 1; i <= count; i++) out.push((duration * i) / (count + 1))
  return out
}

/**
 * Guard the assumption recorded in ADR-0001: that `vision_encoder.onnx` exports
 * the pre-connector SigLIP tensor and is therefore `visionDim` wide.
 *
 * A mismatch is not fatal to correctness — vectors of any width cluster fine,
 * and each tier writes under its own cache prefix — but it must not pass
 * silently, because the cached vectors would then disagree with the descriptor
 * that named them. Fail loudly with the number to put in the tier instead.
 */
export function assertVisionDim(tier: VlmTier, actualDim: number): void {
  if (actualDim !== tier.visionDim) {
    throw new Error(
      `${tier.id}: vision encoder produced ${actualDim}-d features, tier declares ${tier.visionDim}-d. ` +
        `If ${actualDim} is correct (e.g. the ONNX export includes the pixel-shuffle connector), ` +
        `set visionDim: ${actualDim} in src/vlmTiers.ts and bump that tier's cachePrefix.`
    )
  }
}
