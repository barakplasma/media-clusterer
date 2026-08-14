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
    downloadMB: 55
  },
  'smolvlm2-256m': {
    id: 'smolvlm2-256m',
    repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: false,
    visionDim: SIGLIP_HIDDEN_SIZE,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-256m/',
    downloadMB: 189
  },
  'smolvlm2-500m': {
    id: 'smolvlm2-500m',
    repo: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: false,
    visionDim: SIGLIP_HIDDEN_SIZE,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-500m/',
    downloadMB: 358
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

/** All tiers, in ascending download size. Used to build the settings dropdown. */
export function allVlmTiers(): VlmTier[] {
  return Object.values(TIERS).sort((a, b) => a.downloadMB - b.downloadMB)
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
