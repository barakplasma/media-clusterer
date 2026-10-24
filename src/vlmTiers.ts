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
 * Width of `image_features`, the only output of SmolVLM2's `vision_encoder.onnx`:
 * the post-connector tensor (64 tokens × 576), not SigLIP's 768-d hidden state.
 * Measured on the q4f16 export; `assertVisionDim()` catches a future change.
 */
const SMOLVLM_IMAGE_FEATURES = 576

const TIERS: Record<string, VlmTier> = {
  'smolvlm2-vision': {
    id: 'smolvlm2-vision',
    family: 'smolvlm2',
    repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: true,
    visionDim: SMOLVLM_IMAGE_FEATURES,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-vision-576/', // 768-d era vectors never existed (the call failed)
    downloadMB: 190, // vision encoder + embed_tokens + decoder, see vlmWorker load()
    // The only tier the worker can actually honour today.
    selectable: true,
    queryPrefix: null
  },
  'smolvlm2-256m': {
    id: 'smolvlm2-256m',
    family: 'smolvlm2',
    repo: 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: false,
    visionDim: SMOLVLM_IMAGE_FEATURES,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-256m/',
    downloadMB: 189,
    // Not offered until M4 builds the decoder path. The worker loads only
    // vision_encoder, so selecting this today would deliver tier A behaviour
    // under a label promising captions.
    selectable: false,
    queryPrefix: null
  },
  'smolvlm2-500m': {
    id: 'smolvlm2-500m',
    family: 'smolvlm2',
    repo: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    dtype: 'q4f16',
    visionOnly: false,
    visionDim: SMOLVLM_IMAGE_FEATURES,
    framesPerVideo: 4,
    cachePrefix: '@smolvlm2-500m/',
    downloadMB: 358,
    selectable: false, // as above — needs M4
    queryPrefix: null
  },
  // Multimodal embedder: images, video frames and text share one 768-d space,
  // so the same worker answers text search. q4 is the model card's WebGPU
  // pick; q4f16 is out because the activations overflow fp16 (Google's docs).
  // Audio encoder is never loaded (worker drops audio_config).
  embeddinggemma2: {
    id: 'embeddinggemma2',
    family: 'embeddinggemma2',
    repo: 'onnx-community/embeddinggemma-2-ONNX',
    dtype: 'q4',
    visionOnly: true,
    visionDim: 768,
    // 8 frames × 140 soft tokens stays well under the ~2,700-token WebGPU
    // dispatch limit the model card warns about.
    framesPerVideo: 8,
    cachePrefix: '@embeddinggemma2-q4/',
    downloadMB: 316, // text 175 + vision 109 + tokenizer 32
    selectable: true,
    // Asymmetric retrieval: query prefix on text; media takes no prefix.
    queryPrefix: 'task: search result | query: ',
    imageTokens: 280
  }
}

// A phone locked up running EmbeddingGemma 2 at the default budget: ~2,500
// vision patches per image, and attention memory grows with their square.
// 70 tokens (~630 patches) cuts that ~16x. Its own cache prefix, because a
// different budget embeds the same photo differently (AGENT.md).
// Chromebooks take the same row: an Intel gen-12lp one lost the GPU
// (VK_ERROR_DEVICE_LOST) on the first photo at the default budget (Bugsink
// MEDIA-CLUSTERER-21/23/25).
// shortcut: UA sniffing, upgrade to the adapter's limits once pickVlmTier() (M6) exists.
// fixedSeqLen fits the longest input, a 4-frame video: 4 × (70 soft tokens +
// boi/eoi) + bos/eos = 292. Queries and photos are padded up to it.
const PHONE_OVERRIDES: Partial<Record<string, Partial<VlmTier>>> = {
  embeddinggemma2: {
    imageTokens: 70,
    framesPerVideo: 4,
    fixedSeqLen: 296,
    cachePrefix: '@embeddinggemma2-q4-t70-s296/'
  }
}
const IS_PHONE =
  typeof navigator !== 'undefined' && /Android|iPhone|iPad|iPod|CrOS/i.test(navigator.userAgent)

// Settings → Image budget: 'light' / 'full' override the UA guess above. Read
// straight from storage because tiers are looked up before settings load; a
// change reloads the page (app.ts), so reading once is enough.
function savedProfile(): string | undefined {
  try {
    return JSON.parse(localStorage.getItem('mc_settings') ?? '{}').embedProfile
  } catch {
    return undefined
  }
}
const profile = savedProfile()
const USE_LIGHT = profile === 'light' ? true : profile === 'full' ? false : IS_PHONE

/** True when `variant` is one of the SmolVLM2 tiers. */
export function isVlmVariant(variant: ModelVariant): boolean {
  return variant in TIERS
}

/**
 * Tier descriptor for a variant, or null when the variant is not a VLM tier.
 * On phones, tiers with a PHONE_OVERRIDES row come back with its lighter settings.
 */
export function getVlmTier(variant: ModelVariant, phone = USE_LIGHT): VlmTier | null {
  const tier = TIERS[variant]
  if (!tier) return null
  const phoneRow = phone ? PHONE_OVERRIDES[variant] : undefined
  return phoneRow ? { ...tier, ...phoneRow } : tier
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
 * A worker tier can only when its own checkpoint embeds text into the same
 * space (`queryPrefix` set). SmolVLM2 tiers cannot: `state.vectors` hold pooled
 * SigLIP features, and the app's other text encoder, nomic-embed-text, lives
 * in an unrelated space. Equal width (768) is not compatibility — cosine over
 * the two is well-defined and meaningless, which is worse than an error
 * because it looks like it worked.
 */
export function supportsTextSearch(variant: ModelVariant): boolean {
  const tier = getVlmTier(variant)
  return tier ? tier.queryPrefix !== null : true
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
 * Guard that `vision_encoder.onnx` still exports `visionDim`-wide features
 * (ADR-0001 assumed 768-d SigLIP; the export is the 576-d post-connector tensor).
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
