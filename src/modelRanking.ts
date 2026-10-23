/**
 * Order the Settings model and projection lists by measured quality, best
 * first, with the best model this device can run on top as the recommendation
 * and models it cannot run struck through.
 *
 * The order comes from the benchmark (bench/, written to benchmarkRanks.ts by
 * bench/rank.mjs), not from impressions.
 */

import { MODEL_RANK, PROJECTION_RANK } from './benchmarkRanks'
import type { ModelVariant, ProjectionMethod } from './types'

/** What this browser and device can do, as far as the model list cares. */
export interface DeviceCaps {
  /** A real WebGPU adapter exists (navigator.gpu alone is not enough). */
  webgpu: boolean
  /** Chrome's built-in Gemini Nano (Prompt API) is not 'unavailable'. */
  chromeAI: boolean
}

export interface RankedOption<T extends string> {
  value: T
  /** Why it cannot run here, or null when it can. */
  blocked: string | null
  recommended: boolean
}

/** Why `variant` cannot run on a device with `caps`, or null if it can. */
export function blockedReason(variant: string, caps: DeviceCaps): string | null {
  // q4 EmbeddingGemma 2 uses GatherBlockQuantized, which onnxruntime-web
  // implements only on WebGPU. Every other local model has a WASM path.
  if (variant === 'embeddinggemma2' && !caps.webgpu) return 'needs WebGPU'
  if (variant === 'chrome-ai' && !caps.chromeAI) return 'needs Chrome built-in AI'
  return null
}

function byRank<T extends string>(values: readonly T[], rank: readonly string[]): T[] {
  const pos = (v: string) => {
    const i = rank.indexOf(v)
    return i < 0 ? rank.length : i
  }
  // Array.prototype.sort is stable, so unranked values keep their order.
  return [...values].sort((a, b) => pos(a) - pos(b))
}

/**
 * The model list for this device: the recommended model first, then every
 * other model best to worst. A remote endpoint is never the recommendation,
 * since it needs the user's own server and uploads thumbnails.
 */
export function rankModels(values: readonly ModelVariant[], caps: DeviceCaps): RankedOption<ModelVariant>[] {
  const ordered = byRank(values, MODEL_RANK).map((value) => ({
    value,
    blocked: blockedReason(value, caps),
    recommended: false
  }))
  const pick = ordered.find((o) => !o.blocked && o.value !== 'openai')
  if (!pick) return ordered
  pick.recommended = true
  return [pick, ...ordered.filter((o) => o !== pick)]
}

/** Projections best to worst; the first is marked as the recommendation. */
export function rankProjections(values: readonly ProjectionMethod[]): RankedOption<ProjectionMethod>[] {
  return byRank(values, PROJECTION_RANK).map((value, i) => ({ value, blocked: null, recommended: i === 0 }))
}

/**
 * Strike text through with U+0336 COMBINING LONG STROKE OVERLAY. CSS
 * text-decoration is ignored inside <option> by most browsers' native
 * pickers (Android's included); combining characters render everywhere.
 */
export function strike(text: string): string {
  return [...text].map((c) => (c === ' ' ? c : c + '̶')).join('')
}
