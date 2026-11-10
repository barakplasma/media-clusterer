/**
 * Main application logic for Media Clusterer
 */

import * as Sentry from '@sentry/browser'
import './sentry'
import exifr from '@modernized/exifr'
import { pipeline, env, RawImage } from '@huggingface/transformers'
import { projectInWorker } from './projection'
import pLimit from 'p-limit'
import { loadSapiens2, embedWithSapiens2 } from './sapiens2'
import { loadVlm, embedWithVlm, embedTextWithVlm, disposeVlm, vlmDevice } from './vlm'
import { median, recordBench, fastestAccelerator, webnnOrder, type Accelerator, type Candidate } from './benchStore'
import { rankModels, rankProjections, strike, type DeviceCaps } from './modelRanking'
import { MODEL_BADGES, PROJECTION_RANK } from './benchmarkRanks'
import {
  getVlmTier,
  isVlmVariant,
  selectableVlmTiers,
  supportsTextSearch,
  frameTimestamps,
  THUMB_FRAME_PX,
  VLM_FRAME_PX,
  isPowerVRRenderer,
  USE_WEBNN,
  useWebnn,
  DRAFT_VIDEO_FRAMES,
  videoDraftPrefix,
  WEBNN_DEVICES
} from './vlmTiers'
import type { Sapiens2Session, Sapiens2Variant, Sapiens2FallbackReason } from './sapiens2'
import {
  modelDownloadUrls,
  buildUploadCache,
  normalizeHost,
  isDownloadError
} from './modelFallback'
import {
  getChromeAIAvailability,
  downloadChromeAI,
  ChromeAISessionManager,
  DEFAULT_DESCRIBE_PROMPT
} from './chromeAI'
import type { LanguageModelAvailability } from './chromeAI'
import {
  normalizeBaseUrl,
  imageToDataURL,
  embedImages,
  embedQuery,
  probeDimension,
  lookupImageEmbeddingModels,
  openaiCacheNamespace,
  getOpenAIKey,
  setOpenAIKey,
  isOpenAIKeyRemembered,
  hasOpenAIConsent,
  recordOpenAIConsent,
  openaiHost,
  describeOpenAIError,
  type OpenAICompatConfig
} from './openaiCompat'
import { l2normalize, extractVector, extractBatchedVectors, makeCacheKey } from './embeddings'
import {
  openDB,
  cacheGet,
  cacheGetBatch,
  cachePutBatch,
  cacheStats,
  captionGetBatch,
  captionPutBatch,
  cacheKeys,
  cacheHasBatch,
  exportAll
} from './db'
import { readProvenance, type Provenance } from './aiProvenance'
import { loadFaceModels, findFaces, groupFaces, FACE_PREFIX, FACE_STRIDE, type FaceModels } from './faces'
import { BUNDLE_FILE, encodeBundle, decodeBundle, rekeyBundle, sha256Hex, fileKeyOf } from './embeddingBundle'
import { runOvernight, formatDuration, type ChildMsg, type OvernightModel, type OvernightStats, type RunMsg } from './overnight'
import {
  DEFAULT_FILTER,
  applyFileFilter,
  buildEmbeddedSets,
  fileKey,
  isVideoName,
  type EmbeddedSets,
  type FileFilter
} from './fileFilter'
import { getNextImageInDirection } from './spatial'
import {
  THUMB_WORLD,
  kmeansAsync,
  mostSimilarIndex,
  clusterCaptionLabels,
  spreadPointsAsync,
  generateMetadataBasedLayout,
  cullAndPrioritize,
  searchByCosine,
  formatEta
} from './compute'
import '@picocss/pico/css/pico.conditional.min.css'
import { computeOptimalBatchSize, getMemoryPressure } from './hardware'
import { embedBatchAdaptive, createAdaptiveBatcher } from './batching'
import type {
  AppState,
  Camera,
  DOMElements,
  PhotoFile,
  Point,
  Pipeline,
  PipelineInstance,
  ProgressEvent,
  Settings,
  ProjectionMethod,
  DirectoryHandle,
  FileSystemHandle,
  PointerState,
  CanvasPointerPos,
  CacheKey,
  ModelVariant,
  VlmTier
} from './types'

// Enable caching and local model access for persistent storage
env.allowLocalModels = false
env.allowRemoteModels = true
env.cacheDir = 'models'

// Point Transformers.js at an alternative HuggingFace-compatible host when the
// user has configured one (corporate proxy / Artifactory). Called at startup
// and whenever the setting changes via the download-fallback modal.
function applyModelEnv() {
  const host = normalizeHost(state.settings.customModelHost)
  env.remoteHost = host || 'https://huggingface.co'
}

// ── Constants ────────────────────────────────────────────────────────────────
// A hidden overnight worker frame (src/overnight.ts): embeds the files its parent
// posts, writes the cache, and never touches settings or shows the setup wizard.
const OVERNIGHT_CHILD = new URLSearchParams(location.search).has('overnight-child')
const postToParent = (m: ChildMsg) => window.parent.postMessage(m, location.origin)

const IMAGE_EXTS = new Set([
  'jpg',
  'jpeg',
  'png',
  'gif',
  'webp',
  'avif',
  'bmp',
  'tiff',
  'tif',
  'heic',
  'heif'
])
const VIDEO_EXTS = new Set(['mp4', 'webm'])
const FULL_LOD_SIZE = 120 // screen px at which we switch from thumb to full-res
const IS_MOBILE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)
const BATCH_SIZE = IS_MOBILE ? 4 : 16 // fallback before settings load
const MAX_DRAW_PER_FRAME = IS_MOBILE ? 150 : 400
const MAX_THUMBNAILS_CACHE = 2000 // Max decoded thumbnails to keep in memory (LRU)
const FULL_TILE_BUDGET = (IS_MOBILE ? 128 : 256) * 1024 * 1024 // decoded bytes of zoomed-in tiles (LRU)
const CLUSTER_COLORS = [
  '#f87171',
  '#fb923c',
  '#facc15',
  '#4ade80',
  '#38bdf8',
  '#818cf8',
  '#f472b6',
  '#a78bfa'
]

// ── Demo Images (Unsplash API, public authentication) ──────────────────────────
const UNSPLASH_ACCESS_KEY = import.meta.env.VITE_UNSPLASH_ACCESS_KEY || ''
const DEMO_CATEGORIES = [
  'dog',
  'cat',
  'horse',
  'butterfly',
  'spider',
  'chicken',
  'elephant',
  'sheep',
  'cow',
  'squirrel'
]

// ── DOM Elements ─────────────────────────────────────────────────────────────
const dom: DOMElements = {
  loadModelBtn: document.getElementById('load-model-btn') as HTMLButtonElement,
  resumeBtn: document.getElementById('resume-btn') as HTMLButtonElement,
  openBtn: document.getElementById('open-btn') as HTMLButtonElement,
  recenterBtn: document.getElementById('recenter-btn') as HTMLButtonElement,
  resetBtn: document.getElementById('reset-btn') as HTMLButtonElement,
  progressBar: document.getElementById('progress-bar') as HTMLDivElement,
  statusEl: document.getElementById('status') as HTMLDivElement,
  canvas: document.getElementById('canvas') as HTMLCanvasElement,
  modal: document.getElementById('modal') as HTMLDialogElement,
  modalClose: document.getElementById('modal-close') as HTMLButtonElement,
  modalNavLeft: document.getElementById('modal-nav-left') as HTMLButtonElement,
  modalNavRight: document.getElementById('modal-nav-right') as HTMLButtonElement,
  modalNavUp: document.getElementById('modal-nav-up') as HTMLButtonElement,
  modalNavDown: document.getElementById('modal-nav-down') as HTMLButtonElement,
  modalImg: document.getElementById('modal-img') as HTMLImageElement,
  modalVideo: document.getElementById('modal-video') as HTMLVideoElement,
  modalFooter: document.getElementById('modal-footer') as HTMLDivElement,
  modalUp: document.getElementById('modal-up') as HTMLSpanElement,
  modalPath: document.getElementById('modal-path') as HTMLDivElement,
  modalFilename: document.getElementById('modal-filename') as HTMLDivElement,
  modalDatetime: document.getElementById('modal-datetime') as HTMLDivElement,
  modalMeta: document.getElementById('modal-meta') as HTMLSpanElement,
  modalGps: document.getElementById('modal-gps') as HTMLAnchorElement,
  modalExifBtn: document.getElementById('modal-exif-btn') as HTMLButtonElement,
  modalPrevBtn: document.getElementById('modal-prev-btn') as HTMLButtonElement,
  modalNextBtn: document.getElementById('modal-next-btn') as HTMLButtonElement,
  searchWrap: document.getElementById('search-wrap') as HTMLDivElement,
  searchInput: document.getElementById('search-input') as HTMLInputElement,
  searchClearBtn: document.getElementById('search-clear-btn') as HTMLButtonElement,
  fileInput: document.getElementById('file-input') as HTMLInputElement,
  aboutBtn: document.getElementById('about-btn') as HTMLButtonElement,
  aboutModal: document.getElementById('about-modal') as HTMLDialogElement,
  shortcutsBtn: document.getElementById('shortcuts-btn') as HTMLButtonElement,
  slideshowModal: document.getElementById('slideshow-modal') as HTMLDialogElement,
  slideshowClose: document.getElementById('slideshow-close') as HTMLButtonElement,
  slideshowSimilarBtn: document.getElementById('slideshow-similar-btn') as HTMLButtonElement,
  searchPercentSlider: document.getElementById('search-percent-slider') as HTMLInputElement,
  searchPercentLabel: document.getElementById('search-percent-label') as HTMLSpanElement,
  slideshowSecondsSlider: document.getElementById('slideshow-seconds-slider') as HTMLInputElement,
  slideshowSecondsLabel: document.getElementById('slideshow-seconds-label') as HTMLSpanElement,
  slideshowCutVideosToggle: document.getElementById(
    'slideshow-cut-videos-toggle'
  ) as HTMLInputElement,
  modalRandomBtn: document.getElementById('modal-random-btn') as HTMLButtonElement,
  modalPlayBtn: document.getElementById('modal-play-btn') as HTMLButtonElement,
  modalPosition: document.getElementById('modal-position') as HTMLSpanElement,
  shortcutsModal: document.getElementById('shortcuts-modal') as HTMLDialogElement,
  shortcutsClose: document.getElementById('shortcuts-close') as HTMLButtonElement,
  aboutClose: document.getElementById('about-close') as HTMLButtonElement,
  statsEl: document.getElementById('stats') as HTMLDivElement,
  failedChip: document.getElementById('failed-chip') as HTMLButtonElement,
  settingsBtn: document.getElementById('settings-btn') as HTMLButtonElement,
  settingsModal: document.getElementById('settings-modal') as HTMLDialogElement,
  settingsClose: document.getElementById('settings-close') as HTMLButtonElement,
  densitySlider: document.getElementById('density-slider') as HTMLInputElement,
  loopToggle: document.getElementById('loop-toggle') as HTMLInputElement,
  drawBudgetSlider: document.getElementById('draw-budget-slider') as HTMLInputElement,
  enableSearchToggle: document.getElementById('enable-search-toggle') as HTMLInputElement,
  projectionSelect: document.getElementById('projection-select') as HTMLSelectElement,
  viewerOnlyToggle: document.getElementById('viewer-only-toggle') as HTMLInputElement,
  lazyCaptionToggle: document.getElementById('lazy-caption-toggle') as HTMLInputElement,
  doNotTrackToggle: document.getElementById('do-not-track-toggle') as HTMLInputElement,
  batchSizeInput: document.getElementById('batch-size-slider') as HTMLInputElement,
  batchSizeAutoBtn: document.getElementById('batch-size-auto-btn') as HTMLButtonElement,
  randomSampleSizeInput: document.getElementById('random-sample-size') as HTMLInputElement,
  bottomPanel: document.getElementById('bottom-panel') as HTMLDivElement,
  headerRecenterBtn: document.getElementById('header-recenter-btn') as HTMLButtonElement,
  demoBtn: document.getElementById('demo-btn') as HTMLButtonElement,
  modelSelect: document.getElementById('model-select') as HTMLSelectElement,
  embedProfileSelect: document.getElementById('embed-profile-select') as HTMLSelectElement,
  modalCaption: document.getElementById('modal-caption') as HTMLDivElement,
  chromeAIPromptInput: document.getElementById('chrome-ai-prompt') as HTMLTextAreaElement,
  chromeAIPromptReset: document.getElementById('chrome-ai-prompt-reset') as HTMLButtonElement,
  chromeAIPromptSetting: document.getElementById('chrome-ai-prompt-setting') as HTMLDivElement,
  customModelHostInput: document.getElementById('custom-model-host') as HTMLInputElement,
  openaiSetting: document.getElementById('openai-setting') as HTMLDivElement,
  openaiBaseUrl: document.getElementById('openai-base-url') as HTMLInputElement,
  openaiKey: document.getElementById('openai-key') as HTMLInputElement,
  openaiRemember: document.getElementById('openai-remember') as HTMLInputElement,
  openaiModel: document.getElementById('openai-model') as HTMLInputElement,
  openaiModelList: document.getElementById('openai-model-list') as HTMLDataListElement,
  openaiFindModelsBtn: document.getElementById('openai-find-models') as HTMLButtonElement,
  openaiModelsResult: document.getElementById('openai-models-result') as HTMLDivElement,
  openaiTestBtn: document.getElementById('openai-test') as HTMLButtonElement,
  openaiTestResult: document.getElementById('openai-test-result') as HTMLDivElement,
  openaiConsentModal: document.getElementById('openai-consent-modal') as HTMLDialogElement,
  openaiConsentHost: document.getElementById('openai-consent-host') as HTMLElement,
  openaiConsentAccept: document.getElementById('openai-consent-accept') as HTMLButtonElement,
  openaiConsentCancel: document.getElementById('openai-consent-cancel') as HTMLButtonElement,
  modelFallbackModal: document.getElementById('model-fallback-modal') as HTMLDialogElement,
  modelFallbackClose: document.getElementById('model-fallback-close') as HTMLButtonElement,
  modelFallbackUrls: document.getElementById('model-fallback-urls') as HTMLUListElement,
  modelFallbackFile: document.getElementById('model-fallback-file') as HTMLInputElement,
  modelFallbackFileHint: document.getElementById('model-fallback-file-hint') as HTMLDivElement,
  modelFallbackHost: document.getElementById('model-fallback-host') as HTMLInputElement,
  modelFallbackCancel: document.getElementById('model-fallback-cancel') as HTMLButtonElement,
  modelFallbackRetry: document.getElementById('model-fallback-retry') as HTMLButtonElement
}

// ── Version Info ─────────────────────────────────────────────────────────────
const versionEl = document.getElementById('version-info') as HTMLElement | null
if (versionEl) {
  versionEl.textContent = `${__GIT_BRANCH__}@${__GIT_COMMIT__}`
  versionEl.style.display = ''
}

// ── Constants ────────────────────────────────────────────────────────────────
// ── Auto batch size ───────────────────────────────────────────────────────────

// Default where WebGPU is missing: the benchmark (bench/README.md) has nomic
// ahead of SmolVLM2 on both datasets, and it has text search.
const NO_WEBGPU_MODEL = 'nomic'
// Set when a WebGPU-only model was swapped out for this session; saveSettings
// writes the saved choice back instead of the stand-in.
let modelStandIn: { saved: string; used: string } | null = null

const DEFAULT_SETTINGS: Settings = {
  density: 1.0,
  loopVideos: true,
  drawBudget: IS_MOBILE ? 150 : 400,
  searchPercent: 20,
  slideshowSeconds: 3,
  slideshowCutVideos: false,
  enableTextSearch: true, // the default model embeds text itself, so search costs no extra download
  // The benchmark's best projection; a saved choice still wins.
  projectionMethod: PROJECTION_RANK[0],
  batchSize: IS_MOBILE ? 4 : 16,
  randomSampleSize: 100,
  viewerOnly: false,
  // EmbeddingGemma 2 runs on WebGPU, or on WebNN for PowerVR GPUs
  // (src/vlmWorker.ts); loadModelOnce swaps in NO_WEBGPU_MODEL when there is
  // no adapter behind navigator.gpu.
  modelVariant: 'gpu' in navigator || USE_WEBNN ? 'embeddinggemma2' : NO_WEBGPU_MODEL,
  embedProfile: 'auto',
  enableLazyCaption: false,
  findFaces: false,
  doNotTrack: false,
  customModelHost: '',
  openai: { baseUrl: '', model: '', maxImageWidth: 384 }
}

const savedSettings = localStorage.getItem('mc_settings')
// Migrate legacy modelVariant values to the new named variants
if (savedSettings) {
  const parsed = JSON.parse(savedSettings)
  if (!parsed.modelVariant) {
    parsed.modelVariant = 'sapiens2-fp16'
    localStorage.setItem('mc_settings', JSON.stringify(parsed))
  } else if (parsed.modelVariant === 'sapiens2') {
    // 'sapiens2' was the generic name for fp16 — keep pointing at the same model
    parsed.modelVariant = 'sapiens2-fp16'
    localStorage.setItem('mc_settings', JSON.stringify(parsed))
  }
}
const settings: Settings = savedSettings
  ? { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem('mc_settings')!) }
  : DEFAULT_SETTINGS
// The spread above is shallow, so a settings blob written before `openai`
// existed — or one holding a partially-filled object — would leave nested keys
// undefined. Fill them in explicitly.
settings.openai = { ...DEFAULT_SETTINGS.openai, ...settings.openai }

const state: AppState = {
  phase: 'idle',
  files: [],
  vectors: [],
  points: [],
  rawPoints: null,
  clusters: null,
  thumbnails: [],
  captions: [],
  searchResults: null,
  searchQuery: '',
  searchScores: null,
  currentDirHandle: null,
  currentBasePath: '',
  settings,
  activeFileIndex: null,
  lastViewedIndex: null
}

// ── Camera (infinite canvas) ─────────────────────────────────────────────────
const camera: Camera = { x: 0, y: 0, scale: 1 }

const WEBGPU_HELP_KEY = 'mc_hideWebGpuHelp'
// Set while the WebNN model loads. WebNN holds the whole ~880 MB model several times over; on an 8 GB
// Chromebook the OS killed the tab mid-load (npu-test.html). Still set at the next start = that happened.
const WEBNN_LOADING_KEY = 'mc_webnnLoading'

/**
 * Tell a Chromium user how to turn WebGPU on when the browser has it but
 * hides the GPU (blocklisted driver, e.g. Pixel 10's PowerVR) or only offers
 * a software adapter. Stays up until dismissed. Returns false when not shown
 * (other browsers, or "Don't show again").
 */
function showWebGpuHelp(kind: 'blocked' | 'software', model: string): boolean {
  const ua = navigator.userAgent
  if (!('gpu' in navigator) || !/Chrome\//.test(ua)) return false
  try {
    if (localStorage.getItem(WEBGPU_HELP_KEY)) return false
  } catch (_) {}
  const scheme = /\bEdgA?\//.test(ua) ? 'edge' : /\bOPR\//.test(ua) ? 'opera' : 'chrome'
  const url = `${scheme}://flags/#enable-unsafe-webgpu`
  const el = (id: string) => document.getElementById(id)!
  el('webgpu-help-title').textContent =
    kind === 'blocked' ? '⚠️ WebGPU is turned off for this GPU' : '⚠️ WebGPU is running in software'
  el('webgpu-help-why').textContent =
    kind === 'blocked'
      ? `This browser supports WebGPU but hides your GPU from web pages, so ${model} can't run and the app is using ${NO_WEBGPU_MODEL} instead. To use ${model}:`
      : `This browser only offers a software (CPU) WebGPU adapter, so ${model} will be very slow. To use your GPU:`
  el('webgpu-help-url').textContent = url
  el('webgpu-help-accel').hidden = kind !== 'software'
  const modal = el('webgpu-help-modal') as HTMLDialogElement
  el('webgpu-help-copy').onclick = () => {
    void navigator.clipboard?.writeText(url).then(() => (el('webgpu-help-copy').textContent = 'Copied'))
  }
  el('webgpu-help-ok').onclick = () => modal.close()
  el('webgpu-help-close').onclick = () => modal.close()
  el('webgpu-help-never').onclick = () => {
    try {
      localStorage.setItem(WEBGPU_HELP_KEY, '1')
    } catch (_) {}
    modal.close()
  }
  modal.showModal()
  return true
}

// Pixel 10's PowerVR GPU (Chrome now exposes it without any flag, 2026-10):
// ~330 s per photo, the GPU watchdog resets, and loading EmbeddingGemma 2 on
// WebGPU froze the phone at ~95%. With WebNN on, EmbeddingGemma 2 runs there
// instead (USE_WEBNN in vlmTiers.ts), so the GPU no longer matters.
function isPowerVR(adapter: GPUAdapter | null | undefined): boolean {
  return adapter?.info?.vendor === 'img-tec' || isPowerVRRenderer()
}

// Pick the device before loading: after a failed WebGPU load, transformers.js
// 4.3 fails the wasm retry with the same WebGPU error (seen on a GPU-less
// Linux runner), so the catch-and-retry below alone never reached wasm.
// PowerVR (Pixel 10) counts as no GPU: its WebGPU is far too slow (see isPowerVR).
const webgpuOrWasm = async (): Promise<'webgpu' | 'wasm'> => {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null)
  return adapter && !isPowerVR(adapter) ? 'webgpu' : 'wasm'
}

// ── Model singleton ──────────────────────────────────────────────────────────
let extractor: PipelineInstance | null = null // Vision model (for images)
let textExtractor: PipelineInstance | null = null // Text model (for search queries)
let modelDevice: 'webgpu' | 'cpu' | null = null
let modelFallbackReason: Sapiens2FallbackReason | undefined
// Set by the download-fallback modal when the user uploads a Sapiens2 .onnx;
// consumed on the next loadModelOnce() and then cleared.
let pendingSapiens2Buffer: ArrayBuffer | null = null
let sapiens2Session: Sapiens2Session | null = null // Sapiens2 ONNX session
let chromeAIManager: ChromeAISessionManager | null = null // Chrome built-in AI session manager
let chromeAIDownload: Promise<void> | null = null // Gemini Nano download started by a Load Model click
let remoteConfig: OpenAICompatConfig | null = null // Remote OpenAI-compatible endpoint
let vlmTier: VlmTier | null = null // Loaded SmolVLM2 tier (worker-backed), see ADR-0001

// Width of the vectors the active backend produces. All the local backends are
// 768-d; a remote endpoint can be anything, so it is probed at load. Used for
// the zero vectors that stand in for failed embeddings — one of the wrong
// width would break every cosine score in the run.
let activeEmbeddingDim = 768

/** Placeholder for an embedding that could not be computed. */
function zeroVector(): Float32Array {
  return new Float32Array(activeEmbeddingDim)
}

/** zeroVector() for file `idx`, recorded so the map can badge or hide it. */
function failedVector(idx: number, reason: string): Float32Array {
  embedFailed.add(idx)
  if (/^(no frame could be|could not) decode/.test(reason)) undecodable.add(idx)
  // Without a reason the end-of-run report says only "1/100 failed" (Bugsink bc6155f9).
  embedFailReason ||= reason
  return zeroVector()
}
/** End-of-run failure line. A corrupt file is not an app error, so only model failures reach Bugsink. */
function reportEmbedFailures(total: number) {
  const why = embedFailReason ? `: ${embedFailReason}` : ''
  const log = undecodable.size === embedFailed.size ? console.warn : console.error
  log(`Embedding failed for ${embedFailed.size}/${total} files (${state.settings.modelVariant})${why}`)
}
let lazyCaptionManager: ChromeAISessionManager | null = null // On-demand caption for non-chrome-ai modes
let captionAbortController: AbortController | null = null // Cancels in-flight lazy caption
let captionDebounceTimer: ReturnType<typeof setTimeout> | null = null // Debounce before starting AI
let chromeAIAvailability: LanguageModelAvailability = 'unavailable' // Set at page load
let modelLoadAbort: AbortController | null = null // In-flight auto-start abort handle

const CHROME_AI_PROMPT_KEY = 'mc_chrome_ai_prompt'
const getChromeAIPrompt = () =>
  localStorage.getItem(CHROME_AI_PROMPT_KEY) ?? DEFAULT_DESCRIBE_PROMPT

// Refit the camera on layout changes until the user pans or zooms.
let cameraFollow = true

// ── Helpers ──────────────────────────────────────────────────────────────────
const welcomeEl = document.getElementById('welcome')!
const setStatus = (msg: string) => {
  dom.statusEl.textContent = msg
  document.getElementById('welcome-status')!.textContent = msg
}
const setProgress = (pct: number) => {
  if (OVERNIGHT_CHILD && state.phase === 'loading_model') postToParent({ type: 'loading', pct })
  const w = `${Math.min(100, Math.max(0, pct))}%`
  dom.progressBar.style.width = w
  document.getElementById('welcome-bar')!.style.width = w
}
/** The first-load card stands in for the empty map until files load. */
function showWelcome(show: boolean) {
  welcomeEl.hidden = !show
  document.body.classList.toggle('welcome', show)
}
const yieldMain = () => new Promise((resolve) => setTimeout(resolve, 0))

let toastTimer: ReturnType<typeof setTimeout> | null = null
function showToast(message: string, level: 'info' | 'warn' | 'error' = 'info', durationMs = 5000) {
  const el = document.getElementById('toast')!
  el.textContent = message
  el.className = `visible ${level === 'info' ? '' : level}`.trim()
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => {
    el.classList.remove('visible')
    toastTimer = null
  }, durationMs)
}

const cacheSizeEl = document.getElementById('cache-size')
const deviceBadgeEl = document.getElementById('device-badge')
const storageBadgeEl = document.getElementById('storage-badge')

function updateDeviceBadge() {
  if (!deviceBadgeEl) return

  if (state.settings.viewerOnly) {
    deviceBadgeEl.textContent = 'Viewer'
    deviceBadgeEl.style.color = '#4ade80'
  } else if (vlmTier) {
    const d = vlmDevice()
    const dev = d === 'webgpu' ? 'WebGPU' : d === 'wasm' ? 'WASM' : `WebNN ${d.slice(6).toUpperCase()}`
    deviceBadgeEl.textContent = `${vlmTier.id} · ${dev} · worker`
    deviceBadgeEl.style.color = d === 'wasm' || d === 'webnn-cpu' ? '#fbbf24' : '#4ade80'
  } else if (sapiens2Session) {
    const threads = navigator.hardwareConcurrency || 1
    const mt = typeof SharedArrayBuffer !== 'undefined'
    const variant = state.settings.modelVariant.split('-')[1] ?? 'fp16'
    let label: string
    if (modelDevice === 'webgpu') {
      label = `Sapiens2·${variant} · WebGPU`
    } else {
      const reason =
        modelFallbackReason === 'vram-limit'
          ? 'VRAM limit'
          : modelFallbackReason === 'no-webgpu'
            ? 'no WebGPU'
            : modelFallbackReason === 'no-adapter'
              ? 'no GPU'
              : modelFallbackReason === 'device-error'
                ? 'GPU error'
                : modelFallbackReason === 'session-error'
                  ? 'GPU error'
                  : null
      const suffix = reason ? ` · WASM (${reason})` : mt ? ` · ${threads}T` : ' · CPU'
      label = `Sapiens2·${variant}${suffix}`
    }
    deviceBadgeEl.textContent = label
    deviceBadgeEl.style.color = modelDevice === 'webgpu' ? '#4ade80' : '#fb923c'
  } else if (chromeAIManager) {
    deviceBadgeEl.textContent = 'Chrome AI · Built-in'
    deviceBadgeEl.style.color = '#4ade80'
  } else if (modelDevice === 'webgpu') {
    deviceBadgeEl.textContent = 'WebGPU'
    deviceBadgeEl.style.color = '#4ade80'
  } else if (modelDevice === 'cpu') {
    deviceBadgeEl.innerHTML =
      'CPU · <a href="https://webgpureport.org/" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">check WebGPU</a>'
    deviceBadgeEl.style.color = '#fb923c'
    deviceBadgeEl.title =
      'WebGPU not available — running on CPU (slower). Open webgpureport.org to check your GPU support.'
  } else {
    deviceBadgeEl.textContent = 'Local AI'
    deviceBadgeEl.style.color = ''
  }
}

function currentCachePrefix(v: ModelVariant = state.settings.modelVariant): string {
  if (v === 'openai') {
    // Computed from settings, not from `remoteConfig`, so the settings panel
    // can show a cache count before the endpoint has ever been contacted.
    const baseUrl = normalizeBaseUrl(state.settings.openai.baseUrl)
    if (!baseUrl || !state.settings.openai.model) return '@openai/'
    return openaiCacheNamespace({ baseUrl, apiKey: '', model: state.settings.openai.model })
  }
  if (v === 'chrome-ai') return '@chrome-ai/'
  const tier = getVlmTier(v)
  if (tier) return tier.cachePrefix
  if (!v.startsWith('sapiens2')) return ''
  if (v === 'sapiens2-fp16') return '@sapiens2/'
  return `@${v}/`
}

async function refreshCacheSize() {
  try {
    const { count } = await cacheStats(currentCachePrefix())
    const text = count === 0 ? '' : `${count} embeddings`
    if (cacheSizeEl) cacheSizeEl.textContent = text
    if (storageBadgeEl) {
      storageBadgeEl.textContent = text
      ;(storageBadgeEl as HTMLElement).style.display = text ? '' : 'none'
    }
  } catch (err) {
    if (cacheSizeEl) cacheSizeEl.textContent = ''
    if (storageBadgeEl) {
      storageBadgeEl.textContent = ''
      ;(storageBadgeEl as HTMLElement).style.display = 'none'
    }
  }
}

// ── Render scheduling (debounce to one frame) ────────────────────────────────
let renderPending = false
function scheduleRender() {
  if (!renderPending) {
    renderPending = true
    requestAnimationFrame(() => {
      render()
      renderPending = false
    })
  }
}

// ── Layout tween ─────────────────────────────────────────────────────────────
// render() eases from the previous layout to state.points, so photos glide to
// their new spot instead of jumping. state.points always holds the target, so
// saves and hit-tests see final positions.
const LAYOUT_TWEEN_MS = 450
let layoutTween: { from: Point[]; start: number } | null = null
let displayedPoints: Point[] = []

function setPointsAnimated(next: Point[]) {
  // Mid-tween, start from where photos are drawn right now, not the old target.
  const from = layoutTween ? displayedPoints : state.points
  state.points = next
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  layoutTween = from.length && !reduce ? { from, start: performance.now() } : null
}

// ── File collection ──────────────────────────────────────────────────────────
async function collectImages(
  dirHandle: DirectoryHandle,
  sampleSize: number = 0,
  basePath: string = ''
): Promise<PhotoFile[]> {
  // Phase 1: walk the tree and collect lightweight file references.
  // When sampleSize > 0, use reservoir sampling so memory stays bounded at O(sampleSize).
  type Ref = { name: string; handle: FileSystemHandle }
  const refs: Ref[] = []
  let seen = 0

  // Navigate to base path if specified
  let currentHandle = dirHandle
  if (basePath) {
    const pathParts = basePath.split('/').filter((p) => p)
    for (const part of pathParts) {
      currentHandle = await currentHandle.getDirectoryHandle(part)
    }
  }

  async function walk(handle: DirectoryHandle, prefix: string) {
    for await (const [name, entry] of handle as any) {
      if (name.startsWith('.')) continue
      if (entry.kind === 'directory') {
        await walk(entry as DirectoryHandle, `${prefix}${name}/`)
      } else {
        const ext = name.split('.').pop()?.toLowerCase() ?? ''
        if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
          const ref: Ref = {
            name: `${basePath}${prefix}${name}`,
            handle: entry as FileSystemHandle
          }
          if (sampleSize <= 0) {
            refs.push(ref)
          } else if (refs.length < sampleSize) {
            refs.push(ref)
          } else {
            // Reservoir sampling (Algorithm R): replace slot j with prob sampleSize/(seen+1)
            const j = Math.floor(Math.random() * (seen + 1))
            if (j < sampleSize) refs[j] = ref
          }
          seen++
        }
      }
    }
  }
  await walk(currentHandle, '')

  // Phase 2: fetch File objects only for the selected refs.
  // In parallel: one at a time took ~22 s for 100k files, since each call is a browser IPC round trip.
  const limit = pLimit(64)
  let done = 0
  return Promise.all(
    refs.map((ref) =>
      limit(async () => {
        const file = await (ref.handle as FileSystemFileHandle).getFile()
        if (++done % 2000 === 0) setStatus(`Reading file info… ${done}/${refs.length}`)
        return { name: ref.name, size: file.size, lastModified: file.lastModified, file, objectURL: null }
      })
    )
  )
}

// ── Demo images loader ──────────────────────────────────────────────────────
async function loadDemoImages(): Promise<PhotoFile[]> {
  const files: PhotoFile[] = []

  for (const category of DEMO_CATEGORIES) {
    try {
      // Fetch 3 random images from Unsplash for this category
      const url = `https://api.unsplash.com/photos/random?client_id=${UNSPLASH_ACCESS_KEY}&query=${category}&count=3&orientation=landscape`
      const res = await fetch(url)

      if (!res.ok) {
        console.warn(`Unsplash API error for ${category}:`, res.status)
        continue
      }

      const photos = (await res.json()) as Array<{
        id: string
        urls: { regular: string }
      }>

      for (let i = 0; i < photos.length; i++) {
        try {
          const photoUrl = photos[i].urls.regular
          const imgRes = await fetch(photoUrl)
          const blob = await imgRes.blob()
          const name = `${category}-${i + 1}.jpg`
          const file = new File([blob], name, { type: 'image/jpeg' })

          files.push({
            name,
            size: file.size,
            lastModified: Date.now(),
            file,
            objectURL: null
          })
        } catch (e) {
          console.warn(`Failed to load ${category} image ${i}:`, e)
        }
      }
    } catch (e) {
      console.warn(`Failed to fetch ${category} from Unsplash:`, e)
    }
  }

  return files
}

// ── Media helpers ────────────────────────────────────────────────────────────
/**
 * Sample `n` frames from a video, each resized to `px` on its longest edge.
 *
 * One `<video>` element is seeked `n` times and torn down once. Creating an
 * element per frame would multiply exactly the pressure the teardown and the
 * `pLimit(4)` wrapper exist to contain: Chrome caps concurrent WebMediaPlayers
 * at ~75 and starts refusing to decode past that.
 *
 * Resolution is a parameter, not a constant. Display thumbnails and VLM frames
 * are deliberately different sizes — collapsing them would promote every cached
 * video thumbnail to VLM resolution, over 5x the pixel memory per entry across
 * an LRU holding up to MAX_THUMBNAILS_CACHE entries, on the low-memory device
 * this whole feature targets.
 *
 * Returns fewer than `n` frames if some seeks fail, and [] if the video cannot
 * be decoded at all.
 */
async function extractVideoFrames(file: File, n: number, px: number): Promise<ImageBitmap[]> {
  return new Promise((resolve) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.playsInline = true
    const url = URL.createObjectURL(file)
    const frames: ImageBitmap[] = []
    let timestamps: number[] = []
    let cursor = 0
    let settled = false
    let probingDuration = false
    let timer: ReturnType<typeof setTimeout> | undefined

    // video.remove() is a no-op when the element was never added to the DOM.
    // The only way to release a WebMediaPlayer in Chrome is: pause → clear src → load().
    // Skipping this causes the "too many WebMediaPlayers" intervention at ~75 concurrent.
    // Null out all handlers before tearing down the element.
    // Setting video.src='' and video.load() can re-fire onerror on the same
    // handler, turning one failed decode into N cascading error logs.
    const cleanup = () => {
      video.onloadedmetadata = null
      video.onseeked = null
      video.onerror = null
      URL.revokeObjectURL(url)
      video.pause()
      video.src = ''
      video.load()
    }

    const finish = () => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      cleanup()
      resolve(frames)
    }

    const startSampling = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 0
      timestamps = frameTimestamps(duration, n)
      video.currentTime = timestamps[cursor]
    }

    video.onloadedmetadata = () => {
      // MediaRecorder WebMs (screen recordings, many Android recorders) carry
      // no duration, which sampled the first frame n times. Seeking far past
      // the end makes the browser scan to the end and fill in the real one.
      if (video.duration === Infinity && n > 1) {
        probingDuration = true
        video.currentTime = 1e101
        return
      }
      startSampling()
    }

    video.onseeked = async () => {
      if (probingDuration) {
        probingDuration = false
        startSampling()
        return
      }
      try {
        frames.push(await createImageBitmap(video, { resizeWidth: px, resizeQuality: 'medium' }))
      } catch (e) {
        // One bad seek should cost one frame, not the whole video.
        console.warn('Failed to extract video frame:', e)
      }
      cursor++
      if (cursor >= timestamps.length) {
        finish()
        return
      }
      video.currentTime = timestamps[cursor]
    }

    video.onerror = () => {
      // MEDIA_ERR_NETWORK on a blob: is usually a truncated file (index points past EOF) or one changed since it was picked.
      const e = video.error
      console.debug(`Video load error (${e ? `code ${e.code}${e.message ? `: ${e.message}` : ''}` : 'unknown'}):`, file.name)
      finish()
    }

    // A seek that never completes, or a file whose drive is still fetching it, would hold the run for good. An
    // overnight frame that hears nothing for 5 minutes is killed and its chunk halved down to the culprit,
    // a stall per halving: hours for one video. Give up and keep the frames taken so far.
    const limit = (window as Window & { __mcFrameTimeoutMs?: number }).__mcFrameTimeoutMs ?? 60_000
    timer = setTimeout(() => {
      console.debug(`Video frame extraction timed out after ${limit / 1000}s:`, file.name)
      finish()
    }, limit)

    video.src = url
  })
}

/** Single mid-video frame at thumbnail resolution — the pre-M3 behaviour. */
async function extractVideoFrame(file: File): Promise<ImageBitmap | null> {
  const [frame] = await extractVideoFrames(file, 1, THUMB_FRAME_PX)
  return frame ?? null
}

// ── Thumbnail preloader ──────────────────────────────────────────────────────
function initThumbnails(files: PhotoFile[]): (ImageBitmap | null)[] {
  // ObjectURLs are now created lazily on first use to avoid OOM with large folders
  return new Array<ImageBitmap | null>(files.length).fill(null)
}

function lazyDecodeThumbnail(idx: number) {
  if (thumbDecoding.has(idx) || state.thumbnails[idx] || thumbFailed.has(idx)) return
  thumbDecoding.add(idx)
  const f = state.files[idx]
  if (!f) {
    thumbDecoding.delete(idx)
    return
  }

  // Create objectURL lazily if needed (for large folders, don't create all upfront)
  if (!f.objectURL) f.objectURL = URL.createObjectURL(f.file)

  const ext = f.name.split('.').pop()?.toLowerCase() ?? ''

  const done = (bm: ImageBitmap | null) => {
    // The file list was replaced mid-decode: this bitmap belongs to an old file.
    if (state.files[idx] !== f) {
      bm?.close()
      return
    }
    state.thumbnails[idx] = bm
    thumbDecoding.delete(idx)

    // Only track in LRU when we have an actual bitmap — null entries (failed
    // decodes) would crowd out real thumbnails and trigger unnecessary evictions.
    if (!bm) {
      scheduleRender()
      return
    }

    thumbnailLRU.delete(idx)
    thumbnailLRU.add(idx)

    // Evict least recently used thumbnails if over cache limit
    while (thumbnailLRU.size > MAX_THUMBNAILS_CACHE) {
      const lruIdx = thumbnailLRU.values().next().value // Get first (oldest)
      if (lruIdx !== undefined) {
        thumbnailLRU.delete(lruIdx)
        const oldBm = state.thumbnails[lruIdx]
        if (oldBm?.close) oldBm.close()
        state.thumbnails[lruIdx] = null
      }
    }

    scheduleRender()
  }

  if (VIDEO_EXTS.has(ext)) {
    videoFrameLimit(() => extractVideoFrame(f.file))
      .then((result) => {
        if (!result && state.files[idx] === f) thumbFailed.add(idx)
        done(result)
      })
      .catch((err) => {
        if (state.files[idx] === f) thumbFailed.add(idx)
        done(null)
      })
  } else {
    createImageBitmap(f.file, { resizeWidth: 96, resizeQuality: 'low' })
      .then(done)
      .catch(() => done(null))
  }
}

// ── Text embedding (uses text model with search_query prefix) ────────────────
async function embedText(text: string): Promise<Float32Array> {
  // Queries must be embedded by whatever embedded the images, or the two sides
  // land in different spaces (AGENT.md, "Embedding space"). Note there is no
  // 'search_query:' prefix here: that scheme belongs to nomic, and applying it
  // to an arbitrary remote model would just embed a literal stray word.
  if (state.settings.modelVariant === 'openai') {
    if (!remoteConfig) throw new Error('Remote endpoint not connected')
    return embedQuery(remoteConfig, text)
  }

  const tier = getVlmTier(state.settings.modelVariant)
  if (tier?.queryPrefix != null) {
    // The worker's own text encoder shares the media vectors' space; its
    // retrieval scheme prefixes the query only (media take none).
    if (!vlmTier) throw new Error(`${tier.id} not loaded`)
    const [vec] = await embedTextWithVlm([tier.queryPrefix + text])
    return vec
  }

  if (!supportsTextSearch(state.settings.modelVariant)) {
    // Refusing beats answering. state.vectors hold pooled SigLIP features and
    // the only text encoder here is nomic-embed-text; the cosine between them
    // is well-defined and meaningless, so a query would return a confidently
    // ranked list of nothing. See ADR-0001 and M6.
    throw new Error(
      `Text search is not available with ${state.settings.modelVariant} — its image vectors ` +
        'and the text model are in different embedding spaces. Use search-by-example, or ' +
        'switch to another model.'
    )
  }

  if (!textExtractor) throw new Error('Text model not loaded')

  // Add required task prefix for nomic-embed-text
  const prefixed = `search_query: ${text}`
  const output = await textExtractor(prefixed, {
    pooling: 'mean',
    normalize: true
  })

  return Float32Array.from(output.data)
}

// ── Text search (exact cosine scan over normalized vectors) ─────────────────
async function searchImages(query: string) {
  selectPerson(null)
  if (!query.trim() || !state.vectors.length) {
    state.searchResults = null
    state.searchQuery = ''
    state.searchScores = null
    dom.searchWrap.classList.remove('loading')
    return
  }

  dom.searchWrap.classList.add('loading')
  setStatus(`Searching for "${query}"…`)

  try {
    const queryVector = await embedText(query)
    state.searchQuery = query

    const { indices, scores } = searchByCosine(queryVector, state.vectors)
    state.searchScores = scores
    state.searchResults = indices

    const topScore = scores[indices[0]] ?? 0
    const statusMsg = `${state.vectors.length} media files · top match: ${(topScore * 100).toFixed(0)}% similar`
    setStatus(statusMsg)
    if (dom.statsEl) dom.statsEl.textContent = statusMsg
  } catch (err) {
    // warn, not error: "model not loaded" / "no text search" are user states, not bugs
    // (Bugsink: "Text search is not available with smolvlm2-vision", "… not loaded")
    console.warn('Search failed:', err)
    setStatus(`Search failed: ${(err as Error).message}`)
  } finally {
    dom.searchWrap.classList.remove('loading')
  }
}

// ── Faces: find, group into people, highlight a person ──────────────────────
let faceModels: Promise<FaceModels> | null = null
let activePerson: number | null = null
let people: { files: number[]; best: { file: number; box: Float32Array } }[] = []
let facesRun = 0

/** Find faces in the open folder's photos (cached under FACE_PREFIX), then show the People strip. */
async function runFaces() {
  if (!state.settings.findFaces) return
  const run = ++facesRun
  const files = state.files
  const keys = files.map((f) => `${FACE_PREFIX}${makeCacheKey(f)}` as CacheKey)
  const found = await cacheGetBatch(keys)
  const todo = files.flatMap((f, i) => (found[i] || VIDEO_EXTS.has(f.name.split('.').pop()?.toLowerCase() ?? '') ? [] : [i]))
  if (todo.length) {
    try {
      setStatus('Loading face models…')
      faceModels ??= loadFaceModels(state.settings.customModelHost || undefined)
      const m = await faceModels
      const batch: [CacheKey, Float32Array][] = []
      for (let n = 0; n < todo.length; n++) {
        if (run !== facesRun || state.files !== files) return // another folder opened meanwhile
        const i = todo[n]
        try {
          const full = await createImageBitmap(files[i].file)
          const k = Math.min(1, 1280 / Math.max(full.width, full.height))
          const bmp = await createImageBitmap(full, { resizeWidth: Math.round(full.width * k), resizeHeight: Math.round(full.height * k) })
          full.close()
          found[i] = await findFaces(m, bmp)
          bmp.close()
          batch.push([keys[i], found[i]!])
        } catch (err) {
          console.warn('Face detection failed for one file:', files[i].name, (err as Error).message)
        }
        if (batch.length >= 20 || n === todo.length - 1) await cachePutBatch(batch.splice(0))
        if (n % 10 === 0) setStatus(`Finding faces… ${n + 1} / ${todo.length}`)
        await yieldMain()
      }
    } catch (err) {
      faceModels = null
      showToast(`Face detection unavailable: ${(err as Error).message}`, 'warn', 10000)
      return
    }
  }
  // One entry per face, then people with faces in at least two photos, biggest first.
  const owner: number[] = []
  const vecs: Float32Array[] = []
  const boxes: Float32Array[] = []
  found.forEach((p, i) => {
    for (let o = 0; p && o + FACE_STRIDE <= p.length; o += FACE_STRIDE) {
      owner.push(i)
      boxes.push(p.subarray(o, o + 5))
      vecs.push(p.subarray(o + 5, o + FACE_STRIDE))
    }
  })
  const ids = groupFaces(vecs, owner)
  const byPerson = new Map<number, number[]>()
  ids.forEach((p, f) => byPerson.set(p, [...(byPerson.get(p) ?? []), f]))
  const list = [...byPerson.values()]
    .map((faces) => ({
      files: [...new Set(faces.map((f) => owner[f]))],
      best: faces.reduce((b, f) => (boxes[f][4] * boxes[f][2] > boxes[b][4] * boxes[b][2] ? f : b))
    }))
    .sort((a, b) => b.files.length - a.files.length)
    .slice(0, 60)
    .map((p) => ({ files: p.files, best: { file: owner[p.best], box: boxes[p.best] } }))
  if (run !== facesRun) return
  showPeople(list)
  const inPhotos = found.filter((p) => p && p.length).length
  const repeat = list.filter((p) => p.files.length >= 2).length
  setStatus(
    vecs.length === 0
      ? `No faces found in ${todo.length ? 'the' : 'these'} ${files.length.toLocaleString()} files`
      : `${vecs.length.toLocaleString()} faces in ${inPhotos.toLocaleString()} photos · ${byPerson.size.toLocaleString()} people, ${repeat.toLocaleString()} in more than one photo`
  )
}

function showPeople(list: typeof people) {
  people = list
  selectPerson(null)
  const strip = document.getElementById('people-strip')!
  strip.hidden = list.length === 0
  strip.replaceChildren(
    ...list.map((p, i) => {
      const b = document.createElement('button')
      b.type = 'button'
      b.setAttribute('aria-pressed', 'false')
      b.setAttribute('aria-label', `Person ${i + 1}, ${p.files.length} photos`)
      b.title = `${p.files.length} photos`
      const c = document.createElement('canvas')
      c.width = c.height = 96
      const n = document.createElement('span')
      n.textContent = String(p.files.length)
      b.append(c, n)
      b.addEventListener('click', () => selectPerson(activePerson === i ? null : i))
      void drawFaceChip(c, state.files[p.best.file].file, p.best.box)
      return b
    })
  )
}

async function drawFaceChip(c: HTMLCanvasElement, file: File, box: Float32Array) {
  try {
    const bmp = await createImageBitmap(file, { resizeWidth: 512 })
    const [x, y, w, h] = [box[0] * bmp.width, box[1] * bmp.height, box[2] * bmp.width, box[3] * bmp.height]
    const side = Math.max(w, h) * 1.3 // a little margin around the face
    c.getContext('2d')!.drawImage(bmp, x + w / 2 - side / 2, y + h / 2 - side / 2, side, side, 0, 0, c.width, c.height)
    bmp.close()
  } catch (_) {
    // undecodable now: the chip stays blank but still works
  }
}

function selectPerson(i: number | null) {
  const was = activePerson
  activePerson = i
  document.querySelectorAll('#people-strip button').forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)))
  if (i !== null) {
    state.searchResults = Int32Array.from(people[i].files)
    state.searchScores = null
    state.searchQuery = ''
    dom.searchInput.value = ''
  } else if (was !== null) state.searchResults = null
  scheduleRender()
}

// ── Canvas ───────────────────────────────────────────────────────────────────
// Zoomed-in tiles are bitmaps resized near their on-screen size: a decoded 12 MP original is 48 MB.
interface FullTile {
  bmp: ImageBitmap
  px: number // long-side bucket it was decoded at
  frame: number // last frame that drew it
}
const fullImages = new Map<number, FullTile>() // index → tile, least recently drawn first
const fullPending = new Map<number, { px: number; frame: number }>() // index → queued or running decode
const fullFailed = new Set<number>() // undecodable files, so render() does not retry them every frame
let fullBytes = 0 // decoded bytes held by fullImages
let fullFrame = 0
const fullDecodeLimit = pLimit(4)

function dropFullTile(i: number) {
  const t = fullImages.get(i)
  if (!t) return
  fullBytes -= t.bmp.width * t.bmp.height * 4
  t.bmp.close()
  fullImages.delete(i)
}

function clearFullTiles(filesChanged: boolean) {
  for (const t of fullImages.values()) t.bmp.close()
  fullImages.clear()
  fullPending.clear() // decodes still running see their job gone and close their bitmap
  fullBytes = 0
  if (filesChanged) fullFailed.clear()
}

/** Closes tiles the current frame did not draw, oldest first, until at most `limit` bytes are held. */
function evictStaleFullTiles(limit: number) {
  for (const [i, t] of fullImages) {
    if (fullBytes <= limit || t.frame === fullFrame) break
    dropFullTile(i)
  }
}

function requestFullTile(i: number, px: number) {
  const f = state.files[i]
  const job = { px, frame: fullFrame }
  fullPending.set(i, job)
  void fullDecodeLimit(async () => {
    if (fullPending.get(i) !== job) return // scrolled away or files replaced while queued
    const thumb = state.thumbnails[i]
    let bmp: ImageBitmap | null = null
    try {
      // px bounds the long side; the thumbnail, when still cached, says which side that is.
      bmp = await createImageBitmap(f.file, {
        ...(thumb && thumb.height > thumb.width ? { resizeHeight: px } : { resizeWidth: px }),
        resizeQuality: 'medium'
      })
    } catch {
      // undecodable: recorded below
    }
    const current = fullPending.get(i) === job
    if (current) fullPending.delete(i)
    if (!current || state.files[i] !== f) {
      bmp?.close()
      return
    }
    if (!bmp) {
      fullFailed.add(i)
      return
    }
    dropFullTile(i) // the smaller bucket this one replaces
    fullImages.set(i, { bmp, px, frame: job.frame })
    fullBytes += bmp.width * bmp.height * 4
    scheduleRender()
  })
}
const thumbDecoding = new Set<number>() // indices currently being decoded
const thumbFailed = new Set<number>() // indices where video frame extraction permanently failed
const embedFailed = new Set<number>() // indices whose embedding failed (zero vector stand-in)
const undecodable = new Set<number>() // subset of embedFailed: the file itself could not be decoded
let hideFailed = false
let embedFailReason = '' // first error message of the run, shown in the end-of-run toast
const isHidden = (i: number) => hideFailed && embedFailed.has(i)
const thumbnailLRU = new Set<number>() // indices in LRU order (most recently used at end)

// Chrome limits concurrent WebMediaPlayers to ~75; cap video frame extraction
// well below that so the render loop + embedAll can't together exceed the limit.
const videoFrameLimit = pLimit(4)

// ── Caption storage ──────────────────────────────────────────────────────────
// Captions live in IndexedDB (db.ts v2). They used to be one synchronous
// localStorage.setItem per file inside the embed loop; see IMPROVEMENT_PLAN
// P1-1. Reads fall back to the old localStorage keys once and migrate them
// through, so captions generated by earlier builds survive the upgrade.

const LEGACY_CAPTION_PREFIX = '@caption/'

function captionKey(f: PhotoFile): CacheKey {
  return `${f.name}:${f.size}:${f.lastModified}`
}

function legacyCaptionKey(f: PhotoFile): string {
  return `${LEGACY_CAPTION_PREFIX}${captionKey(f)}`
}

/**
 * Read captions for a batch, migrating any that are still in localStorage.
 * Positional: entry i corresponds to files[i], null when there is no caption.
 */
async function readCaptions(files: PhotoFile[]): Promise<(string | null)[]> {
  if (files.length === 0) return []
  let stored: (string | null)[]
  try {
    stored = await captionGetBatch(files.map(captionKey))
  } catch {
    // A caption is a nice-to-have; never let its store take the embed down.
    stored = new Array(files.length).fill(null)
  }

  const migrate: [CacheKey, string][] = []
  for (let i = 0; i < files.length; i++) {
    if (stored[i]) continue
    const legacy = localStorage.getItem(legacyCaptionKey(files[i]))
    if (legacy) {
      stored[i] = legacy
      migrate.push([captionKey(files[i]), legacy])
    }
  }

  if (migrate.length > 0) {
    // Migrate then drop the localStorage copy, so the quota it was occupying
    // is actually returned rather than doubled.
    captionPutBatch(migrate)
      .then(() => {
        for (const f of files) localStorage.removeItem(legacyCaptionKey(f))
      })
      .catch(() => {
        /* keep the legacy copy if the write failed */
      })
  }

  return stored
}

/** Persist captions. Fire-and-forget: failure costs a re-generation, nothing more. */
function writeCaptions(entries: [CacheKey, string][]): void {
  if (entries.length === 0) return
  captionPutBatch(entries).catch(() => {
    console.warn('Caption store write failed')
  })
}

function resizeCanvas() {
  dom.canvas.width = window.innerWidth || 800
  dom.canvas.height = window.innerHeight || 600
}

function fitCamera(pts: Point[] = state.points) {
  if (!pts.length) return
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (const [x, y] of pts) {
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  const pad = THUMB_WORLD + 16
  const scX = (dom.canvas.width - pad * 2) / (maxX - minX || 1)
  const scY = (dom.canvas.height - pad * 2) / (maxY - minY || 1)
  camera.x = (minX + maxX) / 2
  camera.y = (minY + maxY) / 2
  camera.scale = Math.min(scX, scY)
}

function resetAll() {
  for (const bmp of state.thumbnails) {
    if (bmp?.close) bmp.close()
  }
  clearFullTiles(true)
  thumbDecoding.clear()
  thumbFailed.clear()
  embedFailed.clear()
  undecodable.clear()
  embedFailReason = ''
  updateFailedChip()
  thumbnailLRU.clear()
  for (const f of state.files) {
    if (f.objectURL) {
      URL.revokeObjectURL(f.objectURL)
      f.objectURL = null
    }
  }
  state.phase = 'idle'
  state.files = []
  state.vectors = []
  state.points = []
  layoutTween = null
  state.clusters = null
  state.thumbnails = []
  state.captions = []
  state.searchResults = null
  state.searchQuery = ''
  state.searchScores = null
  state.activeFileIndex = null
  state.lastViewedIndex = null
  localStorage.removeItem('po_fileKeys')
  localStorage.removeItem('po_umapPoints')
  localStorage.removeItem('po_projectedPoints')
  localStorage.removeItem('po_clusters')
  localStorage.removeItem('po_viewerMode')
  camera.x = 0
  camera.y = 0
  camera.scale = 1
  const ctx = dom.canvas.getContext('2d')
  if (ctx) ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height)
  dom.recenterBtn.disabled = true
  dom.resetBtn.disabled = true
  dom.headerRecenterBtn.disabled = true
  dom.searchInput.disabled = true
  dom.searchInput.value = ''
  dom.searchClearBtn.hidden = true
  dom.resumeBtn.hidden = true
  dom.resumeBtn.disabled = true
  dom.resumeBtn.classList.remove('primary')
  dom.openBtn.disabled = false
  dom.demoBtn.disabled = false
  setProgress(0)
  setStatus('Cleared. Open a folder to start.')
  refreshCacheSize()
}

function render() {
  const ctx = dom.canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, dom.canvas.width, dom.canvas.height)
  let pts = state.points
  if (layoutTween) {
    const t = (performance.now() - layoutTween.start) / LAYOUT_TWEEN_MS
    if (t >= 1) layoutTween = null
    else {
      const e = 1 - (1 - t) ** 3 // ease-out cubic
      const from = layoutTween.from
      pts = state.points.map((p, i): Point => {
        const f = from[i]
        return f ? [f[0] + (p[0] - f[0]) * e, f[1] + (p[1] - f[1]) * e] : p
      })
      // scheduleRender() is a no-op while this frame is still pending.
      requestAnimationFrame(scheduleRender)
    }
  }
  displayedPoints = pts
  const thumbs = state.thumbnails
  if (!pts.length) return

  const rank = new Int32Array(pts.length).fill(pts.length)
  if (state.searchResults) {
    for (let i = 0; i < state.searchResults.length; i++) {
      rank[state.searchResults[i]] = i
    }
  }

  const s = camera.scale
  const cxW = dom.canvas.width / 2
  const cyW = dom.canvas.height / 2
  const drawSize = THUMB_WORLD * s
  const half = drawSize / 2
  const useFull = drawSize > FULL_LOD_SIZE
  if (useFull) fullFrame++
  const fullMisses: [index: number, px: number][] = []

  // Find index closest to center for highlighting if nothing viewed yet
  let currentActive = state.lastViewedIndex
  if (currentActive === null && pts.length > 0) {
    let minD = Infinity
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i][0] - camera.x
      const dy = pts[i][1] - camera.y
      const d = dx * dx + dy * dy
      if (d < minD) {
        minD = d
        currentActive = i
      }
    }
  }

  // Frustum culling + draw prioritization (search results first, then center distance)
  const visibleIndices = cullAndPrioritize(
    pts,
    camera,
    dom.canvas.width,
    dom.canvas.height,
    half,
    state.searchResults ? rank : null,
    state.settings.drawBudget
  )

  // Matches inside the cutoff shrink from full size (best) to 35% (last kept);
  // everything past it is small and dim.
  // A person's photos are all matches; a text search keeps its top percent.
  const keep =
    activePerson !== null && state.searchResults
      ? state.searchResults.length
      : Math.max(1, Math.ceil((pts.length * state.settings.searchPercent) / 100))

  for (const i of visibleIndices) {
    if (isHidden(i)) continue
    const sx = (pts[i][0] - camera.x) * s + cxW
    const sy = (pts[i][1] - camera.y) * s + cyW

    let alphaMultiplier = 1.0
    let highlightBorder = null
    let isSelected = i === currentActive
    let imgW = 0 // drawn image box, for the type badge
    let imgH = 0
    let sizeMul = 1

    if (state.searchResults) {
      const r = rank[i]
      if (r < keep) {
        sizeMul = 1 - 0.65 * (r / keep)
        if (r < 20) highlightBorder = '#facc15' // Bright yellow for top results
      } else {
        sizeMul = 0.3
        alphaMultiplier = 0.1
      }
    }
    const ds = drawSize * sizeMul

    if (isSelected) {
      highlightBorder = '#fff'
      alphaMultiplier = 1.0
    }

    const clr =
      highlightBorder ??
      (state.clusters?.length
        ? CLUSTER_COLORS[state.clusters[i] % CLUSTER_COLORS.length]
        : '#6b7280')
    let drawn = false

    const file = state.files[i]
    // A video is never a tile source: createImageBitmap would read the whole file just to reject it.
    if (useFull && file && !isVideoName(file.name)) {
      const want = ds <= 512 ? 512 : 1024 // zoom stops at scale 20, a 960 px tile
      const tile = fullImages.get(i)
      if (tile) {
        tile.frame = fullFrame
        fullImages.delete(i)
        fullImages.set(i, tile)
      }
      if ((!tile || tile.px < want) && !fullFailed.has(i)) {
        const job = fullPending.get(i)
        if (job) job.frame = fullFrame
        else fullMisses.push([i, want])
      }

      if (tile) {
        const fullImg = tile.bmp
        const ratio = fullImg.width / fullImg.height
        const dw = ratio >= 1 ? ds : ds * ratio
        const dh = ratio >= 1 ? ds / ratio : ds
        ctx.globalAlpha = 0.9 * alphaMultiplier
        ctx.drawImage(fullImg, sx - dw / 2, sy - dh / 2, dw, dh)
        ctx.globalAlpha = 1.0
        ctx.strokeStyle = clr
        ctx.lineWidth = isSelected
          ? Math.max(3, 4 * Math.min(s, 1))
          : Math.max(1.5, 2 * Math.min(s, 1))
        ctx.strokeRect(sx - dw / 2, sy - dh / 2, dw, dh)
        imgW = dw
        imgH = dh
        drawn = true
      }
    }

    if (!drawn) {
      const thumb = thumbs[i]
      if (!thumb) lazyDecodeThumbnail(i)
      if (thumb && thumb.width > 0) {
        // Update LRU when thumbnail is actually used for rendering
        thumbnailLRU.delete(i)
        thumbnailLRU.add(i)

        const ratio = thumb.width / thumb.height
        const dw = ratio >= 1 ? ds : ds * ratio
        const dh = ratio >= 1 ? ds / ratio : ds
        ctx.globalAlpha = 0.9 * alphaMultiplier
        ctx.drawImage(thumb, sx - dw / 2, sy - dh / 2, dw, dh)
        ctx.globalAlpha = 1.0
        ctx.strokeStyle = clr
        ctx.lineWidth = isSelected
          ? Math.max(3, 4 * Math.min(s, 1))
          : Math.max(1.5, 2 * Math.min(s, 1))
        ctx.strokeRect(sx - dw / 2, sy - dh / 2, dw, dh)
        imgW = dw
        imgH = dh
      } else {
        const r = isSelected ? Math.max(5, half * 0.4) : Math.max(3, half * 0.3)
        ctx.beginPath()
        ctx.arc(sx, sy, r, 0, Math.PI * 2)
        ctx.fillStyle = clr
        ctx.globalAlpha = 0.8
        ctx.fill()
        ctx.globalAlpha = 1
        if (isSelected) {
          ctx.strokeStyle = '#fff'
          ctx.lineWidth = 2
          ctx.stroke()
        }
      }
    }

    if (Math.min(imgW, imgH) >= 20)
      drawTypeBadge(ctx, sx - imgW / 2, sy + imgH / 2, Math.min(imgW, imgH), isVideoName(state.files[i]?.name ?? ''))

    // When every file failed a badge on each says nothing; the chip and toast do.
    if (embedFailed.has(i) && embedFailed.size < pts.length && drawSize >= 12)
      drawFailedBadge(ctx, sx + half, sy - half, drawSize)
  }

  drawClusterLabels(ctx, pts, s, cxW, cyW, drawSize)
  drawFolderLabels(ctx, pts, s, cxW, cyW, drawSize)

  if (useFull) {
    let pending = 0
    for (const [i, job] of fullPending) {
      if (job.frame !== fullFrame) fullPending.delete(i)
      else pending += job.px * job.px * 4
    }
    for (const [i, px] of fullMisses) {
      const need = px * px * 4
      evictStaleFullTiles(FULL_TILE_BUDGET - pending - need)
      // The tiles on screen fill the budget: the rest stay thumbnails rather than evict each other.
      if (fullBytes + pending + need > FULL_TILE_BUDGET) break
      requestFullTile(i, px)
      pending += need
    }
    evictStaleFullTiles(FULL_TILE_BUDGET - pending)
  } else {
    clearFullTiles(false)
  }
}

// ── Cluster labels ───────────────────────────────────────────────────────────
// Recomputed only when the clusters or the number of captions change; counting
// captions each frame is a cheap O(n) scan next to drawing.
let clusterLabelCache: { clusters: Int32Array; captionCount: number; labels: Map<number, string> } | null =
  null
let clusterLabelHits: { x: number; y: number; w: number; h: number; cluster: number }[] = []
const LABEL_MAX_THUMB_PX = 64 // only label when zoomed out enough that thumbnails are small

function currentClusterLabels(): Map<number, string> | null {
  const clusters = state.clusters
  if (!clusters) return null
  let captionCount = 0
  for (const c of state.captions) if (c) captionCount++
  const cache = clusterLabelCache
  if (cache && cache.clusters === clusters && cache.captionCount === captionCount) return cache.labels
  const labels = clusterCaptionLabels(clusters, state.captions)
  clusterLabelCache = { clusters, captionCount, labels }
  return labels
}

function drawClusterLabels(
  ctx: CanvasRenderingContext2D,
  pts: Point[],
  s: number,
  cxW: number,
  cyW: number,
  drawSize: number
) {
  clusterLabelHits = []
  const labels = drawSize < LABEL_MAX_THUMB_PX ? currentClusterLabels() : null
  if (!labels?.size || !state.clusters) return
  const sum = new Map<number, [number, number, number]>()
  for (let i = 0; i < pts.length; i++) {
    const c = state.clusters[i]
    if (!labels.has(c) || isHidden(i)) continue
    const acc = sum.get(c) ?? [0, 0, 0]
    acc[0] += pts[i][0]
    acc[1] += pts[i][1]
    acc[2]++
    sum.set(c, acc)
  }
  ctx.font = '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  for (const [c, [x, y, n]] of sum) {
    const text = labels.get(c)!
    const sx = (x / n - camera.x) * s + cxW
    const sy = (y / n - camera.y) * s + cyW
    const w = ctx.measureText(text).width + 16
    const h = 24
    ctx.fillStyle = 'rgba(17, 24, 39, 0.78)'
    ctx.beginPath()
    ctx.roundRect(sx - w / 2, sy - h / 2, w, h, h / 2)
    ctx.fill()
    ctx.strokeStyle = CLUSTER_COLORS[c % CLUSTER_COLORS.length]
    ctx.lineWidth = 1.5
    ctx.stroke()
    ctx.fillStyle = '#fff'
    ctx.fillText(text, sx, sy + 0.5)
    clusterLabelHits.push({ x: sx - w / 2, y: sy - h / 2, w, h, cluster: c })
  }
}

// Viewer mode lays files out one block per folder (generateMetadataBasedLayout);
// name each block just above its top-left thumbnail.
let folderNameCache: { files: PhotoFile[]; names: string[] } | null = null
function folderNames(): string[] {
  if (folderNameCache?.files !== state.files) {
    const names = state.files.map((f) => f.name.split('/').slice(0, -1).join('/') || '(root)')
    folderNameCache = { files: state.files, names }
  }
  return folderNameCache.names
}

function drawFolderLabels(
  ctx: CanvasRenderingContext2D,
  pts: Point[],
  s: number,
  cxW: number,
  cyW: number,
  drawSize: number
) {
  // AI layouts group by meaning, not folder (state.clusters is set there).
  if (!state.settings.viewerOnly || state.clusters) return
  const names = folderNames()
  const topLeft = new Map<string, [number, number]>()
  for (let i = 0; i < pts.length; i++) {
    if (isHidden(i) || !names[i]) continue
    const acc = topLeft.get(names[i])
    if (!acc) topLeft.set(names[i], [pts[i][0], pts[i][1]])
    else {
      acc[0] = Math.min(acc[0], pts[i][0])
      acc[1] = Math.min(acc[1], pts[i][1])
    }
  }
  ctx.font = '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
  ctx.textAlign = 'left'
  ctx.textBaseline = 'middle'
  const h = 22
  for (const [name, [x, y]] of topLeft) {
    const sx = (x - camera.x) * s + cxW - drawSize / 2
    const sy = (y - camera.y) * s + cyW - drawSize / 2 - h / 2 - 4
    if (sy < -h || sy > dom.canvas.height + h) continue
    const w = ctx.measureText(name).width + 16
    if (sx > dom.canvas.width || sx + w < 0) continue
    ctx.fillStyle = 'rgba(17, 24, 39, 0.78)'
    ctx.beginPath()
    ctx.roundRect(sx, sy - h / 2, w, h, 6)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.fillText(name, sx + 8, sy + 0.5)
  }
}

/** Zoom the camera onto one cluster's photos. */
function zoomToCluster(cluster: number) {
  if (!state.clusters) return
  const members = state.points.filter((_, i) => state.clusters![i] === cluster && !isHidden(i))
  cameraFollow = false
  state.lastViewedIndex = null
  fitCamera(members)
  camera.scale = Math.min(camera.scale, 20) // same cap as wheel/pinch zoom
  scheduleRender()
}

/** Red "!" disc on a thumbnail's top-right corner: its embedding failed. */
/** Small, translucent corner mark: ▶ for a video, a picture frame for a photo. */
function drawTypeBadge(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, video: boolean) {
  const r = Math.min(8, Math.max(4, size * 0.09))
  const cx = x + r + 2
  const cy = y - r - 2
  ctx.globalAlpha = 0.55
  ctx.fillStyle = '#000'
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.fill()
  ctx.globalAlpha = 0.85
  ctx.fillStyle = '#fff'
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 1
  ctx.beginPath()
  if (video) {
    ctx.moveTo(cx - r * 0.3, cy - r * 0.45)
    ctx.lineTo(cx + r * 0.5, cy)
    ctx.lineTo(cx - r * 0.3, cy + r * 0.45)
    ctx.closePath()
    ctx.fill()
  } else {
    // a frame with a mountain line
    ctx.strokeRect(cx - r * 0.5, cy - r * 0.4, r, r * 0.8)
    ctx.moveTo(cx - r * 0.5, cy + r * 0.4)
    ctx.lineTo(cx - r * 0.1, cy)
    ctx.lineTo(cx + r * 0.5, cy + r * 0.4)
    ctx.closePath()
    ctx.fill()
  }
  ctx.globalAlpha = 1
}

function drawFailedBadge(ctx: CanvasRenderingContext2D, x: number, y: number, size: number) {
  const r = Math.min(10, Math.max(5, size * 0.12))
  ctx.beginPath()
  ctx.arc(x - r, y + r, r, 0, Math.PI * 2)
  ctx.fillStyle = '#dc2626'
  ctx.fill()
  ctx.fillStyle = '#fff'
  ctx.font = `bold ${Math.round(r * 1.5)}px sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('!', x - r, y + r + 0.5)
}

function updateFailedChip() {
  const n = embedFailed.size
  dom.failedChip.hidden = n === 0
  dom.failedChip.textContent = hideFailed ? `⚠ ${n} failed (hidden) · show` : `⚠ ${n} failed · hide`
  dom.failedChip.title = hideFailed
    ? 'Show the files whose embedding failed'
    : 'These files could not be embedded and sit at a meaningless spot on the map. Click to hide them.'
}

// ── Projection ───────────────────────────────────────────────────────────────
async function runProjection(
  vectors: Float32Array[],
  method: ProjectionMethod,
  nNeighbors: number,
  { silent = false } = {}
): Promise<number[][]> {
  // UMAP/ISOMAP/LLE need n_neighbors < N, so 1–2 files crashed (Bugsink: "n_neighbors (=2) … N=1")
  if (vectors.length < 3) return vectors.map((_, i) => [i, 0])
  try {
    if (!silent) setStatus(`Projecting with ${method}…`)
    // DruidJS runs in a worker (src/projectionWorker.ts); its transform() is one long synchronous call.
    return await projectInWorker(vectors, method, nNeighbors)
  } catch (err) {
    setStatus(`${method} projection failed: ${(err as Error).message}`)
    throw err
  }
}

/**
 * Ask before the first upload to a given host. Resolves false if declined.
 *
 * Consent is per host and remembered, so switching endpoints asks again —
 * agreeing to upload to your own LAN box is not agreement to upload to a
 * third-party API.
 */
function showRemoteConsentModal(baseUrl: string): Promise<boolean> {
  const host = openaiHost(baseUrl)
  if (hasOpenAIConsent(baseUrl)) return Promise.resolve(true)

  for (const el of document.querySelectorAll('#openai-consent-host, #openai-consent-host-2')) {
    el.textContent = host
  }

  return new Promise((resolve) => {
    const done = (accepted: boolean) => {
      dom.openaiConsentAccept.removeEventListener('click', onAccept)
      dom.openaiConsentCancel.removeEventListener('click', onCancel)
      dom.openaiConsentModal.removeEventListener('cancel', onCancel)
      dom.openaiConsentModal.close()
      if (accepted) recordOpenAIConsent(baseUrl)
      resolve(accepted)
    }
    const onAccept = () => done(true)
    const onCancel = () => done(false)

    dom.openaiConsentAccept.addEventListener('click', onAccept)
    dom.openaiConsentCancel.addEventListener('click', onCancel)
    // Esc and the backdrop both count as declining.
    dom.openaiConsentModal.addEventListener('cancel', onCancel)
    dom.openaiConsentModal.showModal()
  })
}

// ── Model singleton ──────────────────────────────────────────────────────────
async function loadModelOnce(signal?: AbortSignal) {
  applyModelEnv()
  activeEmbeddingDim = 768
  state.phase = 'loading_model'
  setStatus('Loading model…')
  setProgress(0)

  if (state.settings.modelVariant === 'openai') {
    const baseUrl = normalizeBaseUrl(state.settings.openai.baseUrl)
    const model = state.settings.openai.model.trim()

    const bailOut = (message: string) => {
      offerRetry(message)
      state.phase = 'idle'
    }

    if (!baseUrl || !model) {
      bailOut('Set a remote endpoint URL and model in Settings first.')
      return
    }

    if (!(await showRemoteConsentModal(baseUrl))) {
      bailOut('Upload declined — pick a local model in Settings instead.')
      return
    }

    const cfg: OpenAICompatConfig = {
      baseUrl,
      apiKey: getOpenAIKey(),
      model,
      batchSize: Math.max(1, Math.min(state.settings.batchSize, 16))
    }

    setStatus(`Contacting ${openaiHost(baseUrl)}…`)
    // Probing doubles as the connection check and as the source of the vector
    // width, which must be known before any zero vector is built.
    activeEmbeddingDim = await probeDimension(cfg, signal)
    remoteConfig = cfg

    // Mirror the ready-state handover the other backends do. Without it the
    // phase stays 'loading_model' and Open/Demo/Resume stay disabled from page
    // load, so a successful connection would dead-end at "Connected…".
    updateDeviceBadge()
    setProgress(100)
    state.phase = 'model_ready'
    setProgress(0)
    dom.loadModelBtn.hidden = true
    dom.modelSelect.disabled = false
    dom.openBtn.disabled = false
    dom.openBtn.classList.add('primary')
    dom.demoBtn.disabled = false

    const where = `${openaiHost(baseUrl)} (${activeEmbeddingDim}-d)`
    if (!dom.resumeBtn.hidden) {
      dom.resumeBtn.disabled = false
      dom.resumeBtn.classList.add('primary')
      setStatus(`Connected to ${where} — resume or open a folder.`)
    } else {
      setStatus(`Connected to ${where} — open a folder to start.`)
    }
    return
  }

  // navigator.gpu can exist with no usable adapter (emulators, blocklisted GPUs).
  if (getVlmTier(state.settings.modelVariant)?.family === 'embeddinggemma2' && !USE_WEBNN) {
    const adapter = await navigator.gpu?.requestAdapter().catch(() => null)
    const saved = state.settings.modelVariant
    const slow = isPowerVR(adapter)
    if (!adapter || slow) {
      if (savedSettings) modelStandIn = { saved, used: NO_WEBGPU_MODEL }
      state.settings.modelVariant = NO_WEBGPU_MODEL
      dom.modelSelect.value = NO_WEBGPU_MODEL
    }
    if (slow) {
      // Never point these users at the WebGPU flag: it is what makes it hang.
      if (savedSettings) showToast(`${saved} is too slow on this GPU — using ${NO_WEBGPU_MODEL}. Turn on chrome://flags "WebNN API" to run ${saved} here.`, 'warn', 10000)
    } else if (!adapter || adapter.info?.isFallbackAdapter) {
      const shown = showWebGpuHelp(!adapter ? 'blocked' : 'software', saved)
      if (!shown && !adapter && savedSettings) {
        showToast(`${saved} needs WebGPU or WebNN, which this device lacks — using ${NO_WEBGPU_MODEL} this session.`, 'warn', 8000)
      }
    }
  }

  if (isVlmVariant(state.settings.modelVariant)) {
    const tier = getVlmTier(state.settings.modelVariant)!
    setStatus(`Loading ${tier.id} (${tier.downloadMB} MB)…`)
    try {
      activeEmbeddingDim = tier.visionDim
      if (tier.backend === 'webnn') {
        try {
          localStorage.setItem(WEBNN_LOADING_KEY, __GIT_COMMIT__)
        } catch {
          /* storage full: the crash guard is lost, the load is not */
        }
      }
      await loadVlm(state.settings.modelVariant, normalizeHost(state.settings.customModelHost), (pct) => {
        setProgress(pct)
        setStatus(`Downloading ${tier.id}… ${pct.toFixed(0)}%`)
      })
    } catch (err) {
      signal?.throwIfAborted()
      throw err
    } finally {
      localStorage.removeItem(WEBNN_LOADING_KEY)
    }
    vlmTier = tier
    updateDeviceBadge()
    setProgress(100)
    state.phase = 'model_ready'
    setProgress(0)
    dom.loadModelBtn.hidden = true
    dom.modelSelect.disabled = false
    dom.openBtn.disabled = false
    dom.openBtn.classList.add('primary')
    dom.demoBtn.disabled = false
    if (!dom.resumeBtn.hidden) {
      dom.resumeBtn.disabled = false
      dom.resumeBtn.classList.add('primary')
      setStatus(`${tier.id} ready — resume or open a folder.`)
    } else {
      setStatus(`${tier.id} ready — open a folder to start.`)
    }
    return
  }

  if (state.settings.modelVariant === 'chrome-ai') {
    setStatus('Checking Chrome AI availability…')
    const availability = await getChromeAIAvailability()

    if (availability === 'unavailable') {
      setStatus(
        'Chrome AI unavailable. Enable chrome://flags/#prompt-api-for-gemini-nano in Chrome 138+.'
      )
      dom.loadModelBtn.hidden = false
      state.phase = 'idle'
      return
    }

    // Chrome starts the download only from a user gesture (the Load Model
    // click, see downloadChromeAI); the auto-load at startup has none.
    if (availability === 'downloadable' && !chromeAIDownload) {
      setStatus('Gemini Nano needs a one-time download, managed by Chrome. Tap "Load Model" to start it.')
      dom.loadModelBtn.hidden = false
      dom.loadModelBtn.disabled = false
      state.phase = 'idle'
      return
    }

    if (chromeAIDownload && availability !== 'available') {
      try {
        await chromeAIDownload
      } catch (err) {
        chromeAIDownload = null
        setStatus(`Gemini Nano download failed: ${(err as Error).message}. Tap "Load Model" to retry.`)
        dom.loadModelBtn.hidden = false
        dom.loadModelBtn.disabled = false
        state.phase = 'idle'
        return
      }
    } else if (availability === 'downloadable' || availability === 'downloading') {
      setStatus('Chrome AI model downloading (managed by Chrome, this happens once)…')
      let polls = 0
      while (polls < 300) {
        signal?.throwIfAborted()
        await new Promise((r) => setTimeout(r, 2000))
        signal?.throwIfAborted()
        const current = await getChromeAIAvailability()
        if (current === 'available') break
        setStatus(`Chrome AI model downloading… (${++polls * 2}s elapsed)`)
      }
      if ((await getChromeAIAvailability()) !== 'available') {
        setStatus('Chrome AI model download timed out. Try again later.')
        dom.loadModelBtn.hidden = false
        state.phase = 'idle'
        return
      }
    }

    setStatus('Initializing Chrome AI session…')
    chromeAIManager = new ChromeAISessionManager()

    if (!textExtractor) {
      setStatus('Loading text embedding model (134 MB)…')
      const TEXT_MODEL_SIZE_BYTES = 134 * 1024 * 1024
      const textLoaded = new Map<string, number>()

      const textProgressCb = (e: ProgressEvent) => {
        if (e.status === 'progress') {
          textLoaded.set(e.file, e.loaded ?? 0)
          const total = [...textLoaded.values()].reduce((a, b) => a + b, 0)
          const pct = Math.min(99, (total / TEXT_MODEL_SIZE_BYTES) * 100)
          setProgress(pct)
          setStatus(`Loading text model… ${pct.toFixed(0)}%`)
        }
      }

      const tryLoadText = (device: 'webgpu' | 'wasm') =>
        (pipeline as Pipeline)('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5', {
          device,
          dtype: 'fp32',
          progress_callback: textProgressCb
        }) as Promise<PipelineInstance>

      try {
        textExtractor = await tryLoadText(await webgpuOrWasm())
      } catch {
        textLoaded.clear()
        textExtractor = await tryLoadText('wasm')
      }
    }

    updateDeviceBadge()
    setProgress(100)
    state.phase = 'model_ready'
    setProgress(0)
    dom.loadModelBtn.hidden = true
    dom.modelSelect.disabled = false
    dom.openBtn.disabled = false
    dom.openBtn.classList.add('primary')
    dom.demoBtn.disabled = false
    if (!dom.resumeBtn.hidden) {
      dom.resumeBtn.disabled = false
      dom.resumeBtn.classList.add('primary')
      setStatus('Chrome AI ready — resume or open a folder.')
    } else {
      setStatus('Chrome AI ready — open a folder to start.')
    }
    return
  }

  if (state.settings.modelVariant.startsWith('sapiens2')) {
    const sapiens2Variant = state.settings.modelVariant.split('-')[1] as Sapiens2Variant
    let result: Awaited<ReturnType<typeof loadSapiens2>>
    try {
      result = await loadSapiens2(
        sapiens2Variant,
        (pct, fromCache) => {
          setProgress(pct)
          setStatus(
            fromCache
              ? `Loading Sapiens2 (${sapiens2Variant}) from cache…`
              : `Downloading Sapiens2 (${sapiens2Variant})… ${pct.toFixed(0)}%`
          )
        },
        signal,
        {
          host: state.settings.customModelHost,
          uploadedBuffer: pendingSapiens2Buffer ?? undefined
        }
      )
    } finally {
      // Always release the uploaded buffer — even if loading threw — so it
      // isn't held in memory or silently reused on the next attempt.
      pendingSapiens2Buffer = null
    }
    sapiens2Session = result.session
    modelDevice = result.device === 'webgpu' ? 'webgpu' : 'cpu'
    modelFallbackReason = result.fallbackReason
    updateDeviceBadge()
    setProgress(100)
    state.phase = 'model_ready'
    setProgress(0)
    dom.loadModelBtn.hidden = true
    dom.modelSelect.disabled = false
    dom.openBtn.disabled = false
    dom.openBtn.classList.add('primary')
    dom.demoBtn.disabled = false
    if (!dom.resumeBtn.hidden) {
      dom.resumeBtn.disabled = false
      dom.resumeBtn.classList.add('primary')
      setStatus('Sapiens2 ready — resume or open a folder.')
    } else {
      setStatus('Sapiens2 ready — open a folder to start.')
    }
    return
  }

  const MODEL_SIZE_BYTES = 380 * 1024 * 1024 + 134 * 1024 * 1024
  const loaded = new Map<string, number>()

  const progressCb = (e: ProgressEvent) => {
    if (e.status === 'progress') {
      loaded.set(e.file, e.loaded ?? 0)
      const total = [...loaded.values()].reduce((a, b) => a + b, 0)
      const pct = Math.min(99, (total / MODEL_SIZE_BYTES) * 100)
      setProgress(pct)
      setStatus(`Loading model… ${pct.toFixed(0)}%`)
    }
  }

  const tryLoad = (device: 'webgpu' | 'wasm') =>
    (pipeline as Pipeline)('image-feature-extraction', 'nomic-ai/nomic-embed-vision-v1.5', {
      device,
      dtype: 'fp32',
      progress_callback: progressCb,
      pooling: 'mean',
      normalize: true
    }) as Promise<PipelineInstance>

  try {
    const device = await webgpuOrWasm()
    extractor = await tryLoad(device)
    modelDevice = device === 'webgpu' ? 'webgpu' : 'cpu'
  } catch (gpuErr) {
    console.warn('WebGPU init failed, falling back to wasm:', gpuErr)
    setStatus('WebGPU unavailable — using CPU (slower)…')
    loaded.clear()
    extractor = await tryLoad('wasm')
    modelDevice = 'cpu'
  }
  updateDeviceBadge()
  setProgress(100)

  if (state.settings.enableTextSearch) {
    setStatus('Loading text model for search…')
    const TEXT_MODEL_SIZE_BYTES = 134 * 1024 * 1024
    const textLoaded = new Map<string, number>()

    const textProgressCb = (e: ProgressEvent) => {
      if (e.status === 'progress') {
        textLoaded.set(e.file, e.loaded ?? 0)
        const total = [...textLoaded.values()].reduce((a, b) => a + b, 0)
        const pct = Math.min(99, (total / TEXT_MODEL_SIZE_BYTES) * 100)
        setProgress(pct)
        setStatus(`Loading text model… ${pct.toFixed(0)}%`)
      }
    }

    const tryLoadText = (device: 'webgpu' | 'wasm') =>
      (pipeline as Pipeline)('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5', {
        device,
        dtype: 'fp32',
        progress_callback: textProgressCb
      }) as Promise<PipelineInstance>

    try {
      textExtractor = await tryLoadText(await webgpuOrWasm())
    } catch (gpuErr) {
      console.warn('Text model WebGPU failed, using wasm:', gpuErr)
      textLoaded.clear()
      textExtractor = await tryLoadText('wasm')
    }
    setProgress(100)
  }

  state.phase = 'model_ready'
  setProgress(0)
  dom.loadModelBtn.hidden = true
  dom.modelSelect.disabled = true
  dom.openBtn.disabled = false
  dom.openBtn.classList.add('primary')
  dom.demoBtn.disabled = false
  if (!dom.resumeBtn.hidden) {
    dom.resumeBtn.disabled = false
    dom.resumeBtn.classList.add('primary')
    setStatus('Model ready — resume or open a folder.')
  } else {
    setStatus('Model ready — open a folder to start.')
  }
}

// ── Model download fallback (offline / corporate proxy) ──────────────────────
interface FallbackChoice {
  host?: string // alternative HuggingFace-compatible host (may be '')
  sapiensBuffer?: ArrayBuffer // uploaded single .onnx (Sapiens2)
  uploadFiles?: Map<string, File> // uploaded model folder (Transformers.js)
}

// Show the recovery modal after a download failure. Resolves with the user's
// choice, or null if they cancelled.
function showModelFallbackModal(variant: string): Promise<FallbackChoice | null> {
  const isSapiens = variant.startsWith('sapiens2')

  // Visible, copy/curl-friendly direct links (always against the canonical Hub).
  const urls = modelDownloadUrls(variant, {
    includeText: variant === 'nomic' && state.settings.enableTextSearch
  })
  dom.modelFallbackUrls.innerHTML = urls
    .map((u) => `<li>🔗 <a href="${u}" target="_blank" rel="noopener">${u}</a></li>`)
    .join('')

  // Single .onnx for Sapiens2; whole model folder for Transformers.js repos.
  if (isSapiens) {
    dom.modelFallbackFile.accept = '.onnx'
    dom.modelFallbackFile.removeAttribute('webkitdirectory')
    dom.modelFallbackFile.multiple = false
    dom.modelFallbackFileHint.textContent = 'Select the single .onnx file listed above.'
  } else {
    dom.modelFallbackFile.removeAttribute('accept')
    dom.modelFallbackFile.setAttribute('webkitdirectory', '')
    dom.modelFallbackFile.multiple = true
    dom.modelFallbackFileHint.textContent =
      'Select the downloaded model folder (must contain config.json and the onnx/ folder).'
  }
  dom.modelFallbackFile.value = ''
  dom.modelFallbackHost.value = state.settings.customModelHost || ''
  dom.modelFallbackModal.showModal()

  return new Promise((resolve) => {
    const finish = (choice: FallbackChoice | null) => {
      dom.modelFallbackModal.close()
      dom.modelFallbackRetry.onclick = null
      dom.modelFallbackCancel.onclick = null
      dom.modelFallbackClose.onclick = null
      dom.modelFallbackModal.oncancel = null
      dom.modelFallbackRetry.disabled = false
      dom.modelFallbackCancel.disabled = false
      resolve(choice)
    }

    dom.modelFallbackRetry.onclick = async () => {
      // Reading uploaded files is async — disable both buttons so a double-click
      // can't trigger redundant file reads or a second finish() call.
      dom.modelFallbackRetry.disabled = true
      dom.modelFallbackCancel.disabled = true
      try {
        const choice: FallbackChoice = {
          host: dom.modelFallbackHost.value.trim()
        }
        const files = dom.modelFallbackFile.files
        if (files && files.length) {
          if (isSapiens) {
            choice.sapiensBuffer = await files[0].arrayBuffer()
          } else {
            const map = new Map<string, File>()
            for (let i = 0; i < files.length; i++) {
              map.set(files[i].webkitRelativePath || files[i].name, files[i])
            }
            choice.uploadFiles = map
          }
        }
        finish(choice)
      } catch (err) {
        // File read failed — re-enable so the user can try again.
        dom.modelFallbackRetry.disabled = false
        dom.modelFallbackCancel.disabled = false
        showToast(`Couldn't read the uploaded file: ${(err as Error).message}`, 'error')
      }
    }
    dom.modelFallbackCancel.onclick = () => finish(null)
    dom.modelFallbackClose.onclick = () => finish(null)
    dom.modelFallbackModal.oncancel = () => finish(null) // Esc key
  })
}

function applyFallbackChoice(choice: FallbackChoice) {
  if (choice.host !== undefined) {
    state.settings.customModelHost = normalizeHost(choice.host)
    saveSettings()
    applyModelEnv()
  }
  if (choice.sapiensBuffer) pendingSapiens2Buffer = choice.sapiensBuffer
  if (choice.uploadFiles) {
    // Feed the uploaded files to Transformers.js via a custom Web Cache.
    ;(env as unknown as { useCustomCache: boolean }).useCustomCache = true
    ;(env as unknown as { customCache: unknown }).customCache = buildUploadCache(choice.uploadFiles)
  }
}

// Load the model, retrying through the fallback modal on download failures.
// Only Chrome AI keeps the button: Chrome starts the Gemini Nano download from
// a click. Every other model loads at startup, so a failure is retried by reload.
function offerRetry(message: string) {
  const chrome = state.settings.modelVariant === 'chrome-ai'
  setStatus(chrome ? message : `${message} Reload the page to retry.`)
  dom.loadModelBtn.hidden = !chrome
  dom.loadModelBtn.disabled = false
}

async function loadModel(signal?: AbortSignal) {
  if (extractor || sapiens2Session || chromeAIManager || remoteConfig || vlmTier) return
  // Clear any custom upload cache from a previous run — env is a global
  // singleton, so a stale cache would otherwise keep serving old files when the
  // user switches models or starts a new session. A fresh upload re-sets it
  // inside the retry loop below via applyFallbackChoice().
  ;(env as unknown as { useCustomCache: boolean }).useCustomCache = false
  ;(env as unknown as { customCache: unknown }).customCache = null
  while (true) {
    try {
      await loadModelOnce(signal)
      return
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') throw err
      // `isDownloadError` matches on message text ('unauthorized', '403', …),
      // which describes a failed *model download*. A remote endpoint rejecting
      // a key produces the same words, and offering to upload .onnx files in
      // response would be nonsense — this mode downloads no model at all.
      // A worker frame has nobody to answer the fallback dialog; its parent retries or gives up.
      if (OVERNIGHT_CHILD) throw err
      if (state.settings.modelVariant !== 'openai' && isDownloadError(err)) {
        const choice = await showModelFallbackModal(state.settings.modelVariant)
        if (choice) {
          applyFallbackChoice(choice)
          continue // retry with the new host / uploaded files
        }
      }
      // describeOpenAIError adds the actionable hint ("key rejected", "no such
      // model") and is already redacted; the raw message may hold the key.
      offerRetry(
        state.settings.modelVariant === 'openai'
          ? `Remote endpoint failed: ${describeOpenAIError(err)}`
          : `Failed to load model: ${(err as Error).message}`
      )
      throw err
    }
  }
}

// ── Embedding helpers ─────────────────────────────────────────────────────────

// Resize a File to 256px wide before GPU upload. Returns null if the browser
// cannot decode/resize it (caller should skip inference, not fall back to
// full-res, to avoid the OOM this fix is meant to prevent).
async function resizeForEmbedding(file: File): Promise<RawImage | null> {
  try {
    const bmp = await createImageBitmap(file, {
      resizeWidth: 256,
      resizeQuality: 'medium'
    })
    try {
      const cvs = document.createElement('canvas')
      cvs.width = bmp.width
      cvs.height = bmp.height
      const ctx2d = cvs.getContext('2d')
      if (!ctx2d) return null
      ctx2d.drawImage(bmp, 0, 0)
      return await RawImage.fromCanvas(cvs)
    } finally {
      bmp.close()
    }
  } catch {
    return null
  }
}

// Batch-read cached embeddings using the folder-independent key (basename based),
// falling back to the legacy full-path key for entries written by older builds.
// Returns the vectors aligned to `files`, the new keys (for writing fresh misses),
// and any [newKey, vector] pairs that should be migrated to the new key format so
// future parent/child opens hit directly.
async function readCachedEmbeddings(
  files: PhotoFile[],
  cachePrefix: string,
  /** Namespace of quick video vectors, used for a video with no vector under `cachePrefix`. */
  draftPrefix: string | null = null
): Promise<{
  keys: CacheKey[]
  cached: (Float32Array | null)[]
  migrate: [CacheKey, Float32Array][]
}> {
  const keys = files.map((f) => `${cachePrefix}${makeCacheKey(f)}` as CacheKey)

  // Query new and legacy keys in a single IDB transaction. Legacy keys only
  // differ for files inside a subfolder (basename !== full path), so we only
  // append those — keeping the extra reads minimal while avoiding a second
  // transaction per batch (which matters when batchSize is 1, e.g. Sapiens2).
  const queryKeys = [...keys]
  const legacyForFile: number[] = [] // queryKeys index -> file index
  files.forEach((f, i) => {
    if (f.name.includes('/')) {
      legacyForFile[queryKeys.length] = i
      queryKeys.push(`${cachePrefix}${f.name}:${f.size}:${f.lastModified}` as CacheKey)
    }
  })

  const draftForFile: number[] = [] // queryKeys index -> file index
  if (draftPrefix) {
    files.forEach((f, i) => {
      if (!isVideoName(f.name)) return
      draftForFile[queryKeys.length] = i
      queryKeys.push(`${draftPrefix}${makeCacheKey(f)}` as CacheKey)
    })
  }

  const results = await cacheGetBatch(queryKeys)
  const cached = results.slice(0, files.length)
  const migrate: [CacheKey, Float32Array][] = []

  // Legacy keys come before draft ones in queryKeys, so an old full vector beats a quick one.
  for (let q = files.length; q < results.length; q++) {
    const v = results[q]
    const i = legacyForFile[q] ?? draftForFile[q]
    if (i === undefined || cached[i] || !v) continue // new key already hit
    cached[i] = v
    // Legacy vectors move to the new key; a quick vector must never be written as a full one.
    if (legacyForFile[q] !== undefined) migrate.push([keys[i], v])
  }
  return { keys, cached, migrate }
}

// ── Embedding loop ───────────────────────────────────────────────────────────
/** Scale and shift `pts` to fill the bounding box of `ref`. */
function fitInto(pts: Point[], ref: readonly Point[]): Point[] {
  // Loops, not Math.min(...xs): spreading overflows the stack on huge folders.
  const box = (ps: readonly Point[]) => {
    let x0 = Infinity
    let x1 = -Infinity
    let y0 = Infinity
    let y1 = -Infinity
    for (const [x, y] of ps) {
      x0 = Math.min(x0, x)
      x1 = Math.max(x1, x)
      y0 = Math.min(y0, y)
      y1 = Math.max(y1, y)
    }
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0 || 1, h: y1 - y0 || 1 }
  }
  const a = box(pts)
  const b = box(ref)
  const k = Math.min(b.w / a.w, b.h / a.h)
  return pts.map((p): Point => [b.cx + (p[0] - a.cx) * k, b.cy + (p[1] - a.cy) * k])
}

type EmbedStop = 'ok' | 'gpu-lost' | 'low-memory'
interface EmbedOptions {
  /** Overnight worker frame: no map, no projection, nothing kept for display. */
  headless?: boolean
  onProgress?: (done: number, total: number) => void
  /** Stop early once less than this share of the JS heap is free (default 0.2). */
  minFreeRatio?: number
  /** Overnight quick pass: videos from DRAFT_VIDEO_FRAMES frames, written under videoDraftPrefix. */
  draft?: boolean
}
let lastEmbedStop: EmbedStop = 'ok'

// Milliseconds the worker took per still photo this run, for recordAppBench.
let appStillMs: number[] = []

/**
 * A real run's photo speed, remembered next to the benchmark page's (src/benchStore.ts) so the accelerator
 * choice uses what the app actually runs. The first photo pays warm-up and is left out.
 */
function recordAppBench() {
  const ms = appStillMs.slice(1)
  appStillMs = []
  if (ms.length < 3 || !vlmTier) return
  recordBench({ at: Date.now(), source: 'app', device: vlmDevice(), dtype: vlmTier.dtype, embedMs: median(ms), n: ms.length })
}

async function embedAll(files: PhotoFile[], opts: EmbedOptions = {}) {
  // Models and embeddings are best-effort storage until this is granted. Not awaited: it may prompt.
  void navigator.storage?.persist?.().catch(() => false)
  appStillMs = []
  try {
    // A headless (overnight) frame's parent already holds the wake lock.
    if (opts.headless) return await embedAllInner(files, opts)
    awakeWanted++
    void keepAwake()
    try {
      return await embedAllInner(files, opts)
    } finally {
      if (--awakeWanted === 0) void overnightWakeLock?.release().catch(() => {})
    }
  } finally {
    recordAppBench()
  }
}

async function embedAllInner(files: PhotoFile[], opts: EmbedOptions = {}) {
  const headless = opts.headless ?? false
  const minFreeRatio = opts.minFreeRatio ?? 0.2
  lastEmbedStop = 'ok'
  state.phase = 'embedding'
  const vectors = new Array<Float32Array>(files.length)
  let cacheHits = 0
  const writeQueue: [CacheKey, Float32Array][] = []
  cameraFollow = true
  const embedStart = performance.now()
  // Photos not embedded yet stay in their grid slot; the embedded ones glide
  // to the selected projection (fitted into the grid's bounds) every few seconds.
  const grid = state.points
  let projecting = false
  let lastProjection = 0
  // Grows with the projection's own cost, so a big folder isn't re-projected back to back.
  let projectionEvery = 2500
  // Batch-of-one models would otherwise pay one IndexedDB transaction per file.
  let lastFlush = performance.now()

  const isSapiens2 = state.settings.modelVariant.startsWith('sapiens2')
  const isChromeAI = state.settings.modelVariant === 'chrome-ai'
  const isRemote = state.settings.modelVariant === 'openai'
  const isVlm = isVlmVariant(state.settings.modelVariant)
  const reportedRemoteErrors = new Set<string>()
  let gpuLost = false
  // Sapiens2 runs one image at a time; chrome-ai uses 2 for potential parallel
  // session speedup. The transformers path adapts: the working batch size
  // halves after a GPU failure and creeps back up after sustained successes.
  const batcher = createAdaptiveBatcher(state.settings.batchSize)
  // Separate cache namespace per variant so vectors don't collide across models
  // (AGENT.md, "Embedding space"). This used to be a second copy of the same
  // ternary chain that currentCachePrefix() already implements; the two are now
  // one function, so the settings panel's cache count and the keys actually
  // written cannot disagree.
  const variantDraftPrefix = videoDraftPrefix(state.settings.modelVariant)
  const draft = !!opts.draft && variantDraftPrefix !== null
  const cachePrefix = draft ? variantDraftPrefix : currentCachePrefix()
  // A video that only has its quick vector counts as embedded, except in a worker frame, whose whole job is
  // to write the vector the quick one stands in for.
  const draftFallback = headless || draft ? null : variantDraftPrefix

  for (let i = 0; i < files.length; ) {
    // Remote batches are bounded by request size, not GPU memory, so the
    // adaptive batcher does not apply.
    // The VLM worker runs images one at a time internally (no VRAM headroom on
    // the target device), so a larger batch here would only delay progress
    // reporting without winning any throughput.
    const batchSize = isRemote ? 8 : isChromeAI ? 2 : isSapiens2 || isVlm ? 1 : batcher.size
    const batch = files.slice(i, Math.min(i + batchSize, files.length))

    // One IDB transaction for the whole batch, with legacy-key fallback so
    // embeddings cached by older builds (full-path keys) are reused and migrated.
    const { keys, cached, migrate } = await readCachedEmbeddings(batch, cachePrefix, draftFallback)
    if (migrate.length > 0) writeQueue.push(...migrate)

    // Width guard: a cached vector of the wrong width is from a different
    // embedding space, so treat it as a miss rather than mixing spaces. This
    // is what lets the remote cache namespace omit the dimension — it catches
    // a provider re-pointing a model id at a different width, which a
    // namespace built from host+model alone would not.
    for (let bi = 0; bi < cached.length; bi++) {
      if (cached[bi] && cached[bi]!.length !== activeEmbeddingDim) cached[bi] = null
    }

    // Resolve inputs for cache misses
    const missIndices: number[] = []
    // ImageBitmap[] entries are VLM frame groups: several frames for a video,
    // one for a still, pooled to a single vector by the worker.
    const missInputs: (File | RawImage | ImageBitmap | ImageBitmap[] | string)[] = []
    // Headless: video frames that would have been display thumbnails, closed after this batch.
    const tempBitmaps: ImageBitmap[] = []

    await Promise.all(
      batch.map(async (f, bi) => {
        if (cached[bi]) return // cache hit — handled below
        const idx = i + bi
        const ext = f.name.split('.').pop()?.toLowerCase() ?? ''
        if (VIDEO_EXTS.has(ext)) {
          // Use cached thumbnail if available, otherwise extract frame now
          let thumb: ImageBitmap | null = state.thumbnails[idx] ?? null
          if (!thumb && !thumbFailed.has(idx)) {
            thumb = await videoFrameLimit(() => extractVideoFrame(f.file))
            if (!thumb) thumbFailed.add(idx)
          }
          if (thumb) {
            if (headless) tempBitmaps.push(thumb)
            else if (!state.thumbnails[idx]) {
              state.thumbnails[idx] = thumb
              thumbDecoding.delete(idx)
            }
            if (isRemote) {
              // imageToDataURL only closes bitmaps it created itself, so the
              // one cached in state.thumbnails stays valid.
              missInputs.push(await imageToDataURL(thumb, state.settings.openai.maxImageWidth))
            } else if (isSapiens2) {
              missInputs.push(thumb)
            } else if (isVlm) {
              // Sample the video properly rather than upscaling the 224 px
              // thumbnail: decode framesPerVideo frames at VLM resolution so a
              // clip is represented by its content over time, not by whichever
              // single frame the thumbnail happened to catch.
              const tier = getVlmTier(state.settings.modelVariant)!
              const frames = await videoFrameLimit(() =>
                extractVideoFrames(f.file, draft ? DRAFT_VIDEO_FRAMES : tier.framesPerVideo, VLM_FRAME_PX)
              )
              if (frames.length > 0) {
                missInputs.push(frames)
              } else {
                // Decoded for the thumbnail but not for us; fall back to a
                // clone of it rather than dropping the file entirely.
                missInputs.push([await createImageBitmap(thumb, { resizeWidth: VLM_FRAME_PX })])
              }
            } else if (isChromeAI) {
              // Clone so Chrome AI doesn't close the bitmap that's cached in state.thumbnails
              missInputs.push(await createImageBitmap(thumb))
            } else {
              const canvas = document.createElement('canvas')
              canvas.width = thumb.width
              canvas.height = thumb.height
              const ctx = canvas.getContext('2d')
              if (ctx) {
                ctx.drawImage(thumb, 0, 0)
                missInputs.push(await RawImage.fromCanvas(canvas))
              } else {
                missInputs.push(f.file)
              }
            }
          } else {
            // No frame to send, so this video fails here. Handing the model the File instead (as the
            // non-VLM models once got) takes the whole batch, and every valid item in it, down with
            // one bad video: remote JSON.stringifies it to {}, the worker wants decoded frames, and
            // nomic's batch failed and halved down to 1 per such video, leaving its batch size at 1.
            vectors[idx] = failedVector(idx, `no frame could be decoded from a .${ext} video`)
            return
          }
        } else if (isRemote) {
          try {
            missInputs.push(await imageToDataURL(f.file, state.settings.openai.maxImageWidth))
          } catch {
            // Undecodable image — skip it rather than uploading the original
            // at full size, which is what this downscale exists to prevent.
            vectors[idx] = failedVector(idx, `could not decode a .${ext} image`)
            return
          }
        } else if (isSapiens2) {
          // Pass File directly; sapiens2 embedder resizes to 1024×768 internally
          missInputs.push(f.file)
        } else if (isVlm) {
          try {
            // A group of one: a still image is a one-frame clip.
            missInputs.push([await createImageBitmap(f.file, { resizeWidth: VLM_FRAME_PX })])
          } catch {
            // Undecodable image. A zero vector keeps this session going but is
            // never cached (see failedInputs below), so a later build can retry.
            vectors[idx] = failedVector(idx, `could not decode a .${ext} image`)
            return
          }
        } else if (isChromeAI) {
          // Use cached thumbnail if already decoded, else create a small bitmap.
          // Always create a fresh bitmap for the inference queue — never share the
          // state.thumbnails reference, because Chrome AI closes the bitmap after use.
          const cached = state.thumbnails[idx]
          try {
            missInputs.push(
              cached
                ? await createImageBitmap(cached) // clone so the canvas copy stays intact
                : await createImageBitmap(f.file, {
                    resizeWidth: 128,
                    resizeQuality: 'medium'
                  })
            )
          } catch {
            missInputs.push(f.file)
            missIndices.push(bi)
            return
          }
        } else {
          const resized = await resizeForEmbedding(f.file)
          if (resized !== null) {
            missInputs.push(resized)
          } else {
            // Cannot resize — use zero vector rather than risking OOM with full-res
            vectors[i + bi] = failedVector(i + bi, `could not decode a .${ext} image`)
            return
          }
        }
        missIndices.push(bi)
      })
    )

    // Apply cache hits
    const hitIndices: number[] = []
    for (let bi = 0; bi < batch.length; bi++) {
      if (cached[bi]) {
        vectors[i + bi] = cached[bi]!
        cacheHits++
        if (isChromeAI) hitIndices.push(bi)
      }
    }
    if (hitIndices.length > 0) {
      // One batched read for the whole batch instead of a synchronous
      // localStorage hit per file.
      const restored = await readCaptions(hitIndices.map((bi) => batch[bi]))
      for (let h = 0; h < hitIndices.length; h++) {
        state.captions[i + hitIndices[h]] = restored[h]
      }
    }

    // Run inference for cache misses
    if (missInputs.length > 0) {
      // Inputs that failed even alone get a zero vector in RAM for this
      // session but must NOT be written to the cache — a cached zero would
      // permanently poison that file's embedding.
      const failedInputs = new Set<File | RawImage | ImageBitmap | ImageBitmap[] | string>()
      try {
        let extracted: Float32Array[]
        if (isRemote) {
          if (!remoteConfig) throw new Error('Remote endpoint not connected')
          extracted = await embedImages(remoteConfig, missInputs as string[])
        } else if (isVlm) {
          if (!vlmTier) throw new Error(`${state.settings.modelVariant} not loaded`)
          const tVlm = performance.now()
          const res = await embedWithVlm(missInputs as ImageBitmap[][])
          // One still alone in the batch is a clean per-photo timing; videos and failures are not.
          if (missInputs.length === 1 && (missInputs[0] as ImageBitmap[]).length === 1 && !res.errors.length) {
            appStillMs.push(performance.now() - tVlm)
          }
          extracted = res.vectors
          embedFailReason ||= res.errors[0] ?? ''
        } else if (isSapiens2) {
          if (!sapiens2Session) throw new Error('Sapiens2 session not loaded')
          extracted = await embedWithSapiens2(sapiens2Session, missInputs as (File | ImageBitmap)[])
        } else if (isChromeAI) {
          if (!chromeAIManager) throw new Error('Chrome AI not initialized')
          if (!textExtractor) throw new Error('Text embedding model not loaded')
          extracted = []
          // Fire all describes in parallel (each using its own session slot)
          const prompt = getChromeAIPrompt()
          const descs = await Promise.all(
            missInputs.map((input, m) =>
              chromeAIManager!.describe(input as ImageBitmap | Blob, prompt, undefined, m)
            )
          )
          const captionWrites: [CacheKey, string][] = []
          for (let m = 0; m < descs.length; m++) {
            const description = descs[m]
            const f = batch[missIndices[m]]
            state.captions[i + missIndices[m]] = description
            captionWrites.push([captionKey(f), description])
            // search_document: prefix for nomic-embed-text indexing (vs search_query: for querying)
            const output = await textExtractor(`search_document: ${description}`, {
              pooling: 'mean',
              normalize: true
            })
            extracted.push(extractVector(output))
          }
          // One transaction for the batch, off the critical path.
          writeCaptions(captionWrites)
        } else {
          if (!extractor) throw new Error('Extractor not loaded')
          const ex = extractor
          // On failure (typically GPU OOM) bisect the batch and retry, so a
          // whole batch is never zero-filled because of one bad input or a
          // transient memory spike.
          extracted = await embedBatchAdaptive(
            missInputs,
            async (chunk) => {
              const output = await ex(chunk.length === 1 ? chunk[0] : chunk, {
                pooling: 'mean',
                normalize: true
              })
              const vecs =
                chunk.length === 1
                  ? [extractVector(output)]
                  : extractBatchedVectors(output, chunk.length)
              batcher.recordSuccess()
              return vecs
            },
            (input, err) => {
              failedInputs.add(input)
              embedFailReason ||= (err as Error).message
              console.warn(
                'Embedding failed for one file, using zero vector:',
                (err as Error).message
              )
              return zeroVector()
            },
            (len, err) => {
              batcher.recordFailure()
              console.warn(
                `Inference failed at batch size ${len}, retrying smaller (working size now ${batcher.size}):`,
                (err as Error).message
              )
            }
          )
        }

        for (let m = 0; m < missIndices.length; m++) {
          const bi = missIndices[m]
          const idx = i + bi
          // Zero-length: this item failed alone in the worker (see embedWithVlm).
          if (extracted[m].length === 0) {
            vectors[idx] = failedVector(idx, embedFailReason || 'worker could not embed this file')
            continue
          }
          vectors[idx] = extracted[m]
          if (failedInputs.has(missInputs[m])) embedFailed.add(idx)
          else writeQueue.push([keys[bi], vectors[idx]])
        }
      } catch (err) {
        console.warn('Batch inference failed, filling zeros:', (err as Error).message)
        embedFailReason ||= (err as Error).message
        // A lost GPU device never comes back for this page: every later
        // batch would fail the same way (a Pixel 7a log had one reset
        // followed by a failure per remaining photo).
        if (/device[^.]*\blost|DEVICE_LOST/i.test((err as Error).message)) gpuLost = true
        // A remote failure usually repeats for every remaining batch (a bad key
        // stays bad), so without a toast the whole run silently zero-fills.
        // One per error kind keeps that from becoming a wall of notifications.
        if (isRemote) {
          const kind = (err as { kind?: string }).kind ?? 'unknown'
          if (!reportedRemoteErrors.has(kind)) {
            reportedRemoteErrors.add(kind)
            showToast(describeOpenAIError(err), 'warn')
          }
        }
        for (const bi of missIndices) {
          vectors[i + bi] = failedVector(i + bi, (err as Error).message)
        }
      }
    }

    for (const b of tempBitmaps) b.close()

    const done = i + batch.length
    if (writeQueue.length >= 32 || performance.now() - lastFlush > 2000 || done >= files.length) {
      await cachePutBatch(writeQueue)
      writeQueue.length = 0
      lastFlush = performance.now()
    }
    opts.onProgress?.(done, files.length)

    if (!headless && done >= 3 && !projecting && performance.now() - lastProjection >= projectionEvery) {
      projecting = true
      lastProjection = performance.now()
      const started = performance.now()
      runProjection(
        vectors.slice(0, done),
        state.settings.projectionMethod,
        Math.max(2, Math.min(15, done - 1)),
        { silent: true }
      )
        .then((raw) => spreadPointsAsync(raw, state.settings.density))
        .then((pts) => {
          if (state.phase !== 'embedding') return
          const fitted = fitInto(pts, grid)
          setPointsAnimated(grid.map((g, j) => (j < done ? fitted[j] : g)))
          scheduleRender()
        })
        .catch((err) => console.error('Progressive projection failed:', err))
        .finally(() => {
          projecting = false
          projectionEvery = Math.max(2500, 4 * (performance.now() - started))
        })
    }

    const fromCache = cacheHits > 0 ? ` (${cacheHits} cached)` : ''
    // Overall rate self-corrects as cache-hit bursts fade; only show the ETA
    // once a few seconds have passed so early estimates aren't nonsense.
    const elapsedSec = (performance.now() - embedStart) / 1000
    const eta =
      elapsedSec > 3 && done > 0
        ? ` · ${(done / elapsedSec).toFixed(1)}/s · ${formatEta(((files.length - done) * elapsedSec) / done)}`
        : ''
    setStatus(`Embedding ${done} / ${files.length} images…${eta}${fromCache}`)
    setProgress(10 + (done / files.length) * 80) // 10% to 90%

    if (gpuLost) {
      setStatus(`The GPU reset — stopped at ${done} / ${files.length} files. Reload the page to retry.`)
      showToast(
        `The GPU reset (device lost) after ${done} of ${files.length} files. ` +
          'Reload the page to retry, or set Image budget to Light (or pick SmolVLM2) in Settings.',
        'error',
        15000
      )
      // Context, not console args: extra args would change the message Bugsink groups by.
      const gpu = await navigator.gpu?.requestAdapter().catch(() => null)
      Sentry.setContext('gpu', {
        vendor: gpu?.info.vendor,
        architecture: gpu?.info.architecture,
        device: gpu?.info.device,
        description: gpu?.info.description,
        maxBufferSize: gpu?.limits.maxBufferSize,
        maxStorageBufferBindingSize: gpu?.limits.maxStorageBufferBindingSize,
        dtype: getVlmTier(state.settings.modelVariant)?.dtype,
        deviceMemoryGB: (navigator as { deviceMemory?: number }).deviceMemory,
        reason: embedFailReason.slice(0, 300)
      })
      console.error(`GPU device lost while embedding: ${state.settings.modelVariant}, ${done}/${files.length} files`)
      await cachePutBatch(writeQueue)
      writeQueue.length = 0
      lastEmbedStop = 'gpu-lost'
      return vectors.slice(0, done)
    }

    const pressure = getMemoryPressure()
    if (pressure !== null && pressure.freeRatio < minFreeRatio) {
      setStatus(
        `Low memory (${(pressure.freeRatio * 100).toFixed(0)}% free) — stopping at ${done} / ${files.length} files…`
      )
      // console.error reaches Bugsink (src/sentry.ts); a status line on a phone does not.
      // Headless runs stop here on purpose to recycle their frame, which is no error.
      ;(headless ? console.info : console.error)(
        `Embedding stopped for low memory: ${state.settings.modelVariant}, ${done}/${files.length} files, ` +
          `${(pressure.freeRatio * 100).toFixed(0)}% JS heap free`
      )
      await cachePutBatch(writeQueue)
      writeQueue.length = 0
      // Overnight frames never reach runMain's report, so their failures were never logged.
      if (headless && embedFailed.size > 0) reportEmbedFailures(files.length)
      lastEmbedStop = 'low-memory'
      return vectors.slice(0, done)
    }

    i = done
    await yieldMain()
  }

  refreshCacheSize()
  if (headless && embedFailed.size > 0) reportEmbedFailures(files.length)
  return vectors
}

// ── Main run ─────────────────────────────────────────────────────────────────
/** Lay every file out in the folder/date grid (viewer mode's arrangement) and
 *  show it now. Returns the files in grid order, so index i matches point i. */
async function showFolderGrid(files: PhotoFile[]): Promise<PhotoFile[]> {
  // Sort by folder then date so n/p navigation follows the visual order.
  const folderGroups = new Map<string, PhotoFile[]>()
  for (const f of files) {
    const folder = f.name.split('/').slice(0, -1).join('/') || '(root)'
    if (!folderGroups.has(folder)) folderGroups.set(folder, [])
    folderGroups.get(folder)!.push(f)
  }
  const sortedFolders = Array.from(folderGroups.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  for (const [, folderFiles] of sortedFolders) folderFiles.sort((a, b) => a.lastModified - b.lastModified)
  const sorted = sortedFolders.flatMap(([, g]) => g)
  state.files = sorted
  thumbDecoding.clear() // in-flight decodes of the old order drop their results
  state.thumbnails = initThumbnails(sorted)
  state.rawPoints = generateMetadataBasedLayout(sorted).map((p) => [p[0], p[1]])
  resizeCanvas()
  state.points = await spreadPointsAsync(state.rawPoints, state.settings.density)
  fitCamera()
  scheduleRender()
  return sorted
}

async function processFiles(files: PhotoFile[]) {
  showWelcome(false)
  if (files.length === 0) {
    setStatus('No images found.')
    return
  }

  // Clean up old resources before processing new files (prevents blob URL errors)
  for (const bmp of state.thumbnails) {
    if (bmp?.close) bmp.close()
  }
  for (const f of state.files) {
    if (f.objectURL) {
      URL.revokeObjectURL(f.objectURL)
      f.objectURL = null
    }
  }
  thumbDecoding.clear()
  thumbFailed.clear()
  embedFailed.clear()
  undecodable.clear()
  embedFailReason = ''
  updateFailedChip()
  thumbnailLRU.clear()
  clearFullTiles(true)

  // Clear points immediately so render() exits early while state.files is being replaced.
  // Without this, a render triggered during async embedding would iterate old point indices
  // against the new (possibly shorter) files array, causing objectURL access on undefined.
  state.points = []
  // Indices into the old list must not outlive it: a breadcrumb/date filter
  // replaces the files while the viewer is open, and its ‹ › or a key press
  // then indexed past the new list (Bugsink: "reading 'name'" / "'0'").
  state.lastViewedIndex = null
  randomHistory = []
  if (dom.modal.open) closeModal()

  state.files = files
  showPeople([]) // its file indices belong to the previous list

  // VIEWER MODE: Skip all AI processing
  if (state.settings.viewerOnly) {
    setStatus(`Viewer mode: ${files.length} files (no AI)`)
    setProgress(10)

    const sortedFiles = await showFolderGrid(files)

    // No semantic clustering in viewer mode - use folder-based colors if needed
    state.clusters = null
    state.vectors = []
    state.phase = 'done' // Set after async work completes
    setProgress(100)

    setStatus(`${sortedFiles.length} media files — viewer mode (arranged by folder & date)`)
    if (dom.statsEl) dom.statsEl.textContent = `${sortedFiles.length} files · viewer mode`
    dom.recenterBtn.disabled = false
    dom.resetBtn.disabled = false
    dom.headerRecenterBtn.disabled = false
    dom.searchInput.disabled = true // No search in viewer mode

    // Save session state for resume (viewer mode)
    const fileKeys = sortedFiles.map((f) => `${f.name}:${f.size}:${f.lastModified}` as CacheKey)
    try {
      localStorage.setItem('po_fileKeys', JSON.stringify(fileKeys))
      localStorage.setItem('po_projectedPoints', JSON.stringify(state.points))
      localStorage.setItem('po_clusters', JSON.stringify([])) // No semantic clusters in viewer mode
      localStorage.setItem('po_viewerMode', 'true') // Flag for resume handler
    } catch (_) {
      showToast(
        "Session state couldn't be saved — browser storage is full. Results are visible but won't be resumable.",
        'warn'
      )
    }

    dom.openBtn.disabled = false
    void runFaces()
    return
  }

  // AI MODE: Full embedding and projection pipeline
  setStatus(`Found ${files.length} media files. Loading thumbnails…`)

  try {
    // Every photo is on screen in the viewer-mode grid from the start; the
    // embedded layout then glides in (setPointsAnimated) instead of photos
    // popping in one by one as they are embedded.
    files = await showFolderGrid(files)

    const vectors = await embedAll(files)

    if (vectors.length < files.length) {
      const n = vectors.length
      state.files = state.files.slice(0, n)
      state.thumbnails = state.thumbnails.slice(0, n)
      files = state.files
    }

    state.vectors = vectors

    state.phase = 'projecting'
    setStatus(`Projecting with ${state.settings.projectionMethod}…`)
    setProgress(90)
    const nNeighbors = Math.max(2, Math.min(15, files.length - 1))
    const rawPoints = await runProjection(vectors, state.settings.projectionMethod, nNeighbors)
    state.rawPoints = rawPoints
    setProgress(94)

    setStatus('Clustering…')
    const k = Math.min(8, Math.max(2, Math.ceil(Math.sqrt(files.length / 2))))
    state.clusters = await kmeansAsync(rawPoints, k, 60, undefined, (f) => setProgress(94 + f * 2))

    state.phase = 'done'
    updateFailedChip()
    if (embedFailed.size > 0) {
      // Failures only reach the console otherwise, which a phone can't show.
      const why = embedFailReason ? `: ${embedFailReason}` : ''
      showToast(`${embedFailed.size} of ${files.length} files could not be embedded${why}`, 'warn', 12000)
      reportEmbedFailures(files.length)
    }
    resizeCanvas()
    setStatus('Arranging layout…')
    setPointsAnimated(
      await spreadPointsAsync(rawPoints, state.settings.density, undefined, (f) =>
        setProgress(96 + f * 4)
      )
    )
    fitCamera()
    scheduleRender()
    setProgress(100)
    const finalMsg = `${files.length} media files · ${k} clusters`
    setStatus(`${files.length} media files — tap to view · ${k} clusters`)
    if (dom.statsEl) dom.statsEl.textContent = finalMsg
    dom.recenterBtn.disabled = false
    dom.resetBtn.disabled = false
    dom.headerRecenterBtn.disabled = false
    dom.searchInput.disabled = false

    state.fileKeys = files.map((f) => `${f.name}:${f.size}:${f.lastModified}`)
    try {
      localStorage.setItem('po_fileKeys', JSON.stringify(state.fileKeys))
      localStorage.setItem('po_projectedPoints', JSON.stringify(state.points))
      localStorage.setItem('po_clusters', JSON.stringify(Array.from(state.clusters)))
    } catch (_) {
      showToast(
        "Session state couldn't be saved — browser storage is full. Results are visible but won't be resumable.",
        'warn'
      )
    }
    void runFaces()
  } catch (err) {
    console.error(err)
    setStatus(`Error: ${(err as Error).message}`)
  } finally {
    dom.openBtn.disabled = false
  }
}

/** Random `n` of `files` (reservoir sampling); all of them when n <= 0 or there are fewer. */
function sampleFiles(files: PhotoFile[], n: number): PhotoFile[] {
  if (n <= 0 || files.length <= n) return files
  const sampled = files.slice(0, n)
  for (let i = n; i < files.length; i++) {
    const j = Math.floor(Math.random() * (i + 1))
    if (j < n) sampled[j] = files[i]
  }
  return sampled
}

// ── Folder filter dialog ────────────────────────────────────────────────────
const FILTER_STORAGE = 'mc_fileFilter'
const filterModal = document.getElementById('filter-modal') as HTMLDialogElement
const filterForm = document.getElementById('filter-form') as HTMLFormElement
const filterCount = document.getElementById('filter-count') as HTMLDivElement
const filterLoadBtn = document.getElementById('filter-load') as HTMLButtonElement

function loadFileFilter(): FileFilter {
  try {
    return { ...DEFAULT_FILTER, ...JSON.parse(localStorage.getItem(FILTER_STORAGE) ?? '{}') }
  } catch {
    return { ...DEFAULT_FILTER }
  }
}

async function loadEmbeddedSets(): Promise<EmbeddedSets> {
  try {
    return buildEmbeddedSets(await cacheKeys(), currentCachePrefix())
  } catch (err) {
    console.warn('Embedding cache unreadable; treating every file as not embedded:', err)
    return { current: new Set(), anyModel: new Set() }
  }
}

function filterField(name: string) {
  return filterForm.elements.namedItem(name) as HTMLInputElement | HTMLSelectElement
}

function readFilterForm(): FileFilter {
  const num = (n: string) => {
    const v = filterField(n).value
    return v === '' || !Number.isFinite(Number(v)) ? null : Number(v)
  }
  // "To today" is the default, so it means open-ended: no EXIF reads, no stale date saved for tomorrow.
  const upTo = (n: string) => (filterField(n).value === todayISO() ? '' : filterField(n).value)
  return {
    images: (filterField('images') as HTMLInputElement).checked,
    videos: (filterField('videos') as HTMLInputElement).checked,
    modifiedFrom: filterField('modifiedFrom').value,
    modifiedTo: upTo('modifiedTo'),
    createdFrom: filterField('createdFrom').value,
    createdTo: upTo('createdTo'),
    minSizeMB: num('minSizeMB'),
    maxSizeMB: num('maxSizeMB'),
    minDurationSec: num('minDurationSec'),
    maxDurationSec: num('maxDurationSec'),
    embedded: filterField('embedded').value as FileFilter['embedded']
  }
}

/** Today as the date input's 'YYYY-MM-DD', in local time. */
function todayISO() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function writeFilterForm(f: FileFilter) {
  ;(filterField('images') as HTMLInputElement).checked = f.images
  ;(filterField('videos') as HTMLInputElement).checked = f.videos
  for (const k of ['modifiedFrom', 'modifiedTo', 'createdFrom', 'createdTo', 'embedded'] as const)
    filterField(k).value = f[k]
  for (const k of ['minSizeMB', 'maxSizeMB', 'minDurationSec', 'maxDurationSec'] as const)
    filterField(k).value = f[k] == null ? '' : String(f[k])
  // Open-ended "to" dates show as today.
  const today = todayISO()
  for (const k of ['modifiedTo', 'createdTo'] as const) {
    filterField(k).value ||= today
    ;(filterField(k) as HTMLInputElement).max = today
  }
}

/** Let the user narrow a scanned folder. Resolves to the matches, or null on cancel. */
// ── Saved-embeddings bundles (export / import / next to the photos) ──────────
const BUNDLE_SEEN = 'mc_bundleSeen' // size:lastModified of bundle files already imported from folders

/** Store a bundle's entries. With `files`, entries of another device's copies are moved onto them by sha256. */
async function importBundle(buf: ArrayBuffer, files?: PhotoFile[]): Promise<number> {
  let bundle = decodeBundle(buf)
  if (files) {
    const byKey = new Map(files.map((f) => [makeCacheKey(f), f]))
    let hashed = 0
    ;({ bundle } = await rekeyBundle(bundle, [...byKey.keys()], (k) => {
      if (++hashed % 50 === 0) setStatus(`Matching saved embeddings to this folder's files… ${hashed} hashed`)
      return sha256Hex(byKey.get(k)!.file)
    }))
  }
  const { entries, captions } = bundle
  for (let i = 0; i < entries.length; i += 1000) {
    await cachePutBatch(entries.slice(i, i + 1000) as [CacheKey, Float32Array][])
  }
  await captionPutBatch(captions as [CacheKey, string][])
  refreshCacheSize()
  return entries.length
}

/** With `onlyFiles`, just their entries, plus each file's sha256 so other devices can match them. */
async function bundleBlob(onlyFiles?: PhotoFile[]): Promise<{ blob: Blob; count: number }> {
  const byKey = new Map((onlyFiles ?? []).map((f) => [makeCacheKey(f), f]))
  const { entries, captions } = await exportAll(onlyFiles && new Set(byKey.keys()))
  const sha256: Record<string, string> = {}
  for (const fk of new Set(entries.map(([k]) => fileKeyOf(k)))) {
    const f = byKey.get(fk)
    const h = f && (await sha256Hex(f.file))
    if (h) sha256[fk] = h
    if (Object.keys(sha256).length % 50 === 0) setStatus(`Hashing files for the embeddings file… ${Object.keys(sha256).length}`)
  }
  return { blob: encodeBundle({ entries, captions, sha256 }), count: entries.length }
}

/** Write the folder's own embeddings into it as BUNDLE_FILE. Needs readwrite permission on `dir`. */
async function saveBundleToFolder(dir: DirectoryHandle, files: PhotoFile[]): Promise<number> {
  const { blob, count } = await bundleBlob(files)
  const handle = await dir.getFileHandle(BUNDLE_FILE, { create: true })
  const w = await handle.createWritable()
  await w.write(blob)
  await w.close()
  // Our own write is not news on the next open of this folder.
  const f = await handle.getFile()
  rememberBundle(f)
  return count
}

function rememberBundle(f: File) {
  try {
    const seen = JSON.parse(localStorage.getItem(BUNDLE_SEEN) ?? '[]') as string[]
    localStorage.setItem(BUNDLE_SEEN, JSON.stringify([...seen.slice(-50), `${f.size}:${f.lastModified}`]))
  } catch (_) {
    // storage full: the bundle is imported again next time, which is harmless
  }
}

/** Import a BUNDLE_FILE found in an opened folder, once per version of that file. */
async function importFolderBundle(f: File | null | undefined, files: PhotoFile[]) {
  if (!f) return
  try {
    const seen = JSON.parse(localStorage.getItem(BUNDLE_SEEN) ?? '[]') as string[]
    if (seen.includes(`${f.size}:${f.lastModified}`)) return
    setStatus(`Importing saved embeddings from ${BUNDLE_FILE}…`)
    const n = await importBundle(await f.arrayBuffer(), files)
    rememberBundle(f)
    showToast(`Imported ${n.toLocaleString()} saved embeddings from this folder.`, 'info')
  } catch (err) {
    showToast(`Could not import ${BUNDLE_FILE}: ${(err as Error).message}`, 'warn', 10000)
  }
}

const bundleInFolder = (dir: DirectoryHandle) =>
  dir.getFileHandle(BUNDLE_FILE).then((h) => h.getFile(), () => null)

async function openFilterDialog(files: PhotoFile[], bundle?: File | null): Promise<PhotoFile[] | null> {
  await importFolderBundle(bundle, files)
  const embedded = await loadEmbeddedSets()
  writeFilterForm(loadFileFilter())
  const sample = state.settings.randomSampleSize
  const ac = new AbortController()
  let gen = 0
  let matched: PhotoFile[] = []
  let timer: ReturnType<typeof setTimeout> | undefined

  const recount = async () => {
    const my = ++gen
    const f = readFilterForm()
    try {
      localStorage.setItem(FILTER_STORAGE, JSON.stringify(f))
    } catch (_) {
      // storage full: the filter still works for this load
    }
    filterLoadBtn.disabled = true
    filterCount.textContent = 'Counting…'
    const res = await applyFileFilter(
      files,
      f,
      embedded,
      (d, t) => {
        if (my === gen && (d % 25 === 0 || d === t)) filterCount.textContent = `Reading metadata… ${d}/${t}`
      },
      () => my !== gen
    )
    if (!res || my !== gen) return
    matched = res
    const note = sample > 0 && res.length > sample ? ` · a random ${sample} will load (Random Sample Size in Settings)` : ''
    filterCount.textContent = `${res.length} of ${files.length} files match${note}`
    filterLoadBtn.disabled = res.length === 0
  }
  const schedule = () => {
    clearTimeout(timer)
    timer = setTimeout(recount, 250)
  }

  return new Promise((resolve) => {
    const finish = (result: PhotoFile[] | null) => {
      gen++ // stops any in-flight count
      clearTimeout(timer)
      ac.abort()
      filterModal.close()
      resolve(result)
    }
    const opts = { signal: ac.signal }
    filterForm.addEventListener('input', schedule, opts)
    filterForm.addEventListener('change', schedule, opts)
    filterForm.addEventListener(
      'submit',
      (e) => {
        e.preventDefault()
        if (!filterLoadBtn.disabled) finish(matched)
      },
      opts
    )
    document.getElementById('filter-cancel')!.addEventListener('click', () => finish(null), opts)
    document.getElementById('filter-reset')!.addEventListener(
      'click',
      () => {
        writeFilterForm(DEFAULT_FILTER)
        schedule()
      },
      opts
    )
    filterModal.addEventListener(
      'cancel',
      (e) => {
        e.preventDefault()
        finish(null)
      },
      opts
    )
    filterModal.showModal()
    recount()
  })
}

async function run(dirHandle: DirectoryHandle, basePath: string = '') {
  dom.openBtn.disabled = true
  setProgress(0)

  try {
    state.currentDirHandle = dirHandle
    state.currentBasePath = basePath
    setStatus('Scanning folder…')
    // Scan everything so the filters see the whole folder; sampling happens after.
    const all = await collectImages(dirHandle, 0, basePath)
    const files = await openFilterDialog(all, await bundleInFolder(dirHandle))
    if (!files) {
      setStatus('Cancelled.')
      dom.openBtn.disabled = false
      return
    }
    await processFiles(sampleFiles(files, state.settings.randomSampleSize))

    // Update URL to show current folder (clears any filter state)
    updateURL({ type: 'folder', path: basePath })
  } catch (err) {
    console.error(err)
    setStatus(`Error: ${(err as Error).message}`)
    dom.openBtn.disabled = false
  }
}

// Navigate to a subfolder (reuses current directory handle)
async function navigateToFolder(targetPath: string) {
  if (!state.currentDirHandle) return

  // Close modal first
  closeModal()

  // Re-run with new base path
  await run(state.currentDirHandle, targetPath)

  // Update URL for bookmarking/back-forward
  updateURL({ type: 'folder', path: targetPath })
}

// Filter files by datetime range - rescans original folder
async function filterByDateTime(
  granularity: 'year' | 'month' | 'day' | 'hour' | 'minute',
  year: number,
  month?: number,
  day?: number,
  hour?: number,
  minute?: number
) {
  if (!state.currentDirHandle) return

  // Close modal first
  closeModal()

  // Re-scan the original folder (no random sample, get all files)
  setStatus(`Scanning folder for ${granularity}…`)
  const allFiles = await collectImages(state.currentDirHandle, 0, state.currentBasePath)

  // Filter by datetime range
  // The saved folder filter still applies; the date link narrows it further.
  const saved = loadFileFilter()
  const base =
    (await applyFileFilter(allFiles, saved, await loadEmbeddedSets(), (d, t) =>
      setStatus(`Reading metadata… ${d}/${t}`)
    )) ?? allFiles
  const filtered = base.filter((f) => {
    const d = new Date(f.lastModified)
    if (d.getFullYear() !== year) return false
    if (month !== undefined && d.getMonth() !== month) return false
    if (day !== undefined && d.getDate() !== day) return false
    if (hour !== undefined && d.getHours() !== hour) return false
    if (minute !== undefined && d.getMinutes() !== minute) return false
    return true
  })

  if (filtered.length === 0) {
    setStatus('No files found in this time range.')
    return
  }

  // Apply random sample limit if configured and filtered results exceed it
  const finalFiles = sampleFiles(filtered, state.settings.randomSampleSize)

  // Update display name for status
  const rangeDesc =
    granularity === 'year'
      ? year.toString()
      : granularity === 'month'
        ? `${year}-${(month! + 1).toString().padStart(2, '0')}`
        : granularity === 'day'
          ? `${year}-${(month! + 1).toString().padStart(2, '0')}-${day!.toString().padStart(2, '0')}`
          : granularity === 'hour'
            ? `${year}-${(month! + 1).toString().padStart(2, '0')}-${day!.toString().padStart(2, '0')} ${hour!.toString().padStart(2, '0')}:00`
            : `${year}-${(month! + 1).toString().padStart(2, '0')}-${day!.toString().padStart(2, '0')} ${hour!.toString().padStart(2, '0')}:${minute!.toString().padStart(2, '0')}`

  // Clear current state and process filtered files
  state.phase = 'idle'
  state.files = []
  state.vectors = []
  state.points = []
  state.rawPoints = null
  state.clusters = null
  state.thumbnails = []
  state.searchResults = null
  state.searchQuery = ''
  state.searchScores = null

  // If extractor isn't loaded, ensure we're in viewer-only mode for filtering
  const wasViewerOnly = state.settings.viewerOnly
  if (!extractor) {
    state.settings.viewerOnly = true
  }

  const sampleNote =
    finalFiles.length < filtered.length
      ? ` (sampled ${finalFiles.length} of ${filtered.length})`
      : ''
  setStatus(`Found ${filtered.length} files from ${rangeDesc}.${sampleNote} Processing...`)

  try {
    await processFiles(finalFiles)
  } finally {
    // Restore original viewer-only setting if we temporarily forced it
    if (!extractor && !wasViewerOnly) {
      state.settings.viewerOnly = wasViewerOnly
    }
  }

  // Update URL for bookmarking/back-forward
  updateURL({ type: 'datetime', granularity, year, month, day, hour, minute })
}

// ── URL State Management ───────────────────────────────────────────────────────
// URL format: #folder:path/to/folder or #dt:2025, #dt:2025-01, #dt:2025-01-15, #dt:2025-01-15T14, #dt:2025-01-15T14:30

type URLState =
  | { type: 'folder'; path: string }
  | {
      type: 'datetime'
      granularity: string
      year: number
      month?: number
      day?: number
      hour?: number
      minute?: number
    }
  | null

function parseURLHash(hash: string): URLState {
  if (!hash || hash === '#') return null

  const content = hash.slice(1) // Remove #

  if (content.startsWith('folder:')) {
    return { type: 'folder', path: content.slice(7) }
  }

  if (content.startsWith('dt:')) {
    const dtStr = content.slice(3)
    // Parse datetime, as coarse as a year: 2025, 2025-01, ... 2025-01-15T14:30
    const parts = dtStr.match(/^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?(?:T(\d{2}))?(?::(\d{2}))?$/)
    if (parts) {
      const [, year, month, day, hour, minute] = parts
      return {
        type: 'datetime',
        granularity: minute ? 'minute' : hour ? 'hour' : day ? 'day' : month ? 'month' : 'year',
        year: parseInt(year),
        month: month ? parseInt(month) - 1 : undefined,
        day: day ? parseInt(day) : undefined,
        hour: hour ? parseInt(hour) : undefined,
        minute: minute ? parseInt(minute) : undefined
      }
    }
  }

  return null
}

function updateURL(state: URLState) {
  if (!state) {
    history.replaceState(null, '', '#')
    return
  }

  let hash = ''
  if (state.type === 'folder') {
    hash = `#folder:${state.path}`
  } else if (state.type === 'datetime') {
    // Only the parts the filter has: a year or month filter has no day
    // (day!.toString() threw there, Bugsink MEDIA-CLUSTERER-27).
    const { year, month, day, hour, minute } = state
    const pad = (n: number) => n.toString().padStart(2, '0')
    hash = `#dt:${year}`
    if (month !== undefined) hash += `-${pad(month + 1)}`
    if (day !== undefined) hash += `-${pad(day)}`
    if (hour !== undefined) hash += `T${pad(hour)}`
    if (minute !== undefined) hash += `:${pad(minute)}`
  }

  history.pushState(state, '', hash)
}

// Handle back/forward navigation
window.addEventListener('popstate', (e) => {
  const urlState = parseURLHash(window.location.hash)
  if (!urlState) {
    // No state - reset to original folder
    if (state.currentBasePath) {
      navigateToFolder('')
    }
    return
  }

  if (urlState.type === 'folder') {
    navigateToFolder(urlState.path)
  } else if (urlState.type === 'datetime') {
    filterByDateTime(
      urlState.granularity as 'year' | 'month' | 'day' | 'hour' | 'minute',
      urlState.year,
      urlState.month,
      urlState.day,
      urlState.hour,
      urlState.minute
    )
  }
})

// ── Interaction ──────────────────────────────────────────────────────────────
const pointers = new Map<number, PointerState>()
let lastPinchDist = 0
let dragMoved = 0

function pointerPos(e: PointerEvent): CanvasPointerPos {
  const rect = dom.canvas.getBoundingClientRect()
  return {
    cx: (e.clientX - rect.left) * (dom.canvas.width / rect.width),
    cy: (e.clientY - rect.top) * (dom.canvas.height / rect.height)
  }
}

dom.canvas.addEventListener('pointerdown', (e) => {
  e.preventDefault()
  dom.canvas.setPointerCapture(e.pointerId)
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
  dragMoved = 0
})

dom.canvas.addEventListener('pointermove', (e) => {
  e.preventDefault()
  const prev = pointers.get(e.pointerId)
  if (!prev) return
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })

  if (pointers.size === 1) {
    const dx = e.clientX - prev.x
    const dy = e.clientY - prev.y
    dragMoved += Math.abs(dx) + Math.abs(dy)
    camera.x -= dx / camera.scale
    camera.y -= dy / camera.scale
    state.lastViewedIndex = null
    cameraFollow = false
    scheduleRender()
  } else if (pointers.size === 2) {
    const pts = [...pointers.values()]
    const dx = pts[0].x - pts[1].x,
      dy = pts[0].y - pts[1].y
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (lastPinchDist > 0) {
      const midX = (pts[0].x + pts[1].x) / 2
      const midY = (pts[0].y + pts[1].y) / 2
      const rect = dom.canvas.getBoundingClientRect()
      const scaleRatio = dom.canvas.width / rect.width
      const px = (midX - rect.left) * scaleRatio - dom.canvas.width / 2
      const py = (midY - rect.top) * scaleRatio - dom.canvas.height / 2
      const ratio = dist / lastPinchDist
      camera.x += px / camera.scale - px / (camera.scale * ratio)
      camera.y += py / camera.scale - py / (camera.scale * ratio)
      camera.scale = Math.max(0.05, Math.min(20, camera.scale * ratio))
      state.lastViewedIndex = null
      cameraFollow = false
      scheduleRender()
    }
    lastPinchDist = dist
  }
})

dom.canvas.addEventListener('pointerup', (e) => {
  e.preventDefault()
  const wasSingleTap = pointers.size === 1 && dragMoved < 15
  pointers.delete(e.pointerId)
  if (pointers.size < 2) lastPinchDist = 0

  if (wasSingleTap && state.points.length) {
    const { cx, cy } = pointerPos(e)
    const label = clusterLabelHits.find((r) => cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h)
    if (label) {
      zoomToCluster(label.cluster)
      return
    }
    const wx = (cx - dom.canvas.width / 2) / camera.scale + camera.x
    const wy = (cy - dom.canvas.height / 2) / camera.scale + camera.y
    const hitRadius = (THUMB_WORLD / 2) ** 2
    let closest = -1,
      minD = hitRadius * 4
    for (let i = 0; i < state.points.length; i++) {
      if (isHidden(i)) continue
      const ddx = state.points[i][0] - wx,
        ddy = state.points[i][1] - wy
      const d = ddx * ddx + ddy * ddy
      if (d < minD) {
        minD = d
        closest = i
      }
    }
    if (closest >= 0) {
      openFileModal(closest)
    }
  }
})

dom.canvas.addEventListener('pointercancel', (e) => {
  pointers.delete(e.pointerId)
  if (pointers.size < 2) lastPinchDist = 0
})

dom.canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault()
    const factor = e.deltaY > 0 ? 0.88 : 1.14
    const rect = dom.canvas.getBoundingClientRect()
    const px = (e.clientX - rect.left) * (dom.canvas.width / rect.width) - dom.canvas.width / 2
    const py = (e.clientY - rect.top) * (dom.canvas.height / rect.height) - dom.canvas.height / 2
    camera.x += px / camera.scale - px / (camera.scale * factor)
    camera.y += py / camera.scale - py / (camera.scale * factor)
    camera.scale = Math.max(0.05, Math.min(20, camera.scale * factor))
    state.lastViewedIndex = null
    cameraFollow = false
    scheduleRender()
  },
  { passive: false }
)

dom.recenterBtn.addEventListener('click', () => {
  cameraFollow = true
  fitCamera()
  scheduleRender()
})
dom.resetBtn.addEventListener('click', resetAll)
dom.failedChip.addEventListener('click', () => {
  hideFailed = !hideFailed
  if (state.activeFileIndex === null && isHidden(state.lastViewedIndex ?? -1)) state.lastViewedIndex = null
  updateFailedChip()
  scheduleRender()
})
dom.headerRecenterBtn.addEventListener('click', () => {
  cameraFollow = true
  fitCamera()
  scheduleRender()
})

// ── Search input ─────────────────────────────────────────────────────────────
let searchDebounce: ReturnType<typeof setTimeout> | null = null

dom.searchInput.addEventListener('input', () => {
  dom.searchClearBtn.hidden = !dom.searchInput.value.trim()

  if (searchDebounce) clearTimeout(searchDebounce)
  searchDebounce = setTimeout(async () => {
    if (dom.searchInput.value.trim()) {
      await searchImages(dom.searchInput.value)
    } else {
      state.searchResults = null
      state.searchQuery = ''
      state.searchScores = null
      dom.searchClearBtn.hidden = true
      setStatus(`${state.files.length} images · tap to view`)
    }
    scheduleRender()
  }, 300)
})

dom.searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    dom.searchInput.value = ''
    state.searchResults = null
    state.searchQuery = ''
    state.searchScores = null
    dom.searchClearBtn.hidden = true
    setStatus(`${state.files.length} images · tap to view`)
    scheduleRender()
  }
})

dom.searchClearBtn.addEventListener('click', () => {
  selectPerson(null)
  dom.searchInput.value = ''
  state.searchResults = null
  state.searchQuery = ''
  state.searchScores = null
  dom.searchClearBtn.hidden = true
  dom.searchInput.focus()
  setStatus(`${state.files.length} images · tap to view`)
  scheduleRender()
})

const openFileModal = (index: number) => {
  const f = state.files[index]
  if (!f) return
  dom.modalPosition.textContent = `${sequencePosition(index) + 1} / ${state.files.length}`
  dom.modalPosition.title = state.settings.viewerOnly ? 'Position in folder order' : 'Position by date'
  const ext = f.name.split('.').pop()?.toLowerCase() ?? ''

  if (VIDEO_EXTS.has(ext)) {
    dom.modalImg.style.display = 'none'
    dom.modalVideo.style.display = 'block'
    // A looping video never ends, so a slideshow waiting on it would stall.
    dom.modalVideo.loop = state.settings.loopVideos && !slideshowWaitsOnVideos()
    // Ensure objectURL exists (created lazily during thumbnail decode)
    if (!f.objectURL) f.objectURL = URL.createObjectURL(f.file)
    dom.modalVideo.src = f.objectURL
    dom.modalVideo.play().catch(() => {}) // Autoplay when opened
  } else {
    dom.modalVideo.style.display = 'none'
    dom.modalVideo.pause()
    dom.modalVideo.src = ''
    dom.modalImg.style.display = 'block'
    // Ensure objectURL exists (created lazily during thumbnail decode); without
    // this, navigating (n/p) to an image not yet rendered on-canvas shows blank.
    if (!f.objectURL) f.objectURL = URL.createObjectURL(f.file)
    dom.modalImg.src = f.objectURL
  }

  // Populate modal footer with metadata
  const pathParts = f.name.split('/')
  const filename = pathParts.pop() || ''

  dom.modalFilename.textContent = filename

  // Clear and create clickable breadcrumb navigation
  while (dom.modalPath.firstChild) {
    dom.modalPath.removeChild(dom.modalPath.firstChild)
  }

  if (pathParts.length === 0) {
    const rootSpan = document.createElement('span')
    rootSpan.className = 'modal-link'
    rootSpan.textContent = '(root)'
    rootSpan.onclick = () => navigateToFolder('')
    dom.modalPath.appendChild(rootSpan)
  } else {
    pathParts.forEach((part, i) => {
      if (i > 0) {
        const sep = document.createElement('span')
        sep.className = 'modal-sep'
        sep.textContent = ' / '
        dom.modalPath.appendChild(sep)
      }
      const link = document.createElement('span')
      link.className = 'modal-link'
      link.textContent = part
      const targetPath = pathParts.slice(0, i + 1).join('/')
      link.onclick = () => navigateToFolder(targetPath)
      dom.modalPath.appendChild(link)
    })
  }

  // Up one level button
  if (pathParts.length > 0) {
    dom.modalUp.style.visibility = 'visible'
    dom.modalUp.onclick = () => {
      const upPath = pathParts.slice(0, -1).join('/')
      navigateToFolder(upPath)
    }
  } else {
    dom.modalUp.style.visibility = 'hidden'
  }

  // Create clickable datetime breadcrumbs
  const date = new Date(f.lastModified)
  while (dom.modalDatetime.firstChild) {
    dom.modalDatetime.removeChild(dom.modalDatetime.firstChild)
  }

  // Year
  const yearLink = document.createElement('span')
  yearLink.className = 'modal-link'
  yearLink.textContent = date.getFullYear().toString()
  yearLink.onclick = () => filterByDateTime('year', date.getFullYear())
  dom.modalDatetime.appendChild(yearLink)

  // Month
  const monthSep = document.createElement('span')
  monthSep.className = 'modal-sep'
  monthSep.textContent = '/'
  dom.modalDatetime.appendChild(monthSep)
  const monthLink = document.createElement('span')
  monthLink.className = 'modal-link'
  monthLink.textContent = (date.getMonth() + 1).toString().padStart(2, '0')
  monthLink.onclick = () => filterByDateTime('month', date.getFullYear(), date.getMonth())
  dom.modalDatetime.appendChild(monthLink)

  // Day
  const daySep = document.createElement('span')
  daySep.className = 'modal-sep'
  daySep.textContent = '/'
  dom.modalDatetime.appendChild(daySep)
  const dayLink = document.createElement('span')
  dayLink.className = 'modal-link'
  dayLink.textContent = date.getDate().toString().padStart(2, '0')
  dayLink.onclick = () =>
    filterByDateTime('day', date.getFullYear(), date.getMonth(), date.getDate())
  dom.modalDatetime.appendChild(dayLink)

  // Hour
  const hourSep = document.createElement('span')
  hourSep.className = 'modal-sep'
  hourSep.textContent = ' '
  dom.modalDatetime.appendChild(hourSep)
  const hourLink = document.createElement('span')
  hourLink.className = 'modal-link'
  hourLink.textContent = date.getHours().toString().padStart(2, '0')
  hourLink.onclick = () =>
    filterByDateTime('hour', date.getFullYear(), date.getMonth(), date.getDate(), date.getHours())
  dom.modalDatetime.appendChild(hourLink)

  // Minute
  const minSep = document.createElement('span')
  minSep.className = 'modal-sep'
  minSep.textContent = ':'
  dom.modalDatetime.appendChild(minSep)
  const minLink = document.createElement('span')
  minLink.className = 'modal-link'
  minLink.textContent = date.getMinutes().toString().padStart(2, '0')
  minLink.onclick = () =>
    filterByDateTime(
      'minute',
      date.getFullYear(),
      date.getMonth(),
      date.getDate(),
      date.getHours(),
      date.getMinutes()
    )
  dom.modalDatetime.appendChild(minLink)

  // Store date for updateSizeInfo closure
  const fileDate = date

  // Update size and resolution after media loads
  const updateSizeInfo = () => {
    const width =
      dom.modalImg.style.display !== 'none' ? dom.modalImg.naturalWidth : dom.modalVideo.videoWidth
    const height =
      dom.modalImg.style.display !== 'none'
        ? dom.modalImg.naturalHeight
        : dom.modalVideo.videoHeight

    if (width && height) {
      const sizeMB = (f.size / (1024 * 1024)).toFixed(f.size < 1024 * 1024 ? 2 : 1)
      const sizeKB = (f.size / 1024).toFixed(0)
      const sizeStr = f.size < 1024 * 1024 ? `${sizeKB} KB` : `${sizeMB} MB`
      dom.modalMeta.textContent = `${width}×${height} · ${sizeStr}`
    }
  }

  // Try immediately (might be cached), otherwise wait for load
  updateSizeInfo()
  if (dom.modalImg.style.display !== 'none') {
    dom.modalImg.onload = updateSizeInfo
  } else {
    dom.modalVideo.onloadedmetadata = updateSizeInfo
  }

  // EXIF: full parse lazily; result feeds footer GPS and detail dialog
  const gpsSep = document.getElementById('modal-gps-sep') as HTMLSpanElement
  const exifSep = document.getElementById('modal-exif-sep') as HTMLSpanElement
  const applyExif = (exif: Record<string, unknown> | null) => {
    const lat = exif?.latitude as number | undefined
    const lon = exif?.longitude as number | undefined
    if (typeof lat === 'number' && typeof lon === 'number') {
      dom.modalGps.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`
      dom.modalGps.href = `https://www.openstreetmap.org/?mlat=${lat.toFixed(5)}&mlon=${lon.toFixed(5)}&zoom=15`
      dom.modalGps.style.display = ''
      gpsSep.style.display = ''
    } else {
      dom.modalGps.style.display = 'none'
      gpsSep.style.display = 'none'
    }
    const hasExif = exif && Object.keys(exif).length > 0
    dom.modalExifBtn.style.display = hasExif ? '' : 'none'
    exifSep.style.display = hasExif ? '' : 'none'
  }

  if (f.exifData !== undefined) {
    applyExif(f.exifData)
  } else {
    applyExif(null)
    exifr
      .parse(f.file, {
        gps: true,
        exif: true,
        iptc: false,
        xmp: false,
        icc: false,
        jfif: false
      })
      .then((exif) => {
        f.exifData = exif ?? null
        f.gps =
          typeof exif?.latitude === 'number' && typeof exif?.longitude === 'number'
            ? {
                latitude: exif.latitude as number,
                longitude: exif.longitude as number
              }
            : null
        if (state.activeFileIndex === index) applyExif(f.exifData ?? null)
      })
      .catch((err) => {
        console.warn('EXIF parse failed:', err)
        f.exifData = null
        f.gps = null
      })
  }

  // AI provenance: read lazily from the file's own bytes, once per file.
  const aiBtn = document.getElementById('modal-ai-btn') as HTMLButtonElement
  const showAi = (ai: Provenance | null | undefined) => {
    aiBtn.hidden = !ai
    ;(document.getElementById('modal-ai-sep') as HTMLElement).hidden = !ai
    if (ai) aiBtn.textContent = `✨ AI · ${ai.generator}${ai.model ? ` · ${ai.model}` : ''} ▸`
  }
  showAi(f.ai)
  if (f.ai === undefined && !VIDEO_EXTS.has(f.name.split('.').pop()?.toLowerCase() ?? '')) {
    void readFileProvenance(f).then(() => state.activeFileIndex === index && showAi(f.ai))
  }

  // Cancel any in-flight lazy caption and pending debounce from a previous modal open
  captionAbortController?.abort()
  captionAbortController = null
  if (captionDebounceTimer !== null) {
    clearTimeout(captionDebounceTimer)
    captionDebounceTimer = null
  }

  // Resolving a stored caption is now async (IndexedDB), so the whole
  // show-or-generate decision moves into the continuation. Deciding before the
  // read returned would race: the 400 ms debounce below would usually win, and
  // we would regenerate a caption that was already on disk.
  const resolveCaption = state.captions[index]
    ? Promise.resolve(state.captions[index] ?? null)
    : readCaptions([f]).then(([stored]) => {
        if (stored) state.captions[index] = stored
        return stored
      })

  void resolveCaption.then((caption) => {
    // The user may have moved on while the store was read.
    if (state.activeFileIndex !== index) return
    showCaptionOrGenerate(index, f, caption)
  })

  dom.modal.showModal()
  state.activeFileIndex = index
  return finishOpenFileModal(index)
}

/** Render a known caption, or kick off lazy generation when enabled. */
function showCaptionOrGenerate(index: number, f: PhotoFile, caption: string | null): void {
  if (caption) {
    dom.modalCaption.textContent = caption
    dom.modalCaption.style.display = 'block'
    dom.modalFooter.style.borderRadius = '0'
  } else if (state.settings.enableLazyCaption && chromeAIAvailability !== 'unavailable') {
    // Hide until the debounce fires — no flash when quickly flipping images
    dom.modalCaption.style.display = 'none'
    dom.modalFooter.style.borderRadius = ''

    const ac = new AbortController()
    captionAbortController = ac
    const captureIndex = index
    const captureFile = f

    captionDebounceTimer = setTimeout(async () => {
      captionDebounceTimer = null
      if (ac.signal.aborted) return

      // Show placeholder now that the user has paused on this image
      dom.modalCaption.textContent = 'Generating caption…'
      dom.modalCaption.style.display = 'block'
      dom.modalFooter.style.borderRadius = '0'

      try {
        if (!lazyCaptionManager) lazyCaptionManager = new ChromeAISessionManager()
        const thumb = state.thumbnails[captureIndex]
        const img = thumb
          ? await createImageBitmap(thumb)
          : await createImageBitmap(captureFile.file, {
              resizeWidth: 128,
              resizeQuality: 'medium'
            })
        if (ac.signal.aborted) {
          img.close()
          return
        }
        const desc = await lazyCaptionManager.describe(img, getChromeAIPrompt(), ac.signal)
        img.close()
        if (ac.signal.aborted) return
        state.captions[captureIndex] = desc
        writeCaptions([[captionKey(captureFile), desc]])
        if (state.activeFileIndex === captureIndex) {
          dom.modalCaption.textContent = desc
        }
      } catch {
        if (!ac.signal.aborted && state.activeFileIndex === captureIndex) {
          dom.modalCaption.style.display = 'none'
          dom.modalFooter.style.borderRadius = ''
        }
      }
    }, 400)
  } else {
    dom.modalCaption.style.display = 'none'
    dom.modalFooter.style.borderRadius = ''
  }
}

/** Everything openFileModal did after the caption block. */
function finishOpenFileModal(index: number): void {
  state.lastViewedIndex = index

  // Center camera on the active image
  const pt = state.points[index]
  if (pt) {
    camera.x = pt[0]
    camera.y = pt[1]
    scheduleRender()
  }
}

const closeModal = () => {
  dom.modal.close()
}

// Cleanup happens on the native close event (handles X button, Escape, and backdrop click)
dom.modal.addEventListener('close', () => {
  stopSlideshow()
  slideshowOrder = null
  dom.modalVideo.pause()
  dom.modalVideo.src = ''
  state.activeFileIndex = null
  if (captionDebounceTimer !== null) {
    clearTimeout(captionDebounceTimer)
    captionDebounceTimer = null
  }
  captionAbortController?.abort()
  captionAbortController = null
})

dom.modalClose.addEventListener('click', closeModal)

// Backdrop click: dialog is fullscreen overlay, click on dialog outside content closes it
dom.modal.addEventListener('click', (e) => {
  if (e.target === dom.modal) closeModal()
})

// ── EXIF detail dialog ───────────────────────────────────────────────────────
const exifDialog = document.getElementById('exif-dialog') as HTMLDialogElement
const exifDialogBody = document.getElementById('exif-dialog-body') as HTMLDivElement
const exifDialogClose = document.getElementById('exif-dialog-close') as HTMLButtonElement

const EXIF_LABELS: Record<string, string> = {
  Make: 'Camera Make',
  Model: 'Camera Model',
  LensModel: 'Lens',
  FNumber: 'Aperture',
  ExposureTime: 'Shutter Speed',
  ISO: 'ISO',
  FocalLength: 'Focal Length',
  FocalLengthIn35mmFormat: '35mm Equiv.',
  DateTimeOriginal: 'Date Taken',
  CreateDate: 'Date Created',
  ImageWidth: 'Width',
  ImageHeight: 'Height',
  Orientation: 'Orientation',
  Flash: 'Flash',
  WhiteBalance: 'White Balance',
  ExposureMode: 'Exposure Mode',
  ExposureProgram: 'Exposure Program',
  MeteringMode: 'Metering Mode',
  ColorSpace: 'Color Space',
  Software: 'Software',
  Artist: 'Artist',
  Copyright: 'Copyright',
  latitude: 'Latitude',
  longitude: 'Longitude'
}

function formatExifValue(key: string, val: unknown): string {
  if (val === null || val === undefined) return ''
  if (key === 'FNumber') return `f/${val}`
  if (key === 'ExposureTime') {
    const s = val as number
    return s < 1 ? `1/${Math.round(1 / s)}s` : `${s}s`
  }
  if (key === 'FocalLength' || key === 'FocalLengthIn35mmFormat') return `${val}mm`
  if (key === 'latitude' || key === 'longitude') return (val as number).toFixed(6)
  if (val instanceof Date) return val.toLocaleString()
  return String(val)
}

/** Fill f.ai from the file. shortcut: reads the whole file, so files over 64 MB are skipped; read only the header regions if huge PNGs matter. */
async function readFileProvenance(f: PhotoFile) {
  try {
    if (f.size > 64 << 20) return void (f.ai = null)
    const exif = await exifr.parse(f.file, { pick: ['UserComment', 'Software', 'ImageDescription'] }).catch(() => null)
    f.ai = await readProvenance(new Uint8Array(await f.file.arrayBuffer()), exif ?? null)
  } catch (err) {
    console.warn('AI provenance read failed:', (err as Error).message)
    f.ai = null
  }
}

document.getElementById('modal-ai-btn')!.addEventListener('click', () => {
  const ai = state.activeFileIndex === null ? null : state.files[state.activeFileIndex]?.ai
  if (!ai) return
  ;(document.getElementById('exif-dialog-title') as HTMLElement).textContent = 'AI-generated'
  exifDialogBody.innerHTML = ''
  const dl = document.createElement('dl')
  dl.style.cssText = 'display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0;'
  const row = (label: string, value: string | undefined) => {
    if (!value) return
    const dt = document.createElement('dt')
    dt.style.cssText = 'color:var(--text-dim);white-space:nowrap;'
    dt.textContent = label
    const dd = document.createElement('dd')
    dd.style.cssText = 'margin:0;color:var(--text-main);white-space:pre-wrap;word-break:break-word;'
    dd.textContent = value
    dl.append(dt, dd)
  }
  row('Generator', ai.generator)
  row('Model', ai.model)
  row('Prompt', ai.prompt)
  row('Negative prompt', ai.negative)
  row('Settings', ai.settings)
  row('Evidence', ai.evidence.map((e) => `• ${e}`).join('\n'))
  exifDialogBody.appendChild(dl)
  exifDialog.showModal()
})

dom.modalExifBtn.addEventListener('click', () => {
  const idx = state.activeFileIndex
  if (idx === null) return
  const exif = state.files[idx]?.exifData
  if (!exif) return

  ;(document.getElementById('exif-dialog-title') as HTMLElement).textContent = 'EXIF Data'
  exifDialogBody.innerHTML = ''
  const dl = document.createElement('dl')
  dl.style.cssText = 'display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0;'

  for (const [key, label] of Object.entries(EXIF_LABELS)) {
    if (!(key in exif) || exif[key] === null || exif[key] === undefined) continue
    const formatted = formatExifValue(key, exif[key])
    if (!formatted) continue
    const dt = document.createElement('dt')
    dt.style.cssText = 'color:var(--text-dim);white-space:nowrap;'
    dt.textContent = label
    const dd = document.createElement('dd')
    dd.style.cssText = 'margin:0;color:var(--text-main);word-break:break-all;'
    dd.textContent = formatted
    dl.append(dt, dd)
  }

  if (!dl.children.length) {
    exifDialogBody.textContent = 'No recognised EXIF fields found.'
  } else {
    exifDialogBody.appendChild(dl)
  }
  exifDialog.showModal()
})

exifDialogClose.addEventListener('click', () => exifDialog.close())
exifDialog.addEventListener('click', (e) => {
  if (e.target === exifDialog) exifDialog.close()
})

// ── Settings ────────────────────────────────────────────────────────────────
const saveSettings = () => {
  // A worker frame shares localStorage but runs a model of its parent's choosing.
  if (OVERNIGHT_CHILD) return
  const s =
    modelStandIn?.used === state.settings.modelVariant
      ? { ...state.settings, modelVariant: modelStandIn.saved }
      : state.settings
  localStorage.setItem('mc_settings', JSON.stringify(s))
}

// Sync UI with initial settings
dom.densitySlider.value = state.settings.density.toString()
dom.drawBudgetSlider.value = state.settings.drawBudget.toString()
dom.searchPercentSlider.value = state.settings.searchPercent.toString()
dom.searchPercentLabel.textContent = state.settings.searchPercent.toString()
dom.slideshowSecondsSlider.value = state.settings.slideshowSeconds.toString()
dom.slideshowSecondsLabel.textContent = state.settings.slideshowSeconds.toString()
dom.slideshowCutVideosToggle.checked = state.settings.slideshowCutVideos
dom.loopToggle.checked = state.settings.loopVideos
dom.enableSearchToggle.checked = state.settings.enableTextSearch

// Text search needs a text encoder sharing the image vectors' space. VLM tiers
// have none yet, so the toggle is disabled with a reason rather than left on
// to produce meaningless rankings.
function updateTextSearchAvailability() {
  const ok = supportsTextSearch(state.settings.modelVariant)
  dom.enableSearchToggle.disabled = !ok
  dom.enableSearchToggle.title = ok
    ? ''
    : `${state.settings.modelVariant} produces image-only vectors; text queries have nothing compatible to match against.`
  if (!ok) dom.enableSearchToggle.checked = false
}
updateTextSearchAvailability()
dom.batchSizeInput.value = state.settings.batchSize.toString()
dom.randomSampleSizeInput.value = state.settings.randomSampleSize.toString()
dom.viewerOnlyToggle.checked = state.settings.viewerOnly
dom.lazyCaptionToggle.checked = state.settings.enableLazyCaption
dom.doNotTrackToggle.checked = state.settings.doNotTrack
dom.customModelHostInput.value = state.settings.customModelHost
// Build the SmolVLM2 options from the tier table rather than hand-writing them
// in index.html, so a new tier row shows up in the UI automatically and the
// advertised download sizes cannot drift from the ones actually fetched.
{
  const group = document.getElementById('vlm-tier-group')
  if (group) {
    for (const tier of selectableVlmTiers()) {
      const opt = document.createElement('option')
      opt.value = tier.id
      const what = tier.queryPrefix
        ? 'embeddings + text search'
        : tier.visionOnly
          ? 'embeddings only'
          : 'captions + embeddings'
      opt.textContent = `${tier.id} (${what} · ${tier.downloadMB} MB)`
      opt.title = tier.queryPrefix
        ? 'Video-native, runs in a Web Worker. Photos, video and text search share one model.'
        : 'Video-native, runs in a Web Worker. No text search.'
      group.appendChild(opt)
    }
  }
}

// Order both lists by the benchmark (src/modelRanking.ts): the best model this
// device can run on top as the recommendation, the rest best to worst, and
// models it cannot run struck through. Labels are captured once, so a
// re-render never stacks markers.
const optionLabels = (select: HTMLSelectElement) =>
  new Map([...select.querySelectorAll('option')].map((o) => [o.value, { text: o.textContent ?? o.value, title: o.title }]))
const modelLabels = optionLabels(dom.modelSelect)
const renderModelOptions = (caps: DeviceCaps) => {
  const ranked = rankModels([...modelLabels.keys()] as ModelVariant[], caps)
  dom.modelSelect.replaceChildren(
    ...ranked.map(({ value, blocked, recommended }) => {
      const { text, title } = modelLabels.get(value)!
      const b = MODEL_BADGES[value]
      // Speed vs quality trade-off at a glance: benchmark ranks, 1 = best.
      const badges = b ? ` ⚡:${b.speed} Q:${b.quality}` : ''
      const opt = document.createElement('option')
      opt.value = value
      opt.title = b ? `${title}${title ? ' ' : ''}Benchmark rank: ⚡ speed ${b.speed}, Q quality ${b.quality} (1 = best).` : title
      if (blocked) {
        opt.textContent = `${strike(text)}${badges} — ${blocked}`
        opt.disabled = true
        opt.title = `Can't run on this device: ${blocked}.`
      } else {
        opt.textContent = recommended ? `★ ${text}${badges} — recommended for this device` : `${text}${badges}`
      }
      return opt
    })
  )
  dom.modelSelect.value = state.settings.modelVariant
  return ranked
}
// Until the checks below resolve, assume what navigator exposes.
renderModelOptions({ webgpu: 'gpu' in navigator, chromeAI: true })

if (dom.projectionSelect) {
  const projectionLabels = optionLabels(dom.projectionSelect)
  dom.projectionSelect.replaceChildren(
    ...rankProjections([...projectionLabels.keys()] as ProjectionMethod[]).map(({ value, recommended }) => {
      const opt = document.createElement('option')
      opt.value = value
      opt.textContent = `${recommended ? '★ ' : ''}${projectionLabels.get(value)!.text}${recommended ? ' — best on benchmark' : ''}`
      return opt
    })
  )
  dom.projectionSelect.value = state.settings.projectionMethod
}

// ── Setup wizard: check the device before any weights download ───────────────
interface DeviceDiag {
  adapter: GPUAdapter | null
  powerVR: boolean
  webnn: ('npu' | 'gpu' | 'cpu')[] // WebNN device types that create a context
  chromeAI: LanguageModelAvailability
  memoryGB?: number
}

/** A check that never answers (seen with the Chrome AI probe) must not stall the wizard. */
const within = <T,>(p: Promise<T>, fallback: T, ms = 3000) =>
  Promise.race([p.catch(() => fallback), new Promise<T>((r) => setTimeout(() => r(fallback), ms))])

/** Cheap capability checks: no model files are fetched. */
async function diagnoseDevice(): Promise<DeviceDiag> {
  const ml = (navigator as Navigator & { ml?: { createContext: (o: { deviceType: string }) => Promise<unknown> } }).ml
  const [adapter, chromeAI, ...contexts] = await Promise.all([
    within(navigator.gpu?.requestAdapter() ?? Promise.resolve(null), null),
    within(getChromeAIAvailability(), 'unavailable' as LanguageModelAvailability),
    ...(['npu', 'gpu', 'cpu'] as const).map((t) =>
      ml ? within(ml.createContext({ deviceType: t }).then(() => t), null) : Promise.resolve(null)
    )
  ])
  const webnn = contexts.filter((t): t is 'npu' | 'gpu' | 'cpu' => !!t)
  return { adapter, powerVR: isPowerVR(adapter), webnn, chromeAI, memoryGB: (navigator as { deviceMemory?: number }).deviceMemory }
}

const esc = (t: string) => t.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`)
const modelSizeLabel = (v: ModelVariant) => {
  const tier = getVlmTier(v)
  if (!tier) return modelLabels.get(v)?.text ?? v
  return `${tier.family === 'embeddinggemma2' ? 'EmbeddingGemma 2' : 'SmolVLM2 vision'} · ${tier.downloadMB} MB`
}

/** Why the wizard suggests (or can't run) a model, in plain words. */
function modelReason(v: ModelVariant, d: DeviceDiag, blocked: string | null, recommended: boolean): string {
  if (blocked) return `Can't run here: ${blocked}.`
  const b = MODEL_BADGES[v]
  const ranks = b ? `Benchmark: speed #${b.speed}, quality #${b.quality}.` : ''
  if (v === 'embeddinggemma2') {
    return USE_WEBNN
      ? `Runs on WebNN (${d.webnn.map((t) => t.toUpperCase()).join(' → ')}), the most direct AI hardware path this browser offers; about 3 s per photo on a Pixel 10. ${ranks}`
      : `Runs on your graphics chip (WebGPU); also answers text searches itself. ${ranks}`
  }
  if (v === 'nomic' && recommended) return `No graphics chip this model can use, so it runs on the CPU. ${ranks}`
  return ranks
}

/** What EmbeddingGemma 2 can run on here, with the model file each runs: the app's WebGPU path is q4, WebNN's fp16. */
const WEBGPU_DTYPE = 'q4' // the embeddinggemma2 tier's dtype in src/vlmTiers.ts
function accelerators(d: DeviceDiag): Candidate[] {
  const gpu = !!d.adapter && !d.powerVR && !d.adapter.info?.isFallbackAdapter
  return [
    ...(gpu ? [{ device: 'webgpu' as Accelerator, dtype: WEBGPU_DTYPE }] : []),
    ...d.webnn.map((t) => ({ device: `webnn-${t}` as Accelerator, dtype: 'fp16' }))
  ]
}

function renderChecks(d: DeviceDiag, fastest: { device: Accelerator; ms: number } | null = null) {
  const gpu = d.adapter
    ? `${d.adapter.info.vendor || 'GPU'} ${d.adapter.info.architecture}`.trim() +
      (d.powerVR ? ': too slow for AI on WebGPU' : d.adapter.info.isFallbackAdapter ? ': software only (slow)' : ' ✓')
    : 'not available to web pages'
  const lines = [
    `🎮 Graphics (WebGPU): ${gpu}`,
    `⚡ WebNN: ${d.webnn.length ? `${d.webnn.join(', ').toUpperCase()} ✓` : 'off (chrome://flags → "WebNN API")'}`,
    ...(fastest ? [`⏱ Fastest in your earlier runs: ${fastest.device}, ${(fastest.ms / 1000).toFixed(1)} s per photo`] : []),
    ...(d.memoryGB ? [`🧠 Memory: ${d.memoryGB} GB`] : []),
    ...(d.chromeAI !== 'unavailable' ? [`💬 Chrome AI: ${d.chromeAI}`] : [])
  ]
  document.getElementById('wizard-checks')!.innerHTML = lines.map((l) => `<li>${esc(l)}</li>`).join('')
}

/**
 * Runs before any model weights download: check the device, rank the models
 * for it (src/modelRanking.ts), and pre-select the one most likely to work.
 * A first visit waits for a tap on the pick (the download can be 880 MB);
 * a saved choice loads right away, through loadModel's usual fallbacks.
 */
async function runSetupWizard() {
  const d = await diagnoseDevice()
  chromeAIAvailability = d.chromeAI
  // Most native first: WebNN (NPU > GPU > CPU) whenever offered, then WebGPU, then CPU models.
  // A load the OS killed on this build would be killed again at every start; a new build tries once more.
  const killed = localStorage.getItem(WEBNN_LOADING_KEY) === __GIT_COMMIT__
  // Else the fastest of the accelerators here by the remembered runs (src/benchStore.ts), or, until two of them
  // have been measured, WebNN first with its most native device first.
  const fastest = killed ? null : fastestAccelerator(accelerators(d))
  useWebnn(killed ? [] : webnnOrder(d.webnn, fastest?.device))
  if (killed && d.webnn.length) {
    showToast(
      'The tab was killed while loading EmbeddingGemma 2 on WebNN (out of memory?), so WebNN is off for this version. It is tried again after the next update or after clearing site data.',
      'warn',
      15000
    )
  }
  const ranked = renderModelOptions({
    // "webgpu" here means EmbeddingGemma 2 has a path: WebNN, or a usable WebGPU.
    webgpu: (!!d.adapter && !d.powerVR) || USE_WEBNN,
    slowGpu: d.powerVR && !USE_WEBNN,
    chromeAI: d.chromeAI !== 'unavailable'
  })
  renderChecks(d, fastest)
  const current = document.getElementById('wizard-current')!
  if (savedSettings) {
    // A saved Chrome AI choice that cannot run falls back to the recommendation.
    // (EmbeddingGemma 2 without WebGPU is handled at load, for the session only.)
    const recommended = ranked.find((o) => o.recommended)
    if (state.settings.modelVariant === 'chrome-ai' && d.chromeAI === 'unavailable' && recommended) {
      state.settings.modelVariant = recommended.value
      dom.modelSelect.value = recommended.value
      saveSettings()
    }
    current.hidden = false
    autoLoadModel()
    return
  }

  const pick = ranked.find((o) => o.recommended)
  if (pick) state.settings.modelVariant = pick.value
  dom.modelSelect.value = state.settings.modelVariant
  const optionsEl = document.getElementById('wizard-options')!
  const go = document.getElementById('wizard-go') as HTMLButtonElement
  const options = ranked.filter((o) => o.value !== 'openai' && (o.value !== 'chrome-ai' || !o.blocked))
  let showAll = false
  const render = () => {
    // The pick and two alternatives; the rest on request, so Start stays in view.
    const shown = showAll ? options : options.filter((o, i) => i < 3 || o.value === state.settings.modelVariant)
    optionsEl.innerHTML = shown
      .map(({ value, blocked, recommended }) => {
        const label = `${recommended ? '★ ' : ''}${modelSizeLabel(value)}${recommended ? ' (recommended)' : ''}`
        return `<button type="button" role="radio" data-v="${value}" aria-checked="${value === state.settings.modelVariant}" ${blocked ? 'disabled' : ''}>${esc(label)}<small>${esc(modelReason(value, d, blocked, recommended))}</small></button>`
      })
      .join('') +
      (shown.length < options.length ? `<button type="button" data-more>More models (${options.length - shown.length})…</button>` : '')
    const tier = getVlmTier(state.settings.modelVariant)
    go.textContent = `Download and start${tier ? ` · ${tier.downloadMB} MB` : ''}`
  }
  optionsEl.onclick = (e) => {
    const btn = (e.target as HTMLElement).closest('button')
    if (!btn || btn.disabled) return
    if (btn.hasAttribute('data-more')) {
      showAll = true
      return render()
    }
    state.settings.modelVariant = btn.dataset.v as ModelVariant
    dom.modelSelect.value = state.settings.modelVariant
    render()
  }
  go.onclick = () => {
    saveSettings()
    updateChromeAIPromptVisibility()
    updateTextSearchAvailability()
    optionsEl.hidden = go.hidden = true
    current.hidden = false
    autoLoadModel()
  }
  render()
  go.hidden = false
  setStatus('Pick a model to start. Nothing has been downloaded yet.')
}

// Initialize Chrome AI prompt textarea with stored value
const updateChromeAIPromptVisibility = () => {
  const isChrome = state.settings.modelVariant === 'chrome-ai'
  dom.chromeAIPromptSetting.style.display = isChrome ? '' : 'none'
  dom.openaiSetting.style.display = state.settings.modelVariant === 'openai' ? '' : 'none'
}

// ── Remote endpoint settings ────────────────────────────────────────────────
if (dom.openaiBaseUrl) {
  dom.openaiBaseUrl.value = state.settings.openai.baseUrl
  dom.openaiModel.value = state.settings.openai.model
  dom.openaiRemember.checked = isOpenAIKeyRemembered()
  // Show that a key exists without ever putting the real one in the DOM, where
  // a screenshot or an extension could read it back out.
  if (getOpenAIKey()) dom.openaiKey.placeholder = '•••••••• (saved)'

  // `change`, not `input`: normalising on every keystroke would fight the user
  // mid-type, and each change here invalidates the cache namespace.
  dom.openaiBaseUrl.addEventListener('change', () => {
    const normalized = normalizeBaseUrl(dom.openaiBaseUrl.value)
    dom.openaiBaseUrl.value = normalized
    state.settings.openai.baseUrl = normalized
    saveSettings()
    reloadIfModelLoaded()
  })

  dom.openaiModel.addEventListener('change', () => {
    state.settings.openai.model = dom.openaiModel.value.trim()
    saveSettings()
    reloadIfModelLoaded()
  })

  // The form only exists to silence Chrome's "password field is not contained in a form"; Enter must not reload.
  dom.openaiKey.form?.addEventListener('submit', (e) => e.preventDefault())
  dom.openaiKey.addEventListener('change', () => {
    setOpenAIKey(dom.openaiKey.value.trim(), dom.openaiRemember.checked)
    // Never leave the key sitting in the input's value.
    dom.openaiKey.value = ''
    dom.openaiKey.placeholder = getOpenAIKey() ? '•••••••• (saved)' : 'Leave blank for local servers'
  })

  dom.openaiRemember.addEventListener('change', () => {
    // Move the existing key between localStorage and sessionStorage.
    const existing = getOpenAIKey()
    if (existing) setOpenAIKey(existing, dom.openaiRemember.checked)
  })

  dom.openaiFindModelsBtn.addEventListener('click', async () => {
    const baseUrl = normalizeBaseUrl(dom.openaiBaseUrl.value)
    if (!baseUrl) {
      dom.openaiModelsResult.textContent = 'Enter an endpoint URL first.'
      return
    }
    dom.openaiFindModelsBtn.disabled = true
    dom.openaiModelsResult.textContent = `Asking ${openaiHost(baseUrl)} what it offers…`
    try {
      // model is not known yet — that is the whole point of looking it up.
      const found = await lookupImageEmbeddingModels({ baseUrl, apiKey: getOpenAIKey(), model: '' })

      // Offer the confirmed image models when the catalogue says which are
      // which, and the raw list otherwise. Suggesting nothing would be worse
      // than suggesting an unfiltered list the user can still pick from.
      const offer = found.imageCapable.length > 0 ? found.imageCapable : found.all
      dom.openaiModelList.replaceChildren(
        ...offer.map((m) => {
          const opt = document.createElement('option')
          opt.value = m.id
          if (m.name && m.name !== m.id) opt.label = m.name
          return opt
        })
      )

      if (found.imageCapable.length > 0) {
        dom.openaiModelsResult.textContent = `✅ ${found.imageCapable.length} image-capable embedding model${found.imageCapable.length === 1 ? '' : 's'} — click the Model box to pick one.`
      } else if (!found.hasModalityMetadata) {
        // Absence of metadata is not absence of image models; say so plainly
        // rather than letting the user read an empty filter as a verdict.
        dom.openaiModelsResult.textContent = `${found.all.length} model${found.all.length === 1 ? '' : 's'} listed, but this endpoint doesn't report which accept images — pick one and use Test connection to check.`
      } else {
        dom.openaiModelsResult.textContent =
          'This endpoint lists no models that take image input and return embeddings.'
      }
    } catch (err) {
      dom.openaiModelsResult.textContent = `❌ ${describeOpenAIError(err)}`
    } finally {
      dom.openaiFindModelsBtn.disabled = false
    }
  })

  dom.openaiTestBtn.addEventListener('click', async () => {
    const baseUrl = normalizeBaseUrl(dom.openaiBaseUrl.value)
    const model = dom.openaiModel.value.trim()
    if (!baseUrl || !model) {
      dom.openaiTestResult.textContent = 'Enter both an endpoint URL and a model first.'
      return
    }
    dom.openaiTestBtn.disabled = true
    dom.openaiTestResult.textContent = `Contacting ${openaiHost(baseUrl)}…`
    try {
      const dim = await probeDimension({ baseUrl, apiKey: getOpenAIKey(), model })
      dom.openaiTestResult.textContent = `✅ Connected. ${model} returns ${dim}-dimensional vectors.`
    } catch (err) {
      dom.openaiTestResult.textContent = `❌ ${describeOpenAIError(err)}`
    } finally {
      dom.openaiTestBtn.disabled = false
    }
  })
}
if (dom.chromeAIPromptInput) {
  dom.chromeAIPromptInput.value = getChromeAIPrompt()
  dom.chromeAIPromptInput.addEventListener('input', () => {
    const val = dom.chromeAIPromptInput.value.trim()
    if (val) {
      localStorage.setItem(CHROME_AI_PROMPT_KEY, val)
    } else {
      localStorage.removeItem(CHROME_AI_PROMPT_KEY)
    }
  })
}
if (dom.chromeAIPromptReset) {
  dom.chromeAIPromptReset.addEventListener('click', () => {
    localStorage.removeItem(CHROME_AI_PROMPT_KEY)
    dom.chromeAIPromptInput.value = DEFAULT_DESCRIBE_PROMPT
  })
}
updateChromeAIPromptVisibility()

const updateSearchUI = () => {
  // In viewer mode, always disable search
  if (state.settings.viewerOnly) {
    dom.bottomPanel.style.display = 'none'
    dom.headerRecenterBtn.parentElement!.style.display = 'flex'
    dom.searchInput.disabled = true
    return
  }

  if (state.settings.enableTextSearch) {
    dom.bottomPanel.style.display = 'flex'
    dom.headerRecenterBtn.parentElement!.style.display = 'none'
  } else {
    dom.bottomPanel.style.display = 'none'
    dom.headerRecenterBtn.parentElement!.style.display = 'flex'
  }
  dom.searchInput.disabled = !state.settings.enableTextSearch || state.phase !== 'done'
  // Keep the failed chip in whichever strip is on screen.
  const strip = dom.bottomPanel.style.display === 'none' ? dom.headerRecenterBtn : dom.recenterBtn
  strip.parentElement!.prepend(dom.failedChip)
}
updateSearchUI()
refreshCacheSize()

dom.settingsBtn.addEventListener('click', () => {
  dom.settingsModal.showModal()
})

dom.settingsClose.addEventListener('click', () => {
  dom.settingsModal.close()
})

dom.settingsModal.addEventListener('click', (e) => {
  if (e.target === dom.settingsModal) dom.settingsModal.close()
})

dom.enableSearchToggle.addEventListener('change', async () => {
  // Prevent enabling search in viewer mode
  if (state.settings.viewerOnly && dom.enableSearchToggle.checked) {
    dom.enableSearchToggle.checked = false
    setStatus('Text search is not available in viewer mode.')
    return
  }

  state.settings.enableTextSearch = dom.enableSearchToggle.checked
  saveSettings()
  updateSearchUI()

  // If enabled and models are already loaded, load the text model now.
  // Remote mode embeds queries at the endpoint (see embedText), so pulling the
  // 134 MB local text model would be both wasted bandwidth and the wrong
  // embedding space to search the remote vectors with.
  if (
    state.settings.enableTextSearch &&
    state.phase !== 'idle' &&
    state.phase !== 'loading_model' &&
    !textExtractor &&
    state.settings.modelVariant !== 'openai' &&
    // Tiers with their own text encoder need no nomic text model.
    getVlmTier(state.settings.modelVariant)?.queryPrefix == null
  ) {
    // We duplicate the text loading logic here for dynamic loading
    dom.settingsModal.close()
    setStatus('Loading text model for search…')
    const TEXT_MODEL_SIZE_BYTES = 134 * 1024 * 1024
    const textLoaded = new Map<string, number>()

    const textProgressCb = (e: ProgressEvent) => {
      if (e.status === 'progress') {
        textLoaded.set(e.file, e.loaded ?? 0)
        const total = [...textLoaded.values()].reduce((a, b) => a + b, 0)
        const pct = Math.min(99, (total / TEXT_MODEL_SIZE_BYTES) * 100)
        setProgress(pct)
        setStatus(`Loading text model… ${pct.toFixed(0)}%`)
      }
    }

    const tryLoadText = (device: 'webgpu' | 'wasm') =>
      (pipeline as Pipeline)('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5', {
        device,
        dtype: 'fp32',
        progress_callback: textProgressCb
      }) as Promise<PipelineInstance>

    try {
      textExtractor = await tryLoadText(await webgpuOrWasm())
    } catch (gpuErr) {
      console.warn('Text model WebGPU failed, using wasm:', gpuErr)
      textLoaded.clear()
      textExtractor = await tryLoadText('wasm')
    }
    setProgress(100)
    setTimeout(() => setProgress(0), 500)
    setStatus('Text model loaded.')
  }
})

/**
 * Reload after a change that invalidates the cache namespace.
 *
 * Anything that alters what gets embedded — model, endpoint, dimension —
 * leaves `state.vectors` holding vectors from the old space while new ones
 * arrive in the new space. Search then returns plausible, meaningless
 * rankings with nothing to trip on, so a reload is the safe response.
 */
function reloadIfModelLoaded() {
  if (extractor || sapiens2Session || chromeAIManager || remoteConfig || vlmTier) {
    // The VLM worker holds a loaded model and a GPU device; drop it before the
    // reload so the old worker cannot outlive the page that owns it.
    disposeVlm()
    window.location.reload()
  }
}

dom.modelSelect.addEventListener('change', () => {
  state.settings.modelVariant = dom.modelSelect.value as ModelVariant
  saveSettings()
  updateChromeAIPromptVisibility()
  updateTextSearchAvailability()
  reloadIfModelLoaded()
})

dom.embedProfileSelect.value = state.settings.embedProfile
dom.embedProfileSelect.addEventListener('change', () => {
  state.settings.embedProfile = dom.embedProfileSelect.value as Settings['embedProfile']
  saveSettings()
  reloadIfModelLoaded() // the budget is read once at startup, like the model
})

dom.densitySlider.addEventListener('input', async () => {
  state.settings.density = parseFloat(dom.densitySlider.value)
  saveSettings()
  if (state.phase === 'done' && state.rawPoints && state.files.length) {
    setPointsAnimated(await spreadPointsAsync(state.rawPoints, state.settings.density))
    scheduleRender()
  }
})

dom.drawBudgetSlider.addEventListener('input', () => {
  state.settings.drawBudget = parseInt(dom.drawBudgetSlider.value)
  saveSettings()
  scheduleRender()
})

dom.searchPercentSlider.addEventListener('input', () => {
  state.settings.searchPercent = parseInt(dom.searchPercentSlider.value)
  dom.searchPercentLabel.textContent = dom.searchPercentSlider.value
  saveSettings()
  scheduleRender()
})

dom.slideshowSecondsSlider.addEventListener('input', () => {
  state.settings.slideshowSeconds = parseInt(dom.slideshowSecondsSlider.value)
  dom.slideshowSecondsLabel.textContent = dom.slideshowSecondsSlider.value
  saveSettings()
  // Apply to a running slideshow right away rather than on the next start.
  restartSlideshowTimer()
})

dom.slideshowCutVideosToggle.addEventListener('change', () => {
  state.settings.slideshowCutVideos = dom.slideshowCutVideosToggle.checked
  saveSettings()
  dom.modalVideo.loop = state.settings.loopVideos && !slideshowWaitsOnVideos()
})

dom.batchSizeInput.addEventListener('input', () => {
  const v = Math.max(1, parseInt(dom.batchSizeInput.value) || 1)
  state.settings.batchSize = v
  saveSettings()
})

const hasMemoryAPI = 'deviceMemory' in navigator || 'memory' in performance
dom.batchSizeAutoBtn.hidden = !hasMemoryAPI

if (hasMemoryAPI && !savedSettings) {
  // No saved preference — auto-detect on first load
  const optimal = computeOptimalBatchSize()
  state.settings.batchSize = optimal
  dom.batchSizeInput.value = optimal.toString()
  // Not saved: a saved mc_settings marks a returning visitor, who skips the setup wizard.
}

dom.batchSizeAutoBtn.addEventListener('click', () => {
  const optimal = computeOptimalBatchSize()
  state.settings.batchSize = optimal
  dom.batchSizeInput.value = optimal.toString()
  // Not saved: a saved mc_settings marks a returning visitor, who skips the setup wizard.
})

dom.randomSampleSizeInput.addEventListener('input', () => {
  const v = parseInt(dom.randomSampleSizeInput.value) || 0
  state.settings.randomSampleSize = Math.max(0, v)
  saveSettings()
})

dom.loopToggle.addEventListener('change', () => {
  state.settings.loopVideos = dom.loopToggle.checked
  saveSettings()
})

{
  const t = document.getElementById('find-faces-toggle') as HTMLInputElement
  t.checked = state.settings.findFaces
  t.addEventListener('change', () => {
    state.settings.findFaces = t.checked
    saveSettings()
    if (t.checked && state.files.length) void runFaces()
    else if (!t.checked) showPeople([])
  })
}
dom.lazyCaptionToggle.addEventListener('change', () => {
  state.settings.enableLazyCaption = dom.lazyCaptionToggle.checked
  saveSettings()
})

dom.doNotTrackToggle.addEventListener('change', () => {
  state.settings.doNotTrack = dom.doNotTrackToggle.checked
  saveSettings()
})

dom.customModelHostInput.addEventListener('change', () => {
  state.settings.customModelHost = normalizeHost(dom.customModelHostInput.value)
  dom.customModelHostInput.value = state.settings.customModelHost
  saveSettings()
  applyModelEnv()
})

dom.viewerOnlyToggle.addEventListener('change', async () => {
  state.settings.viewerOnly = dom.viewerOnlyToggle.checked
  saveSettings()
  updateSearchUI()
  updateDeviceBadge()

  if (state.settings.viewerOnly) {
    // Cancel any in-progress model download
    if (modelLoadAbort) {
      modelLoadAbort.abort()
      modelLoadAbort = null
      state.phase = 'idle'
    }
    if (state.phase === 'idle' || state.phase === 'loading_model') {
      dom.loadModelBtn.hidden = true
      dom.openBtn.disabled = false
      dom.openBtn.classList.add('primary')
      dom.demoBtn.disabled = false
      setStatus('Viewer mode active — open a folder to browse photos by date and folder.')
    }
  } else if (!state.settings.viewerOnly && state.phase === 'idle') {
    // Switching back to AI mode
    autoLoadModel()
  }

  // If changing mode with loaded data, need to reset and reprocess
  if (state.phase === 'done' && state.files.length > 0) {
    setStatus('Mode changed. Reset to apply changes, or open a new folder.')
  }
})

if (dom.projectionSelect) {
  dom.projectionSelect.addEventListener('change', async (e) => {
    const select = e.target as HTMLSelectElement
    state.settings.projectionMethod = select.value as ProjectionMethod
    saveSettings()

    if (state.vectors.length > 0 && state.phase === 'done') {
      try {
        state.phase = 'projecting'
        dom.recenterBtn.disabled = true
        if (dom.headerRecenterBtn) dom.headerRecenterBtn.disabled = true

        const nNeighbors = Math.max(2, Math.min(15, state.files.length - 1))
        const rawPoints = await runProjection(
          state.vectors,
          state.settings.projectionMethod,
          nNeighbors
        )
        state.rawPoints = rawPoints

        setStatus('Clustering…')
        const k = Math.min(8, Math.max(2, Math.ceil(Math.sqrt(state.files.length / 2))))
        state.clusters = await kmeansAsync(rawPoints, k)

        setStatus('Arranging layout…')
        setPointsAnimated(await spreadPointsAsync(rawPoints, state.settings.density))

        try {
          localStorage.setItem('po_projectedPoints', JSON.stringify(state.points))
          localStorage.setItem('po_clusters', JSON.stringify(Array.from(state.clusters)))
        } catch (_) {
          showToast("Session state couldn't be saved — browser storage is full.", 'warn')
        }

        fitCamera()
        scheduleRender()

        const finalMsg = `${state.files.length} media files · ${k} clusters`
        setStatus(`${state.files.length} media files — tap to view · ${k} clusters`)
        if (dom.statsEl) dom.statsEl.textContent = finalMsg
      } catch (err) {
        console.error('Reprojection error:', err)
        setStatus(`Reprojection failed: ${(err as Error).message}`)
      } finally {
        state.phase = 'done'
        dom.recenterBtn.disabled = false
        if (dom.headerRecenterBtn) dom.headerRecenterBtn.disabled = false
      }
    }
  })
}

dom.aboutBtn.addEventListener('click', () => {
  dom.aboutModal.showModal()
})

dom.aboutClose.addEventListener('click', () => {
  dom.aboutModal.close()
})

dom.aboutModal.addEventListener('click', (e) => {
  if (e.target === dom.aboutModal) dom.aboutModal.close()
})

dom.shortcutsBtn.addEventListener('click', () => dom.shortcutsModal.showModal())
dom.shortcutsClose.addEventListener('click', () => dom.shortcutsModal.close())
dom.shortcutsModal.addEventListener('click', (e) => {
  if (e.target === dom.shortcutsModal) dom.shortcutsModal.close()
})

window.addEventListener('resize', () => {
  resizeCanvas()
  if (state.phase === 'done' && state.points.length) scheduleRender()
})

// ── Init ─────────────────────────────────────────────────────────────────────
resizeCanvas()

// First-load card (replaces the about modal that used to open on a first visit).
showWelcome(!state.settings.viewerOnly)
// Until a model loads, name the one that will; then follow the toolbar badge.
const welcomeModel = document.getElementById('welcome-model')!
welcomeModel.textContent = `${state.settings.modelVariant}${USE_WEBNN ? ' · WebNN' : ''}`
if (deviceBadgeEl) {
  new MutationObserver(() => (welcomeModel.textContent = deviceBadgeEl.textContent)).observe(deviceBadgeEl, {
    childList: true,
    characterData: true,
    subtree: true
  })
}
for (const [card, tb] of [
  ['welcome-open', dom.openBtn],
  ['welcome-demo', dom.demoBtn]
] as const) {
  const btn = document.getElementById(card) as HTMLButtonElement
  btn.onclick = () => tb.click()
  // Mirror the toolbar button's enabled state, which the load flow toggles.
  const sync = () => (btn.disabled = tb.disabled)
  new MutationObserver(sync).observe(tb, { attributes: true, attributeFilter: ['disabled'] })
  sync()
}

const _savedKeys = localStorage.getItem('po_fileKeys')
if (_savedKeys) {
  try {
    const n = JSON.parse(_savedKeys).length
    dom.resumeBtn.hidden = false
    dom.resumeBtn.innerHTML = `🔄 <span class="btn-label">Resume last session (${n} images)</span>`
  } catch (_) {
    localStorage.clear()
    showToast('Previous session data was corrupted and has been cleared.', 'warn')
  }
}

dom.loadModelBtn.addEventListener('click', async () => {
  if (extractor || sapiens2Session || chromeAIManager || remoteConfig || vlmTier) return
  dom.loadModelBtn.disabled = true
  // Must run before any await: create() needs this click's user activation.
  if (state.settings.modelVariant === 'chrome-ai' && !chromeAIDownload && 'LanguageModel' in self) {
    chromeAIDownload = downloadChromeAI((f) => setStatus(`Downloading Gemini Nano… ${(f * 100).toFixed(0)}%`))
  }
  await loadModel()
})

dom.openBtn.addEventListener('click', async () => {
  if (window.showDirectoryPicker) {
    try {
      const dir = await window.showDirectoryPicker({ mode: 'read' })
      await run(dir)
    } catch (err) {
      if ((err as Error).name !== 'AbortError') setStatus(`Error: ${(err as Error).message}`)
    }
  } else {
    // Fallback for Safari/iOS
    dom.fileInput.click()
  }
})

dom.fileInput.addEventListener('change', async () => {
  const fileList = dom.fileInput.files
  if (!fileList || fileList.length === 0) return

  dom.openBtn.disabled = true
  setProgress(0)
  setStatus('Processing files…')

  const files: PhotoFile[] = []
  for (let i = 0; i < fileList.length; i++) {
    const file = fileList[i]
    const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
    if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
      files.push({
        name: file.webkitRelativePath || file.name,
        size: file.size,
        lastModified: file.lastModified,
        file,
        objectURL: null
      })
    }
  }

  const bundle = [...fileList].find((f) => f.name === BUNDLE_FILE && f.webkitRelativePath.split('/').length <= 2)
  const filtered = await openFilterDialog(files, bundle)
  if (!filtered) {
    setStatus('Cancelled.')
    dom.openBtn.disabled = false
    return
  }
  await processFiles(sampleFiles(filtered, state.settings.randomSampleSize))
})

dom.demoBtn.addEventListener('click', async () => {
  setProgress(0)
  setStatus('Fetching demo images…')
  dom.demoBtn.disabled = true

  try {
    const files = await loadDemoImages()
    if (files.length > 0) {
      await processFiles(files)
    } else {
      setStatus('Failed to load demo images')
      dom.demoBtn.disabled = false
    }
  } catch (e) {
    console.error('Demo load error:', e)
    setStatus('Error loading demo images')
    dom.demoBtn.disabled = false
  }
})

dom.resumeBtn.addEventListener('click', async () => {
  const savedPoints = JSON.parse(localStorage.getItem('po_projectedPoints') || 'null')
  const savedClusters = JSON.parse(localStorage.getItem('po_clusters') || 'null')
  const savedKeys = JSON.parse(localStorage.getItem('po_fileKeys') || 'null')
  if (!savedPoints || !savedKeys) {
    await dom.openBtn.click()
    return
  }

  try {
    let files: PhotoFile[] = []
    if (window.showDirectoryPicker) {
      const dir = await window.showDirectoryPicker({ mode: 'read' })
      setStatus('Matching files…')
      files = await collectImages(dir)
    } else {
      // Safari/iOS fallback
      setStatus('Please re-select the folder to resume.')
      const fileList = await new Promise<FileList | null>((resolve) => {
        const handler = () => {
          dom.fileInput.removeEventListener('change', handler)
          resolve(dom.fileInput.files)
        }
        dom.fileInput.addEventListener('change', handler)
        dom.fileInput.click()
      })

      if (!fileList || fileList.length === 0) return
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i]
        const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
        if (IMAGE_EXTS.has(ext) || VIDEO_EXTS.has(ext)) {
          files.push({
            name: file.webkitRelativePath || file.name,
            size: file.size,
            lastModified: file.lastModified,
            file,
            objectURL: null
          })
        }
      }
    }

    const keyToFile = new Map(files.map((f) => [`${f.name}:${f.size}:${f.lastModified}`, f]))
    const matched = (savedKeys as string[])
      .map((k) => keyToFile.get(k))
      .filter((f): f is PhotoFile => !!f)

    if (matched.length < (savedKeys as string[]).length * 0.8) {
      await processFiles(files)
      return
    }

    const wasViewerMode = localStorage.getItem('po_viewerMode') === 'true'

    state.files = matched
    clearFullTiles(true)
    randomHistory = []
    state.rawPoints = savedPoints
    state.points = savedPoints.slice(0, matched.length)
    state.clusters = savedClusters ? new Int32Array(savedClusters.slice(0, matched.length)) : null

    if (wasViewerMode) {
      // Viewer mode: no vectors needed
      state.vectors = []
      dom.searchInput.disabled = true
    } else {
      // AI mode: restore cached vectors for search + re-projection
      setStatus('Restoring embeddings…')
      // Was a copy of currentCachePrefix() that predated the remote backend, so
      // it had no 'openai' case and silently fell through to ''. Every resumed
      // read then missed the remote namespace and got zero-filled while the UI
      // reported a successful restore. Call the one function instead.
      const resumePrefix = currentCachePrefix()
      // Chunked reads so large folders show progress instead of a frozen bar,
      // and the main thread gets a breather between IDB transactions.
      const RESUME_CHUNK = 500
      const cachedVectors: (Float32Array | null)[] = []
      for (let c = 0; c < matched.length; c += RESUME_CHUNK) {
        const chunk = matched.slice(c, c + RESUME_CHUNK)
        const { cached, migrate } = await readCachedEmbeddings(chunk, resumePrefix, videoDraftPrefix(state.settings.modelVariant))
        if (migrate.length > 0) await cachePutBatch(migrate)
        cachedVectors.push(...cached)
        setStatus(
          `Restoring embeddings… ${Math.min(c + RESUME_CHUNK, matched.length)} / ${matched.length}`
        )
        setProgress(10 + (cachedVectors.length / matched.length) * 85)
        await yieldMain()
      }
      // Same width guard as the embedding loop: a restored vector of the wrong
      // width is from a different embedding space, and mixing the two yields
      // plausible, meaningless rankings with nothing to trip on.
      state.vectors = cachedVectors.map((v, i) =>
        v && v.length === activeEmbeddingDim ? v : failedVector(i, 'no cached embedding for this model')
      )
      dom.searchInput.disabled = false
    }

    thumbDecoding.clear() // in-flight decodes of the old list drop their results
    state.thumbnails = initThumbnails(matched)
    state.phase = 'done'
    updateFailedChip()
    resizeCanvas()
    fitCamera()
    scheduleRender()
    setProgress(100)
    showPeople([])
    void runFaces()
    const modeSuffix = wasViewerMode ? 'viewer mode' : 'restored'
    const finalMsg = `${matched.length} media files · ${modeSuffix}`
    setStatus(`${matched.length} media files — resumed from session`)
    if (dom.statsEl) dom.statsEl.textContent = finalMsg
    dom.recenterBtn.disabled = false
    dom.resetBtn.disabled = false
    dom.headerRecenterBtn.disabled = false
  } catch (err) {
    if ((err as Error).name !== 'AbortError') setStatus(`Error: ${(err as Error).message}`)
  }
})

// Auto-start model loading (only if not in viewer mode)
function autoLoadModel() {
  modelLoadAbort = new AbortController()
  ;(async () => {
    try {
      await loadModel(modelLoadAbort!.signal)
    } catch (err) {
      if ((err as Error).name === 'AbortError') return // user switched to viewer mode
      offerRetry(`Model failed: ${(err as Error).message}.`)
    } finally {
      modelLoadAbort = null
    }
  })()
}

// ── Overnight embedding ─────────────────────────────────────────────────────
/** Worker-frame side: load one model, embed the posted files, report, done. */
function runOvernightChild() {
  type TestEmbedder = (model: ModelVariant, file: PhotoFile, draft: boolean) => Promise<Float32Array | null>
  const fail = (message: string) => postToParent({ type: 'done', reason: 'error', message })
  window.addEventListener('error', (e) => fail(String(e.message)))
  window.addEventListener('unhandledrejection', (e) => fail(String((e.reason as Error)?.message ?? e.reason)))
  window.addEventListener('message', async (e) => {
    if (e.source !== window.parent || e.origin !== location.origin) return
    const msg = e.data as RunMsg
    if (msg?.type !== 'run') return
    try {
      useWebnn(msg.webnn ?? [])
      state.settings.viewerOnly = false
      state.settings.modelVariant = msg.model
      // e2e seam: stands in for the model so the controller can be tested without a GPU or a download.
      const test = (window as Window & { __mcTestEmbedder?: TestEmbedder }).__mcTestEmbedder
      if (test) {
        postToParent({ type: 'loaded' })
        const prefix = (msg.draft && videoDraftPrefix(msg.model)) || currentCachePrefix(msg.model)
        for (let i = 0; i < msg.files.length; i++) {
          const v = await test(msg.model, msg.files[i], !!msg.draft)
          if (v) await cachePutBatch([[`${prefix}${makeCacheKey(msg.files[i])}` as CacheKey, v]])
          postToParent({ type: 'progress', done: i + 1, total: msg.files.length })
        }
        postToParent({ type: 'done', reason: 'ok' })
        return
      }
      await loadModel()
      // loadModelOnce swaps in a CPU model when the GPU one can't run; that would embed under the wrong model.
      if (state.settings.modelVariant !== msg.model || state.phase === 'idle') {
        postToParent({ type: 'done', reason: 'unsupported', message: `${msg.model} can't run on this device.` })
        return
      }
      postToParent({ type: 'loaded' })
      await embedAll(msg.files, {
        headless: true,
        draft: msg.draft,
        // Recycle the frame well before the heap gets tight.
        minFreeRatio: 0.3,
        onProgress: (done, total) => postToParent({ type: 'progress', done, total })
      })
      postToParent({
        type: 'done',
        reason: lastEmbedStop,
        message: embedFailed.size ? `${embedFailed.size} in the last chunk could not be embedded: ${embedFailReason}` : undefined
      })
    } catch (err) {
      fail((err as Error)?.message ?? String(err))
    }
  })
  postToParent({ type: 'ready' })
}

const OVERNIGHT_JOB = 'mc_overnight_job'
const overnightBtn = document.getElementById('overnight-btn') as HTMLButtonElement
const overnightEl = document.getElementById('overnight') as HTMLDivElement
const $on = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

function savedOvernightJob(): { models: ModelVariant[]; sample?: boolean } | null {
  try {
    return JSON.parse(localStorage.getItem(OVERNIGHT_JOB) ?? 'null')
  } catch {
    return null
  }
}
if (savedOvernightJob()) {
  overnightBtn.querySelector('.btn-label')!.textContent = 'Resume overnight'
  overnightBtn.title = 'An overnight job did not finish. Pick the same folder to embed what is left.'
}

/** Models offered for overnight runs: what this device can run, minus ones that need a person. */
function overnightModelChoices() {
  const remoteReady = !!normalizeBaseUrl(state.settings.openai.baseUrl) && !!state.settings.openai.model
  return [...dom.modelSelect.querySelectorAll('option')]
    .filter((o) => o.value !== 'chrome-ai') // its download and sessions need a click
    .map((o) => ({
      id: o.value as ModelVariant,
      label: o.textContent ?? o.value,
      // Remote is configured right in the overnight dialog, so it is listed even before it is set up.
      disabled: o.value === 'openai' ? !remoteReady : o.disabled
    }))
}

/**
 * The same n files every time for the same folder, so a resumed overnight run continues the
 * sample it started instead of drawing a new one (sampleFiles is random per call).
 */
function stableSample(files: PhotoFile[], n: number): PhotoFile[] {
  if (n <= 0 || files.length <= n) return files
  const hash = (k: string) => {
    let h = 0x811c9dc5 // FNV-1a
    for (let i = 0; i < k.length; i++) h = Math.imul(h ^ k.charCodeAt(i), 0x01000193)
    return h >>> 0
  }
  return files
    .map((f) => [hash(makeCacheKey(f)), f] as const)
    .sort((a, b) => a[0] - b[0])
    .slice(0, n)
    .map(([, f]) => f)
}

function renderOvernight(st: OvernightStats) {
  const pct = Math.min(100, st.pct)
  $on('overnight-pct').textContent = `${pct.toFixed(1)}%`
  $on('overnight-bar').style.width = `${pct}%`
  $on('overnight-line').textContent =
    `${st.processed.toLocaleString()} / ${st.work.toLocaleString()} files · ` +
    `${st.rate.toFixed(2)} files/s · ETA ${st.finished ? '—' : formatDuration(st.etaSec)} · elapsed ${formatDuration(st.elapsedSec)}`
  $on('overnight-model-rows').replaceChildren(
    ...st.models.map((m) => {
      const row = document.createElement('div')
      row.className = 'on-model'
      const done = m.cachedAtStart + m.embedded + m.failed
      const mPct = m.total ? (100 * done) / m.total : 100
      const state =
        m.state === 'loading' ? `loading ${m.loadPct.toFixed(0)}%` : m.state === 'running' ? 'embedding' : m.state
      const head = document.createElement('div')
      head.className = 'on-model-head'
      head.textContent = `${m.label} — ${state} · ${mPct.toFixed(1)}%`
      const bar = document.createElement('div')
      bar.className = 'on-bar on-bar-small'
      bar.innerHTML = '<div></div>'
      ;(bar.firstChild as HTMLElement).style.width = `${mPct}%`
      const line = document.createElement('div')
      line.className = 'on-dim'
      const active = m.state === 'loading' || m.state === 'running'
      line.textContent =
        `embedded ${m.embedded.toLocaleString()} · already cached ${m.cachedAtStart.toLocaleString()} · ` +
        `failed ${m.failed.toLocaleString()} · left ${m.remaining.toLocaleString()}` +
        (active ? ` · ${m.rate.toFixed(2)} files/s · ETA ${formatDuration(m.etaSec)}` : '') +
        (m.message ? ` · ${m.message}` : '')
      row.append(head, bar, line)
      return row
    })
  )
  const mem = st.heapMB != null ? `${st.heapMB.toFixed(0)} / ${st.heapLimitMB!.toFixed(0)} MB` : 'n/a'
  $on('overnight-stats').textContent =
    `JS heap ${mem} · worker frames used ${st.recycles} · restarts ${st.restarts} · chunk ${st.chunkSize} files · ` +
    `screen ${overnightWakeLock ? 'kept awake' : 'may sleep — keep this tab in front'}`
}

let overnightWakeLock: WakeLockSentinel | null = null
let awakeWanted = 0 // overnight runs and interactive embeds that want the screen on
async function keepAwake() {
  if (overnightWakeLock) return
  try {
    const lock = (await navigator.wakeLock?.request('screen')) ?? null
    // Granted after the last holder finished, or a concurrent request won: drop this one.
    if (lock && (awakeWanted === 0 || overnightWakeLock)) return void lock.release().catch(() => {})
    overnightWakeLock = lock
    overnightWakeLock?.addEventListener('release', () => (overnightWakeLock = null))
  } catch {
    overnightWakeLock = null // denied or unsupported; the stats line says so
  }
}
// The browser drops the lock whenever the tab is hidden.
document.addEventListener('visibilitychange', () => {
  if (awakeWanted > 0 && document.visibilityState === 'visible' && !overnightWakeLock) void keepAwake()
})

/**
 * What an overnight run does for each picked model. A model with a quick video pass goes photos, then videos from
 * DRAFT_VIDEO_FRAMES frames, and only once every model has had those, videos again from all their frames
 * (the bonus pass). A video costs about a photo per frame, so the order puts a usable library first and the
 * improvement last, where Stop can cut it short. Other models get one pass over everything.
 */
function overnightPasses(picked: ModelVariant[], labels: Map<ModelVariant, string>): OvernightModel[] {
  const first: OvernightModel[] = []
  const bonus: OvernightModel[] = []
  for (const id of picked) {
    const label = labels.get(id) ?? id
    const prefix = currentCachePrefix(id)
    const draftPrefix = videoDraftPrefix(id)
    const tier = getVlmTier(id)
    if (!draftPrefix || !tier) {
      first.push({ id, label, prefix })
      continue
    }
    first.push(
      { id, label: `${label} · photos`, prefix, only: 'images' },
      {
        id,
        label: `${label} · videos, ${DRAFT_VIDEO_FRAMES} frames`,
        prefix: draftPrefix,
        only: 'videos',
        draft: true,
        doneUnder: [prefix]
      }
    )
    bonus.push({ id, label: `${label} · videos, ${tier.framesPerVideo} frames (bonus)`, prefix, only: 'videos', bonus: true })
  }
  return [...first, ...bonus]
}

async function startOvernightFlow() {
  if (!window.showDirectoryPicker) {
    showToast('Overnight embedding needs folder access, which this browser lacks. Use Chrome or Edge.', 'warn')
    return
  }
  let dir: DirectoryHandle
  try {
    dir = await window.showDirectoryPicker({ mode: 'read' })
  } catch (err) {
    if ((err as Error).name !== 'AbortError') setStatus(`Error: ${(err as Error).message}`)
    return
  }
  setStatus('Scanning folder…')
  const all = await openFilterDialog(await collectImages(dir, 0, ''), await bundleInFolder(dir))
  if (!all) {
    setStatus('Cancelled.')
    return
  }
  let files: PhotoFile[] = all
  setStatus(`${files.length} files selected for overnight embedding.`)

  const keys = await cacheKeys().catch(() => [] as string[])
  const job = savedOvernightJob()
  const checked = new Set<string>(job?.models ?? [state.settings.modelVariant])
  const list = $on('overnight-models')
  const sampleBox = $on<HTMLInputElement>('overnight-sample')
  const saveBox = $on<HTMLInputElement>('overnight-save-bundle')
  const sampleN = state.settings.randomSampleSize
  $on('overnight-sample-row').hidden = !(sampleN > 0 && files.length > sampleN)
  $on('overnight-sample-label').textContent =
    `Only a sample of ${sampleN.toLocaleString()} of the ${files.length.toLocaleString()} files (the Random sample setting; the same files each time, so Resume continues it)`
  sampleBox.checked = job?.sample ?? true
  const chosen = () => (sampleBox.checked && !$on('overnight-sample-row').hidden ? stableSample(all, sampleN) : all)

  // The remote endpoint form lives in Settings and only shows for the Remote model; borrow it here.
  const remoteForm = dom.openaiSetting
  const remoteHome = [remoteForm.parentElement!, remoteForm.nextSibling] as const
  $on('overnight-remote').append(remoteForm)
  remoteForm.style.display = ''
  const remoteToggle = $on<HTMLDetailsElement>('overnight-remote')
  remoteToggle.hidden = !overnightModelChoices().some((m) => m.id === 'openai')
  const giveBackRemoteForm = () => {
    remoteHome[0].insertBefore(remoteForm, remoteHome[1])
    updateChromeAIPromptVisibility()
  }

  const render = () => {
    for (const b of list.querySelectorAll<HTMLInputElement>('input')) b.checked ? checked.add(b.value) : checked.delete(b.value)
    const sel = chosen()
    list.replaceChildren(
      ...overnightModelChoices().map((m) => {
        const done = buildEmbeddedSets(keys, currentCachePrefix(m.id)).current
        const todo = sel.filter((f) => !done.has(fileKey(f))).length
        const label = document.createElement('label')
        label.className = 'on-pick'
        const box = document.createElement('input')
        box.type = 'checkbox'
        box.value = m.id
        box.disabled = m.disabled
        box.checked = !m.disabled && checked.has(m.id)
        label.append(box, ` ${m.label} `)
        const n = document.createElement('span')
        n.className = 'on-dim'
        n.textContent =
          m.id === 'openai' && m.disabled
            ? '— set the endpoint URL and model below first'
            : todo === 0
              ? '— all embedded'
              : `— ${todo.toLocaleString()} of ${sel.length.toLocaleString()} to embed`
        label.append(n)
        return label
      })
    )
  }
  render()
  $on('overnight-folder').textContent = `${dir.name} · ${files.length.toLocaleString()} files after filters`
  $on('overnight-setup').hidden = false
  $on('overnight-run').hidden = true
  overnightEl.hidden = false

  const picked = await new Promise<ModelVariant[] | null>((resolve) => {
    const ac = new AbortController()
    sampleBox.addEventListener('change', render, { signal: ac.signal })
    remoteForm.addEventListener('change', render, { signal: ac.signal })
    $on('overnight-start').addEventListener(
      'click',
      async () => {
        render()
        const ids = [...list.querySelectorAll<HTMLInputElement>('input:checked')].map((b) => b.value as ModelVariant)
        if (ids.length === 0) return showToast('Pick at least one model.', 'warn')
        // Write access is asked for now: at the end of the run there is no click to ask with.
        if (saveBox.checked) {
          const h = dir as DirectoryHandle & { requestPermission?: (d: { mode: 'readwrite' }) => Promise<PermissionState> }
          if ((await h.requestPermission?.({ mode: 'readwrite' }).catch(() => 'denied' as const)) !== 'granted') {
            return showToast('No write access to the folder; untick "save into this folder" to run without it.', 'warn')
          }
        }
        // Ask now, while someone is here: the worker frames cannot show the consent dialog.
        if (ids.includes('openai') && !(await showRemoteConsentModal(normalizeBaseUrl(state.settings.openai.baseUrl)))) return
        ac.abort()
        resolve(ids)
      },
      { signal: ac.signal }
    )
    $on('overnight-cancel').addEventListener(
      'click',
      () => {
        ac.abort()
        resolve(null)
      },
      { signal: ac.signal }
    )
  })
  giveBackRemoteForm()
  if (!picked) {
    overnightEl.hidden = true
    return
  }
  const sample = sampleBox.checked
  files = chosen()

  localStorage.setItem(OVERNIGHT_JOB, JSON.stringify({ models: picked, sample }))
  $on('overnight-setup').hidden = true
  $on('overnight-run').hidden = false
  const stopBtn = $on<HTMLButtonElement>('overnight-stop')
  const closeBtn = $on<HTMLButtonElement>('overnight-close')
  stopBtn.hidden = false
  stopBtn.disabled = false
  closeBtn.hidden = true
  const ac = new AbortController()
  const onStop = () => {
    stopBtn.disabled = true
    stopBtn.textContent = 'Stopping…'
    ac.abort()
  }
  stopBtn.addEventListener('click', onStop)
  awakeWanted++
  await keepAwake()
  // Unpersisted storage can be evicted under pressure overnight, and every fresh
  // worker frame reloads the model from the browser cache, so ask to keep it.
  // Not awaited: some browsers prompt for it, and nobody may be there to answer.
  void navigator.storage?.persist?.().catch(() => false)
  const labels = new Map(overnightModelChoices().map((m) => [m.id, m.label]))
  const test = window as Window & { __mcOvernightTest?: { chunkSize?: number; stallMs?: number; loadStallMs?: number } }
  let final: OvernightStats | null = null
  try {
    final = await runOvernight({
      files,
      models: overnightPasses(picked, labels),
      webnn: WEBNN_DEVICES,
      cacheKeys,
      cacheHasBatch: (k) => cacheHasBatch(k as CacheKey[]),
      onUpdate: renderOvernight,
      signal: ac.signal,
      chunkSize: test.__mcOvernightTest?.chunkSize,
      stallMs: test.__mcOvernightTest?.stallMs,
      loadStallMs: test.__mcOvernightTest?.loadStallMs
    })
  } catch (err) {
    showToast(`Overnight embedding stopped: ${(err as Error).message}`, 'error', 15000)
  } finally {
    if (--awakeWanted === 0) await overnightWakeLock?.release().catch(() => {})
    stopBtn.removeEventListener('click', onStop)
    stopBtn.textContent = 'Stop'
    stopBtn.hidden = true
    closeBtn.hidden = false
  }
  // Kept only while work is left, so the button offers to resume it.
  const unfinished = !final || final.models.some((m) => m.state === 'stopped' || m.remaining > 0)
  if (unfinished) localStorage.setItem(OVERNIGHT_JOB, JSON.stringify({ models: picked, sample }))
  else localStorage.removeItem(OVERNIGHT_JOB)
  if (final && saveBox.checked) {
    // Also after Stop: whatever finished is worth carrying to another device.
    await saveBundleToFolder(dir, files).then(
      (n) => showToast(`Saved ${n.toLocaleString()} embeddings into ${dir.name}/${BUNDLE_FILE}.`, 'info', 8000),
      (err) => showToast(`Saving ${BUNDLE_FILE} into the folder failed: ${(err as Error).message}`, 'warn', 12000)
    )
  }
  // Stopping the bonus pass is how a run normally ends, so the layout is still worth saving then.
  if (final && (!ac.signal.aborted || final.models.every((m) => m.bonus || m.state === 'done'))) {
    await projectOvernightResult(files, final).catch((err) =>
      showToast(`Saving the overnight layout failed: ${(err as Error).message}`, 'warn', 12000)
    )
  }
  overnightBtn.querySelector('.btn-label')!.textContent = unfinished ? 'Resume overnight' : 'Overnight'
  refreshCacheSize()
}

/**
 * Projects one finished overnight model and writes the slot the Resume button
 * reads, so the morning view opens without re-projecting. The model selected
 * in Settings wins if it finished; otherwise the first finished one.
 * shortcut: one saved projection (the Resume slot), not one per model; store per model in IndexedDB if switching models after an overnight run becomes common.
 */
async function projectOvernightResult(files: PhotoFile[], final: OvernightStats) {
  // A model is ready once its photos and quick video pass are done; the bonus pass only improves vectors already there.
  const done = [...new Set(final.models.map((m) => m.id))].filter((id) =>
    final.models.every((m) => m.id !== id || m.bonus || m.state === 'done')
  )
  const model = done.includes(state.settings.modelVariant) ? state.settings.modelVariant : done[0]
  if (!model) return
  const line = $on('overnight-line')
  const prefix = currentCachePrefix(model)
  const kept: PhotoFile[] = []
  const vectors: Float32Array[] = []
  const CHUNK = 500
  for (let c = 0; c < files.length; c += CHUNK) {
    const chunk = files.slice(c, c + CHUNK)
    const { cached, migrate } = await readCachedEmbeddings(chunk, prefix, videoDraftPrefix(model))
    if (migrate.length > 0) await cachePutBatch(migrate)
    cached.forEach((v, i) => {
      if (v) (kept.push(chunk[i]), vectors.push(v))
    })
    line.textContent = `Reading ${model} embeddings… ${Math.min(c + CHUNK, files.length).toLocaleString()} / ${files.length.toLocaleString()}`
    await yieldMain()
  }
  if (vectors.length < 3) {
    line.textContent = `Too few ${model} embeddings (${vectors.length}) to lay out; open the folder to embed the rest.`
    return
  }
  line.textContent = `Projecting ${vectors.length.toLocaleString()} files with ${state.settings.projectionMethod} (${model})…`
  await yieldMain()
  const nNeighbors = Math.max(2, Math.min(15, vectors.length - 1))
  const raw = await runProjection(vectors, state.settings.projectionMethod, nNeighbors, { silent: true })
  const k = Math.min(8, Math.max(2, Math.ceil(Math.sqrt(vectors.length / 2))))
  const clusters = await kmeansAsync(raw, k, 60)
  const points = await spreadPointsAsync(raw, state.settings.density)
  try {
    localStorage.setItem('po_fileKeys', JSON.stringify(kept.map((f) => `${f.name}:${f.size}:${f.lastModified}`)))
    localStorage.setItem('po_projectedPoints', JSON.stringify(points))
    localStorage.setItem('po_clusters', JSON.stringify(Array.from(clusters)))
    localStorage.removeItem('po_viewerMode')
  } catch {
    line.textContent = 'Projection done, but browser storage is full — it could not be saved for Resume.'
    return
  }
  // Resume reads vectors under the selected model's namespace.
  if (state.settings.modelVariant !== model) {
    state.settings.modelVariant = model
    dom.modelSelect.value = model
    saveSettings()
  }
  line.textContent = `Saved a ${model} layout of ${kept.length.toLocaleString()} files — reload and press Resume, then pick this folder.`
}

// ── Saved-embeddings controls (Settings) ────────────────────────────────────
{
  const out = $on('bundle-result')
  const guard = (btn: string, fn: () => Promise<string>) =>
    $on<HTMLButtonElement>(btn).addEventListener('click', async (e) => {
      const b = e.currentTarget as HTMLButtonElement
      b.disabled = true
      out.textContent = 'Working…'
      try {
        out.textContent = await fn()
      } catch (err) {
        if ((err as Error).name !== 'AbortError') out.textContent = `❌ ${(err as Error).message}`
        else out.textContent = ''
      } finally {
        b.disabled = false
      }
    })
  guard('bundle-export', async () => {
    const { blob, count } = await bundleBlob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = BUNDLE_FILE
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 60_000)
    return `✅ Exported ${count.toLocaleString()} embeddings (${(blob.size / 1048576).toFixed(1)} MB).`
  })
  const fileInput = $on<HTMLInputElement>('bundle-file')
  fileInput.addEventListener('change', async () => {
    const f = fileInput.files?.[0]
    fileInput.value = ''
    if (!f) return
    try {
      out.textContent = `✅ Imported ${(await importBundle(await f.arrayBuffer())).toLocaleString()} embeddings.`
    } catch (err) {
      out.textContent = `❌ ${(err as Error).message}`
    }
  })
  $on('bundle-import').addEventListener('click', () => fileInput.click())
  guard('bundle-folder', async () => {
    if (!window.showDirectoryPicker) throw new Error('This browser cannot write into folders. Use Export file instead.')
    const dir = await window.showDirectoryPicker({ mode: 'readwrite' })
    setStatus('Scanning folder…')
    const n = await saveBundleToFolder(dir, await collectImages(dir))
    return `✅ Saved ${n.toLocaleString()} embeddings of this folder's files into ${dir.name}/${BUNDLE_FILE}.`
  })
}

overnightBtn.addEventListener('click', () => void startOvernightFlow())
$on('overnight-close').addEventListener('click', () => {
  overnightEl.hidden = true
})

if (OVERNIGHT_CHILD) {
  runOvernightChild()
} else if (!state.settings.viewerOnly) {
  void runSetupWizard()
} else {
  // Viewer mode: update UI to reflect no AI needed
  dom.loadModelBtn.hidden = true
  dom.openBtn.disabled = false
  dom.openBtn.classList.add('primary')
  dom.demoBtn.disabled = false
  setStatus('Viewer mode active — open a folder to browse photos by date and folder.')
  updateDeviceBadge()
}

// ── Debug overlay (press ` to toggle) ────────────────────────────────────────
const debugOverlay = document.getElementById('debug-overlay') as HTMLDivElement

function formatBytes(b: number) {
  if (b >= 1073741824) return `${(b / 1073741824).toFixed(1)} GB`
  if (b >= 1048576) return `${(b / 1048576).toFixed(0)} MB`
  return `${(b / 1024).toFixed(0)} KB`
}

function buildDebugInfo(): string {
  const perfMem = (
    performance as Performance & {
      memory?: {
        jsHeapSizeLimit: number
        usedJSHeapSize: number
        totalJSHeapSize: number
      }
    }
  ).memory
  const devMem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory
  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu

  const lines: string[] = [
    `── state ───────────────────`,
    `phase:       ${state.phase}`,
    `files:       ${state.files.length}`,
    `viewerOnly:  ${state.settings.viewerOnly ? 'YES' : 'no'}`,
    `vectors:     ${state.vectors.length}`,
    `points:      ${state.points.length}`,
    `searchRes:   ${state.searchResults?.length ?? 'none'}`,
    `thumbs:      ${state.thumbnails.filter(Boolean).length} / ${state.thumbnails.length} decoded`,
    `thumbDecode: ${thumbDecoding.size} in-flight`,
    ``,
    `── settings ────────────────`,
    `batchSize:   ${state.settings.batchSize}`,
    `projection:  ${state.settings.projectionMethod}`,
    `textSearch:  ${state.settings.enableTextSearch}`,
    `density:     ${state.settings.density}`,
    ``,
    `── models ──────────────────`,
    `vision:      ${extractor ? `loaded (${modelDevice ?? '?'})` : sapiens2Session ? `sapiens2 (${modelDevice ?? '?'})` : chromeAIManager ? 'chrome-ai (built-in)' : 'none'}`,
    `text:        ${textExtractor ? 'loaded' : 'none'}`,
    ``,
    `── memory ──────────────────`,
    `deviceMemory: ${devMem != null ? devMem + ' GB' : 'unavailable'}`,
    `webgpu:      ${gpu ? 'available' : 'unavailable'}`
  ]

  if (perfMem) {
    lines.push(
      `heapUsed:    ${formatBytes(perfMem.usedJSHeapSize)}`,
      `heapTotal:   ${formatBytes(perfMem.totalJSHeapSize)}`,
      `heapLimit:   ${formatBytes(perfMem.jsHeapSizeLimit)}`
    )
  }

  lines.push(
    ``,
    `── camera ──────────────────`,
    `x: ${camera.x.toFixed(1)}  y: ${camera.y.toFixed(1)}  scale: ${camera.scale.toFixed(3)}`,
    ``,
    `[press \` to close]`
  )

  return lines.join('\n')
}

function refreshDebugOverlay() {
  if (debugOverlay.style.display === 'none') return
  // Update text node after the copy button (first child)
  const btn = debugOverlay.firstElementChild
  debugOverlay.textContent = buildDebugInfo()
  if (btn) debugOverlay.insertBefore(btn, debugOverlay.firstChild)
}

function navigateModal(dir: 'left' | 'right' | 'up' | 'down') {
  if (state.activeFileIndex === null) return
  const nextIndex = getNextImageInDirection(state.activeFileIndex, state.points, dir, isHidden)
  if (nextIndex !== state.activeFileIndex) {
    openFileModal(nextIndex)
  }
}

function navigateCanvas(dir: 'left' | 'right' | 'up' | 'down') {
  const pts = state.points
  if (!pts.length) return

  let startIndex = state.lastViewedIndex
  if (startIndex === null) {
    let minD = Infinity
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i][0] - camera.x
      const dy = pts[i][1] - camera.y
      const d = dx * dx + dy * dy
      if (d < minD) {
        minD = d
        startIndex = i
      }
    }
  }

  if (startIndex === null) return

  const nextIndex = getNextImageInDirection(startIndex, pts, dir, isHidden)
  if (nextIndex !== startIndex) {
    state.lastViewedIndex = nextIndex
    camera.x = pts[nextIndex][0]
    camera.y = pts[nextIndex][1]
    scheduleRender()
  }
}

/** The open photo, else the last one viewed, else the one nearest the view center. */
function currentOrCenterIndex(): number | null {
  let currentIndex = state.activeFileIndex ?? state.lastViewedIndex
  if (currentIndex === null) {
    let minD = Infinity
    for (let i = 0; i < state.points.length; i++) {
      if (isHidden(i)) continue
      const dx = state.points[i][0] - camera.x,
        dy = state.points[i][1] - camera.y
      const d = dx * dx + dy * dy
      if (d < minD) {
        minD = d
        currentIndex = i
      }
    }
  }
  return currentIndex
}

function navigateSequential(delta: 1 | -1) {
  if (!state.points.length) return
  const currentIndex = currentOrCenterIndex()
  if (currentIndex === null) return

  const len = state.files.length
  let pos = sequencePosition(currentIndex)
  if (pos === -1) return
  // Step past hidden (failed) files; give up after one full lap.
  for (let step = 0; step < len; step++) {
    pos = (pos + delta + len) % len
    const next = sequenceAt(pos)
    if (!isHidden(next)) {
      openFileModal(next)
      return
    }
  }
}

// Viewer mode already arranges state.files in visual order (folder then date),
// so an index is also its position — no sort/alloc. AI mode steps in
// chronological (lastModified) order.
function sequencePosition(index: number): number {
  return state.settings.viewerOnly ? index : getChronologicalOrder().indexOf(index)
}
function sequenceAt(pos: number): number {
  return state.settings.viewerOnly ? pos : getChronologicalOrder()[pos]
}

// Cache the sorted order and rebuild it only when the state.files reference
// changes, so repeated n/p presses don't re-sort on every step.
let chronologicalOrderCache: number[] | null = null
let chronologicalOrderFilesRef: PhotoFile[] | null = null
function getChronologicalOrder(): number[] {
  if (chronologicalOrderCache && chronologicalOrderFilesRef === state.files) {
    return chronologicalOrderCache
  }
  chronologicalOrderFilesRef = state.files
  chronologicalOrderCache = Array.from({ length: state.files.length }, (_, i) => i).sort(
    (a, b) => state.files[a].lastModified - state.files[b].lastModified
  )
  return chronologicalOrderCache
}

// Photos open before each R jump, so E can walk back through them.
let randomHistory: number[] = []

function openPreviousRandom() {
  const prev = randomHistory.pop()
  if (prev !== undefined && prev < state.files.length) openFileModal(prev)
}

function openRandomFile() {
  const len = state.points.length
  if (!len) return
  // Skip the photo already on screen (and hidden ones) so R always shows
  // something new; probe forward from a random start.
  const current = state.activeFileIndex ?? -1
  const start = Math.floor(Math.random() * len)
  for (let k = 0; k < len; k++) {
    const next = (start + k) % len
    if (next !== current && !isHidden(next)) {
      if (current !== -1) randomHistory.push(current)
      openFileModal(next)
      return
    }
  }
}

// ── Slideshow ────────────────────────────────────────────────────────────────
type SlideshowOrder = 'date' | 'random' | 'similar'
let slideshow: {
  order: SlideshowOrder
  timer: ReturnType<typeof setInterval>
  seen: Set<number>
} | null = null
// Order picked for this viewer session: ▶/⏸ pause and resume it without
// asking again. Cleared when the viewer closes, so the next ▶ asks afresh.
let slideshowOrder: SlideshowOrder | null = null

function openSlideshowPicker() {
  if (!state.points.length) return
  // Similarity walks the embeddings; viewer mode has none.
  dom.slideshowSimilarBtn.hidden = state.vectors.length === 0
  dom.slideshowModal.showModal()
}

function startSlideshow(order: SlideshowOrder) {
  stopSlideshow()
  const start = currentOrCenterIndex()
  if (start === null) return
  if (state.activeFileIndex === null) openFileModal(start)
  slideshowOrder = order
  slideshow = {
    order,
    seen: new Set([start]),
    timer: setInterval(slideshowStep, state.settings.slideshowSeconds * 1000)
  }
  dom.modalVideo.loop = state.settings.loopVideos && !slideshowWaitsOnVideos()
  dom.modalPlayBtn.textContent = '⏸'
  dom.modalPlayBtn.title = 'Pause slideshow (Space)'
}

function slideshowWaitsOnVideos(): boolean {
  return !!slideshow && !state.settings.slideshowCutVideos
}

function restartSlideshowTimer() {
  if (!slideshow) return
  clearInterval(slideshow.timer)
  slideshow.timer = setInterval(slideshowStep, state.settings.slideshowSeconds * 1000)
}

function stopSlideshow() {
  if (!slideshow) return
  clearInterval(slideshow.timer)
  slideshow = null
  dom.modalVideo.loop = state.settings.loopVideos
  dom.modalPlayBtn.textContent = '▶'
  dom.modalPlayBtn.title = 'Play slideshow (Space)'
}

/** ▶/⏸ and Space: pause, resume the picked order, or ask for one the first time. */
function toggleSlideshow() {
  if (slideshow) stopSlideshow()
  else if (slideshowOrder) startSlideshow(slideshowOrder)
  else openSlideshowPicker()
}

function slideshowStep() {
  if (!slideshow) return
  // Never advance behind a closed viewer, however it was closed.
  if (!dom.modal.open) {
    slideshowOrder = null
    return stopSlideshow()
  }
  // A video on screen holds the slide until its 'ended' listener advances.
  const v = dom.modalVideo
  if (slideshowWaitsOnVideos() && v.style.display !== 'none' && !v.ended && !v.error) return
  if (slideshow.order === 'date') return navigateSequential(1)
  if (slideshow.order === 'random') return openRandomFile()
  const cur = state.activeFileIndex
  if (cur === null) return stopSlideshow()
  slideshow.seen.add(cur)
  const seen = slideshow.seen
  let next = mostSimilarIndex(state.vectors, cur, (i) => seen.has(i) || isHidden(i))
  if (next === -1) {
    // Every photo shown once: start another lap from here.
    slideshow.seen = new Set([cur])
    next = mostSimilarIndex(state.vectors, cur, (i) => i === cur || isHidden(i))
  }
  if (next !== -1) openFileModal(next)
}

for (const btn of dom.slideshowModal.querySelectorAll<HTMLButtonElement>('button[data-order]')) {
  btn.addEventListener('click', () => {
    dom.slideshowModal.close()
    startSlideshow(btn.dataset.order as SlideshowOrder)
  })
}
dom.slideshowClose.addEventListener('click', () => dom.slideshowModal.close())
dom.slideshowModal.addEventListener('click', (e) => {
  if (e.target === dom.slideshowModal) dom.slideshowModal.close()
})
dom.modalPlayBtn.addEventListener('click', toggleSlideshow)
dom.modalImg.addEventListener('click', stopSlideshow)
dom.modalVideo.addEventListener('ended', () => {
  if (!slideshowWaitsOnVideos()) return
  slideshowStep()
  restartSlideshowTimer() // the next slide gets its full time
})

dom.modalRandomBtn.addEventListener('click', openRandomFile)
dom.modalPrevBtn.addEventListener('click', () => navigateSequential(-1))
dom.modalNextBtn.addEventListener('click', () => navigateSequential(1))

document.addEventListener('keydown', (e) => {
  // Chrome fires keydown with no key for autofill; e.key.toLowerCase() threw
  // (Bugsink: 13 events of "reading 'toLowerCase'").
  if (!e.key) return
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName ?? '')
  // Another dialog (settings, about, shortcuts) owns the keyboard while open.
  const otherDialog = document.querySelector('dialog[open]:not(#modal)')
  if (typing || otherDialog || e.ctrlKey || e.metaKey || e.altKey) return

  if (e.key === '?') {
    if (!dom.shortcutsModal.open) dom.shortcutsModal.showModal()
    return
  }

  if (e.key.toLowerCase() === 'r' && state.points.length > 0) {
    e.preventDefault()
    openRandomFile()
    return
  }

  if (e.key.toLowerCase() === 'e' && randomHistory.length > 0) {
    e.preventDefault()
    openPreviousRandom()
    return
  }

  // A button the user tabbed to already acts on Space/Enter; don't fire a
  // second action. Mouse-clicked buttons keep focus too but aren't
  // :focus-visible, so the shortcuts still work after clicking ›.
  const onButton = !!document.activeElement?.matches('button:focus-visible')

  if (e.key === ' ' && !onButton && state.points.length > 0) {
    e.preventDefault()
    toggleSlideshow()
    return
  }

  if (e.key === 'Enter' && !onButton && state.activeFileIndex === null && state.points.length > 0) {
    const i = currentOrCenterIndex()
    if (i !== null) openFileModal(i)
    return
  }

  if (e.key === '/' && !dom.searchInput.disabled && dom.searchInput.offsetParent !== null) {
    e.preventDefault()
    dom.searchInput.focus()
    return
  }

  if (e.key === '`') {
    const open = debugOverlay.style.display === 'none'
    debugOverlay.style.display = open ? 'block' : 'none'
    if (open) refreshDebugOverlay()
    return
  }

  if (state.activeFileIndex !== null && e.key === 'Escape') {
    closeModal()
    return
  }

  let dir: 'left' | 'right' | 'up' | 'down' | null = null
  if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') dir = 'left'
  else if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'd') dir = 'right'
  else if (e.key === 'ArrowUp' || e.key.toLowerCase() === 'w') dir = 'up'
  else if (e.key === 'ArrowDown' || e.key.toLowerCase() === 's') dir = 'down'

  if (dir) {
    if (state.activeFileIndex !== null) {
      e.preventDefault()
      navigateModal(dir)
    } else if (state.phase === 'done') {
      e.preventDefault()
      navigateCanvas(dir)
    }
  }

  // n/p keys for sequential next/previous (in datetime order)
  const key = e.key.toLowerCase()
  if ((key === 'n' || key === 'p') && state.points.length > 0) {
    e.preventDefault()
    navigateSequential(key === 'n' ? 1 : -1)
  }
})

dom.modalNavLeft.addEventListener('click', (e) => {
  e.stopPropagation()
  navigateModal('left')
})
dom.modalNavRight.addEventListener('click', (e) => {
  e.stopPropagation()
  navigateModal('right')
})
dom.modalNavUp.addEventListener('click', (e) => {
  e.stopPropagation()
  navigateModal('up')
})
dom.modalNavDown.addEventListener('click', (e) => {
  e.stopPropagation()
  navigateModal('down')
})

const debugCopyBtn = document.getElementById('debug-copy-btn') as HTMLButtonElement
debugCopyBtn.addEventListener('click', async () => {
  await navigator.clipboard.writeText(buildDebugInfo())
  debugCopyBtn.textContent = 'Copied!'
  setTimeout(() => {
    debugCopyBtn.textContent = 'Copy'
  }, 1500)
})

// Expose to console for deeper inspection
;(window as Window & { __debug?: unknown }).__debug = {
  updateURL,
  parseURLHash,
  get state() {
    return state
  },
  get camera() {
    return camera
  },
  get extractor() {
    return extractor
  },
  get textExtractor() {
    return textExtractor
  },
  get chromeAIManager() {
    return chromeAIManager
  },
  get thumbDecoding() {
    return thumbDecoding
  },
  get embedFailed() {
    return embedFailed
  },
  // Drive the real pipeline from a test page (device tests feed File objects).
  processFiles,
  collectImages,
  openFilterDialog,
  openFileModal,
  searchImages,
  extractVideoFrames,
  buildDebugInfo
}
