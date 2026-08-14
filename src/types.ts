/**
 * Core type definitions for Photo Organizer
 */

/** Application phase states */
export type Phase = 'idle' | 'loading_model' | 'model_ready' | 'embedding' | 'projecting' | 'done'

/** File metadata and object URL */
export interface PhotoFile {
  name: string
  size: number
  lastModified: number
  file: File
  objectURL: string | null
  gps?: { latitude: number; longitude: number } | null // null = parsed but no data
  exifData?: Record<string, unknown> | null // null = parsed but no data
}

/** 2D point in world space */
export type Point = readonly [x: number, y: number]

/** Camera state for infinite canvas */
export interface Camera {
  x: number
  y: number
  scale: number
}

/** Supported projection methods */
export type ProjectionMethod =
  | 'UMAP'
  | 'TSNE'
  | 'PCA'
  | 'ISOMAP'
  | 'LLE'
  | 'MDS'
  | 'SAMMON'
  | 'TriMap'

/** Vision model variant */
export type ModelVariant =
  | 'nomic'
  | 'sapiens2-int8'
  | 'sapiens2-fp16'
  | 'sapiens2-fp32'
  | 'chrome-ai'
  | 'openai'
  | 'smolvlm2-vision'
  | 'smolvlm2-256m'
  | 'smolvlm2-500m'

/** ONNX weight quantisation, as named by Transformers.js `dtype`. */
export type VlmDtype = 'q4f16' | 'q4' | 'int8' | 'fp16' | 'fp32'

/**
 * One row of the SmolVLM2 tier table (`src/vlmTiers.ts`). See ADR-0001: the
 * tier is data, so adding a future sub-1B video model is a new row rather than
 * a new branch.
 */
export interface VlmTier {
  id: ModelVariant
  repo: string // HuggingFace repo, e.g. 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
  dtype: VlmDtype
  visionOnly: boolean // tier A: never download embed_tokens/decoder
  visionDim: number // width of the pooled vision vector
  framesPerVideo: number // frames sampled per video (1 = today's behaviour)
  cachePrefix: string // IndexedDB namespace, e.g. '@smolvlm2-vision/'
  downloadMB: number // approximate, for the settings UI
}

/** Request sent from the main thread to the VLM worker. */
export type VlmRequest =
  | { type: 'load'; id: number; tier: VlmTier; remoteHost: string }
  | { type: 'embed'; id: number; images: ImageBitmap[] }
  | { type: 'dispose'; id: number }

/** Response sent from the VLM worker back to the main thread. */
export type VlmResponse =
  | { type: 'progress'; id: number; loaded: number; total: number; file: string }
  | { type: 'loaded'; id: number; device: VlmDevice; visionDim: number }
  | { type: 'embedded'; id: number; vectors: Float32Array[] }
  | { type: 'error'; id: number; message: string; name: string }

/** Backend the worker actually got, after the webgpu→wasm fallback. */
export type VlmDevice = 'webgpu' | 'wasm'

/**
 * Remote OpenAI-compatible endpoint settings.
 *
 * The API key is deliberately NOT here — it lives under its own storage key,
 * reached only through `getOpenAIKey()` / `setOpenAIKey()` in
 * `src/openaiCompat.ts`. See ADR-0002.
 */
export interface OpenAISettings {
  baseUrl: string // Normalized API root, e.g. https://openrouter.ai/api/v1
  model: string // Model id sent as `model` in each request
  maxImageWidth: number // Thumbnails are downscaled to this before upload
}

/** Application settings */
export interface Settings {
  density: number // 1.0 = default, smaller = tighter, larger = sparse
  loopVideos: boolean
  drawBudget: number // MAX_DRAW_PER_FRAME
  enableTextSearch: boolean
  projectionMethod: ProjectionMethod
  batchSize: number // GPU inference batch size (higher = faster, more memory)
  randomSampleSize: number // 0 = load all; >0 = randomly sample n files when folder has more than n
  viewerOnly: boolean // Skip AI models, arrange by folder/date instead
  modelVariant: ModelVariant // Vision embedding model to use
  enableLazyCaption: boolean // Generate captions on modal open via Chrome AI (off by default)
  doNotTrack: boolean // Disable BugSink error reporting (default false)
  customModelHost: string // Alternative HuggingFace-compatible host (corporate proxy/mirror); '' = huggingface.co
  openai: OpenAISettings // Remote inference endpoint (modelVariant === 'openai')
}

/** Application state */
export interface AppState {
  phase: Phase
  files: PhotoFile[]
  vectors: Float32Array[]
  points: Point[]
  rawPoints: number[][] | null
  clusters: Int32Array | null
  thumbnails: (ImageBitmap | null)[]
  captions: (string | null)[]
  searchResults: Int32Array | null
  searchQuery: string
  searchScores: Float32Array | null
  fileKeys?: string[]
  settings: Settings
  activeFileIndex: number | null
  lastViewedIndex: number | null
  currentDirHandle: FileSystemDirectoryHandle | null
  currentBasePath: string // Track current folder path for navigation
}

/** DOM element references */
export interface DOMElements {
  loadModelBtn: HTMLButtonElement
  resumeBtn: HTMLButtonElement
  openBtn: HTMLButtonElement
  recenterBtn: HTMLButtonElement
  resetBtn: HTMLButtonElement
  progressBar: HTMLDivElement
  statusEl: HTMLDivElement
  canvas: HTMLCanvasElement
  modal: HTMLDialogElement
  modalClose: HTMLButtonElement
  modalNavLeft: HTMLButtonElement
  modalNavRight: HTMLButtonElement
  modalNavUp: HTMLButtonElement
  modalNavDown: HTMLButtonElement
  modalImg: HTMLImageElement
  modalVideo: HTMLVideoElement
  modalFooter: HTMLDivElement
  modalUp: HTMLSpanElement
  modalPath: HTMLDivElement
  modalFilename: HTMLDivElement
  modalDatetime: HTMLDivElement
  modalMeta: HTMLSpanElement
  modalGps: HTMLAnchorElement
  modalExifBtn: HTMLButtonElement
  modalPrevBtn: HTMLButtonElement
  modalNextBtn: HTMLButtonElement
  searchWrap: HTMLDivElement
  searchInput: HTMLInputElement
  searchClearBtn: HTMLButtonElement
  fileInput: HTMLInputElement
  aboutBtn: HTMLButtonElement
  aboutModal: HTMLDialogElement
  aboutClose: HTMLButtonElement
  statsEl: HTMLDivElement
  settingsBtn: HTMLButtonElement
  settingsModal: HTMLDialogElement
  settingsClose: HTMLButtonElement
  densitySlider: HTMLInputElement
  loopToggle: HTMLInputElement
  drawBudgetSlider: HTMLInputElement
  enableSearchToggle: HTMLInputElement
  projectionSelect: HTMLSelectElement
  viewerOnlyToggle: HTMLInputElement
  lazyCaptionToggle: HTMLInputElement
  doNotTrackToggle: HTMLInputElement
  batchSizeInput: HTMLInputElement
  batchSizeAutoBtn: HTMLButtonElement
  randomSampleSizeInput: HTMLInputElement
  bottomPanel: HTMLDivElement
  headerRecenterBtn: HTMLButtonElement
  demoBtn: HTMLButtonElement
  modelSelect: HTMLSelectElement
  modalCaption: HTMLDivElement
  chromeAIPromptInput: HTMLTextAreaElement
  chromeAIPromptReset: HTMLButtonElement
  chromeAIPromptSetting: HTMLDivElement
  customModelHostInput: HTMLInputElement
  openaiSetting: HTMLDivElement
  openaiBaseUrl: HTMLInputElement
  openaiKey: HTMLInputElement
  openaiRemember: HTMLInputElement
  openaiModel: HTMLInputElement
  openaiModelList: HTMLDataListElement
  openaiFindModelsBtn: HTMLButtonElement
  openaiModelsResult: HTMLDivElement
  openaiTestBtn: HTMLButtonElement
  openaiTestResult: HTMLDivElement
  openaiConsentModal: HTMLDialogElement
  openaiConsentHost: HTMLElement
  openaiConsentAccept: HTMLButtonElement
  openaiConsentCancel: HTMLButtonElement
  modelFallbackModal: HTMLDialogElement
  modelFallbackClose: HTMLButtonElement
  modelFallbackUrls: HTMLUListElement
  modelFallbackFile: HTMLInputElement
  modelFallbackFileHint: HTMLDivElement
  modelFallbackHost: HTMLInputElement
  modelFallbackCancel: HTMLButtonElement
  modelFallbackRetry: HTMLButtonElement
}

/** IndexedDB cache entry */
export type CacheKey = `${string}:${number}:${number}`

/** Projection algorithm interface */
export interface IProjection {
  fit(data: Float32Array[] | number[][]): Promise<number[][]>
}

/**
 * Transformers.js pipeline types (simplified)
 */
export interface Pipeline {
  (task: string, model: string, options?: PipelineOptions): Promise<PipelineInstance>
}

declare global {
  const __GIT_BRANCH__: string
  const __GIT_COMMIT__: string
  const __APP_VERSION__: string
  interface Window {
    showDirectoryPicker?: (options?: { mode: 'read' | 'readwrite' }) => Promise<DirectoryHandle>
  }
}

export interface PipelineOptions {
  device?: 'webgpu' | 'wasm' | 'cpu'
  dtype?: 'fp32' | 'fp16' | 'q8'
  progress_callback?: (progress: ProgressEvent) => void
  pooling?: 'mean' | 'cls' | 'max'
  normalize?: boolean
}

export interface PipelineInstance {
  (input: string | URL | Blob | object, options?: InferenceOptions): Promise<PipelineOutput>
}

export interface InferenceOptions extends Record<string, unknown> {
  pooling?: 'mean' | 'cls' | 'max'
}

export interface PipelineOutput extends Record<string, unknown> {
  dims: number[]
  data: Float32Array
  last_hidden_state?: PipelineOutput
  pooler_output?: PipelineOutput
}

export interface ProgressEvent {
  status: 'progress' | 'done' | 'initiate'
  file: string
  loaded?: number
  progress?: number
}

/** Directory handle for File System Access API */
export type DirectoryHandle = FileSystemDirectoryHandle

/** File handle for File System Access API */
export type FileSystemHandle = FileSystemFileHandle

/** Pointer event state */
export interface PointerState {
  x: number
  y: number
}

/** Pointer position in canvas coordinates */
export interface CanvasPointerPos {
  cx: number
  cy: number
}
