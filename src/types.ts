/**
 * Core type definitions for Photo Organizer
 */

/** Application phase states */
export type Phase = 'idle' | 'loading_model' | 'model_ready' | 'embedding' | 'projecting' | 'done';

/** File metadata and object URL */
export interface PhotoFile {
  name: string;
  size: number;
  lastModified: number;
  file: File;
  objectURL: string | null;
  gps?: { latitude: number; longitude: number } | null; // null = parsed but no data
  exifData?: Record<string, unknown> | null; // null = parsed but no data
}

/** 2D point in world space */
export type Point = readonly [x: number, y: number];

/** Camera state for infinite canvas */
export interface Camera {
  x: number;
  y: number;
  scale: number;
}

/** Supported projection methods */
export type ProjectionMethod = 'UMAP' | 'TSNE' | 'PCA' | 'ISOMAP' | 'LLE' | 'MDS' | 'SAMMON' | 'TriMap';

/** Vision model variant */
export type ModelVariant = 'nomic' | 'sapiens2-int8' | 'sapiens2-fp16' | 'sapiens2-fp32' | 'chrome-ai';

/** Application settings */
export interface Settings {
  density: number;      // 1.0 = default, smaller = tighter, larger = sparse
  loopVideos: boolean;
  drawBudget: number;   // MAX_DRAW_PER_FRAME
  enableTextSearch: boolean;
  projectionMethod: ProjectionMethod;
  batchSize: number;    // GPU inference batch size (higher = faster, more memory)
  randomSampleSize: number; // 0 = load all; >0 = randomly sample n files when folder has more than n
  viewerOnly: boolean;  // Skip AI models, arrange by folder/date instead
  modelVariant: ModelVariant; // Vision embedding model to use
  enableLazyCaption: boolean; // Generate captions on modal open via Chrome AI (off by default)
  doNotTrack: boolean;       // Disable BugSink error reporting (default false)
}

/** Application state */
export interface AppState {
  phase: Phase;
  files: PhotoFile[];
  vectors: Float64Array[];
  points: Point[];
  rawPoints: number[][] | null;
  clusters: Int32Array | null;
  thumbnails: (ImageBitmap | null)[];
  captions: (string | null)[];
  searchResults: Int32Array | null;
  searchQuery: string;
  searchScores: Float32Array | null;
  fileKeys?: string[];
  settings: Settings;
  hnsw?: any;
  activeFileIndex: number | null;
  lastViewedIndex: number | null;
  currentDirHandle: FileSystemDirectoryHandle | null;
  currentBasePath: string; // Track current folder path for navigation
}

/** DOM element references */
export interface DOMElements {
  loadModelBtn: HTMLButtonElement;
  resumeBtn: HTMLButtonElement;
  openBtn: HTMLButtonElement;
  recenterBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  progressBar: HTMLDivElement;
  statusEl: HTMLDivElement;
  canvas: HTMLCanvasElement;
  modal: HTMLDialogElement;
  modalClose: HTMLButtonElement;
  modalNavLeft: HTMLButtonElement;
  modalNavRight: HTMLButtonElement;
  modalNavUp: HTMLButtonElement;
  modalNavDown: HTMLButtonElement;
  modalImg: HTMLImageElement;
  modalVideo: HTMLVideoElement;
  modalFooter: HTMLDivElement;
  modalUp: HTMLSpanElement;
  modalPath: HTMLDivElement;
  modalFilename: HTMLDivElement;
  modalDatetime: HTMLDivElement;
  modalMeta: HTMLSpanElement;
  modalGps: HTMLAnchorElement;
  modalExifBtn: HTMLButtonElement;
  modalPrevBtn: HTMLButtonElement;
  modalNextBtn: HTMLButtonElement;
  searchWrap: HTMLDivElement;
  searchInput: HTMLInputElement;
  searchClearBtn: HTMLButtonElement;
  fileInput: HTMLInputElement;
  aboutBtn: HTMLButtonElement;
  aboutModal: HTMLDialogElement;
  aboutClose: HTMLButtonElement;
  statsEl: HTMLDivElement;
  settingsBtn: HTMLButtonElement;
  settingsModal: HTMLDialogElement;
  settingsClose: HTMLButtonElement;
  densitySlider: HTMLInputElement;
  loopToggle: HTMLInputElement;
  drawBudgetSlider: HTMLInputElement;
  enableSearchToggle: HTMLInputElement;
  projectionSelect: HTMLSelectElement;
  viewerOnlyToggle: HTMLInputElement;
  lazyCaptionToggle: HTMLInputElement;
  doNotTrackToggle: HTMLInputElement;
  batchSizeInput: HTMLInputElement;
  batchSizeAutoBtn: HTMLButtonElement;
  randomSampleSizeInput: HTMLInputElement;
  bottomPanel: HTMLDivElement;
  headerRecenterBtn: HTMLButtonElement;
  demoBtn: HTMLButtonElement;
  modelSelect: HTMLSelectElement;
  modalCaption: HTMLDivElement;
  chromeAIPromptInput: HTMLTextAreaElement;
  chromeAIPromptReset: HTMLButtonElement;
  chromeAIPromptSetting: HTMLDivElement;
}

/** IndexedDB cache entry */
export type CacheKey = `${string}:${number}:${number}`;

/** Projection algorithm interface */
export interface IProjection {
  fit(data: Float32Array[] | number[][]): Promise<number[][]>;
}

/**
 * Transformers.js pipeline types (simplified)
 */
export interface Pipeline {
  (
    task: string,
    model: string,
    options?: PipelineOptions
  ): Promise<PipelineInstance>;
}

declare global {
  const __GIT_BRANCH__: string;
  const __GIT_COMMIT__: string;
  const __APP_VERSION__: string;
  interface Window {
    showDirectoryPicker?: (options?: {
      mode: 'read' | 'readwrite';
    }) => Promise<DirectoryHandle>;
  }
}

export interface PipelineOptions {
  device?: 'webgpu' | 'wasm' | 'cpu';
  dtype?: 'fp32' | 'fp16' | 'q8';
  progress_callback?: (progress: ProgressEvent) => void;
  pooling?: 'mean' | 'cls' | 'max';
  normalize?: boolean;
}

export interface PipelineInstance {
  (input: string | URL | Blob | object, options?: InferenceOptions): Promise<PipelineOutput>;
}

export interface InferenceOptions extends Record<string, unknown> {
  pooling?: 'mean' | 'cls' | 'max';
}

export interface PipelineOutput extends Record<string, unknown> {
  dims: number[];
  data: Float32Array;
  last_hidden_state?: PipelineOutput;
  pooler_output?: PipelineOutput;
}

export interface ProgressEvent {
  status: 'progress' | 'done' | 'initiate';
  file: string;
  loaded?: number;
  progress?: number;
}

/** Directory handle for File System Access API */
export type DirectoryHandle = FileSystemDirectoryHandle;

/** File handle for File System Access API */
export type FileSystemHandle = FileSystemFileHandle;

/** Pointer event state */
export interface PointerState {
  x: number;
  y: number;
}

/** Pointer position in canvas coordinates */
export interface CanvasPointerPos {
  cx: number;
  cy: number;
}