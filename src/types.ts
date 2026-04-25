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

/** Application settings */
export interface Settings {
  density: number;      // 1.0 = default, smaller = tighter, larger = sparse
  loopVideos: boolean;
  theme: 'dark' | 'light' | 'system';
  drawBudget: number;   // MAX_DRAW_PER_FRAME
  enableTextSearch: boolean;
  projectionMethod: ProjectionMethod;
  batchSize: number;    // GPU inference batch size (higher = faster, more memory)
  randomSampleSize: number; // 0 = load all; >0 = randomly sample n files when folder has more than n
  viewerOnly: boolean;  // Skip AI models, arrange by folder/date instead
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
  modal: HTMLDivElement;
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
  searchWrap: HTMLDivElement;
  searchInput: HTMLInputElement;
  searchClearBtn: HTMLButtonElement;
  fileInput: HTMLInputElement;
  aboutBtn: HTMLButtonElement;
  aboutModal: HTMLDivElement;
  aboutClose: HTMLButtonElement;
  statsEl: HTMLDivElement;
  settingsBtn: HTMLButtonElement;
  settingsModal: HTMLDivElement;
  settingsClose: HTMLButtonElement;
  densitySlider: HTMLInputElement;
  loopToggle: HTMLInputElement;
  themeSelect: HTMLSelectElement;
  drawBudgetSlider: HTMLInputElement;
  enableSearchToggle: HTMLInputElement;
  projectionSelect: HTMLSelectElement;
  viewerOnlyToggle: HTMLInputElement;
  batchSizeInput: HTMLInputElement;
  batchSizeAutoBtn: HTMLButtonElement;
  randomSampleSizeInput: HTMLInputElement;
  bottomPanel: HTMLDivElement;
  headerRecenterBtn: HTMLButtonElement;
  demoBtn: HTMLButtonElement;
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