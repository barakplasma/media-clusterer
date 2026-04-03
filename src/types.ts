/**
 * Core type definitions for Photo Organizer
 */

/** Application phase states */
export type Phase = 'idle' | 'loading_model' | 'model_ready' | 'embedding' | 'umap' | 'done';

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

/** Application state */
export interface AppState {
  phase: Phase;
  files: PhotoFile[];
  vectors: Float32Array[];
  points: Point[];
  clusters: Int32Array | null;
  thumbnails: (ImageBitmap | null)[];
  searchResults: Int32Array | null;
  searchQuery: string;
  searchScores: Float32Array | null;
  fileKeys?: string[];
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
  modalImg: HTMLImageElement;
  modalName: HTMLDivElement;
  searchInput: HTMLInputElement;
  searchClearBtn: HTMLButtonElement;
  fileInput: HTMLInputElement;
  aboutBtn: HTMLButtonElement;
  aboutModal: HTMLDivElement;
  aboutClose: HTMLButtonElement;
}

/** IndexedDB cache entry */
export type CacheKey = `${string}:${number}:${number}`;

/** UMAP library parameters */
export interface UMAPParameters {
  nComponents?: number;
  nNeighbors?: number;
  nEpochs?: number;
  minDist?: number;
  spread?: number;
  learningRate?: number;
  repulsionStrength?: number;
  negativeSampleRate?: number;
  random?: () => number;
}

/** UMAP library constructor */
export interface UMAPConstructor {
  new(options?: UMAPParameters): UMAPInstance;
}

/** UMAP library instance */
export interface UMAPInstance {
  fit(data: number[][] | Float32Array[]): number[][];
  fitAsync(
    data: number[][] | Float32Array[],
    callback?: (epoch: number) => void | boolean
  ): Promise<number[][]>;
  initializeFit(data: number[][] | Float32Array[]): number;
  step(): number;
  getEmbedding(): number[][];
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
  interface Window {
    UMAP?: UMAPConstructor | { UMAP: UMAPConstructor; default?: UMAPConstructor };
    showDirectoryPicker?: (options?: {
      mode: 'read' | 'readwrite';
    }) => Promise<DirectoryHandle>;
  }
}

export interface PipelineOptions {

  device?: 'webgpu' | 'wasm' | 'cpu';
  dtype?: 'fp32' | 'fp16' | 'q8';
  progress_callback?: (progress: ProgressEvent) => void;
}

export interface PipelineInstance {
  (input: string | URL | Blob, options?: InferenceOptions): Promise<PipelineOutput>;
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
export interface DirectoryHandle {
  kind: 'directory';
  name: string;
  entries(): AsyncIterableIterator<[string, FileSystemHandle | DirectoryHandle]>;
  [Symbol.asyncIterator](): AsyncIterator<[string, FileSystemHandle | DirectoryHandle]>;
}

export interface FileSystemHandle {
  kind: 'file';
  name: string;
  getFile(): Promise<File>;
}

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
