/**
 * Faces: YuNet finds faces and 5 landmarks, SFace turns each aligned face into a 128-d vector,
 * and faces closer than SFace's published cosine threshold are grouped into people.
 *
 * Models are the OpenCV zoo ONNX files (YuNet: MIT, SFace: Apache-2.0), run with onnxruntime-web
 * on WASM: they are small enough that the GPU is not worth claiming from the image embedder.
 * Pre/post-processing mirrors OpenCV's FaceDetectorYN and FaceRecognizerSF::alignCrop.
 */

import * as ort from 'onnxruntime-web'
import { l2normalize } from './embeddings'

const HF = 'https://huggingface.co/opencv'
const YUNET = `${HF}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx`
const SFACE = `${HF}/face_recognition_sface/resolve/main/face_recognition_sface_2021dec_int8.onnx`
const CACHE = 'faces-models-v1'
export const FACES_DOWNLOAD_MB = 10
export const FACE_PREFIX = '@faces/'

const DET = 640 // YuNet 2023mar has a fixed 640×640 input
const SCORE_MIN = 0.8
const NMS_IOU = 0.3
const MIN_FACE_PX = 24 // smaller faces embed too poorly to group
/** SFace's cosine threshold for "same person" (OpenCV zoo). */
export const SAME_PERSON = 0.363
/** Floats stored per face: x, y, w, h (0–1 of the image), score, then the 128-d vector. */
export const FACE_STRIDE = 5 + 128

// ArcFace 112×112 landmark template, the one alignCrop uses.
const TEMPLATE: [number, number][] = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041]
]

export interface Face {
  box: [number, number, number, number] // x, y, w, h in source pixels
  score: number
  kps: [number, number][] // 5 landmarks in source pixels
}

// ── Pure parts (tested in e2e/faces.check.ts) ───────────────────────────────

/** Decode YuNet's per-stride outputs (in 640×640 input pixels), then non-maximum suppression. */
export function decodeYunet(out: Record<string, Float32Array>, size = DET, scoreMin = SCORE_MIN): Face[] {
  const faces: Face[] = []
  for (const s of [8, 16, 32]) {
    const cols = size / s
    const cls = out[`cls_${s}`]
    const obj = out[`obj_${s}`]
    const bbox = out[`bbox_${s}`]
    const kps = out[`kps_${s}`]
    for (let idx = 0; idx < cls.length; idx++) {
      const score = Math.sqrt(Math.min(1, Math.max(0, cls[idx])) * Math.min(1, Math.max(0, obj[idx])))
      if (score < scoreMin) continue
      const r = Math.floor(idx / cols)
      const c = idx % cols
      const cx = (c + bbox[idx * 4]) * s
      const cy = (r + bbox[idx * 4 + 1]) * s
      const w = Math.exp(bbox[idx * 4 + 2]) * s
      const h = Math.exp(bbox[idx * 4 + 3]) * s
      const k: [number, number][] = []
      for (let n = 0; n < 5; n++) k.push([(kps[idx * 10 + 2 * n] + c) * s, (kps[idx * 10 + 2 * n + 1] + r) * s])
      faces.push({ box: [cx - w / 2, cy - h / 2, w, h], score, kps: k })
    }
  }
  faces.sort((a, b) => b.score - a.score)
  const kept: Face[] = []
  for (const f of faces) if (kept.every((k) => iou(k.box, f.box) < NMS_IOU)) kept.push(f)
  return kept
}

function iou(a: Face['box'], b: Face['box']): number {
  const x = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]))
  const y = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]))
  const inter = x * y
  return inter / (a[2] * a[3] + b[2] * b[3] - inter)
}

/**
 * Least-squares similarity transform (rotation, uniform scale, shift) taking `src` points onto
 * `dst`, as canvas setTransform arguments [a, b, c, d, e, f].
 */
export function similarity(src: [number, number][], dst: [number, number][] = TEMPLATE): number[] {
  const n = src.length
  const mean = (p: [number, number][], i: 0 | 1) => p.reduce((s, q) => s + q[i], 0) / n
  const [sx, sy, dx, dy] = [mean(src, 0), mean(src, 1), mean(dst, 0), mean(dst, 1)]
  let num1 = 0
  let num2 = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    const [x, y] = [src[i][0] - sx, src[i][1] - sy]
    const [u, v] = [dst[i][0] - dx, dst[i][1] - dy]
    num1 += x * u + y * v
    num2 += x * v - y * u
    den += x * x + y * y
  }
  const a = num1 / den
  const b = num2 / den
  return [a, b, -b, a, dx - a * sx + b * sy, dy - b * sx - a * sy]
}

/**
 * Group face vectors into people: each face joins the closest person whose mean vector is within
 * SAME_PERSON, else starts a new one. Returns a person id per face.
 * shortcut: one greedy pass, order-dependent and O(faces × people); switch to HDBSCAN if groups split or merge badly on big libraries.
 */
export function groupFaces(vectors: Float32Array[], threshold = SAME_PERSON): Int32Array {
  const ids = new Int32Array(vectors.length)
  const sums: Float32Array[] = []
  for (let i = 0; i < vectors.length; i++) {
    const v = vectors[i]
    let best = -1
    let bestCos = threshold
    for (let p = 0; p < sums.length; p++) {
      const c = cosine(v, sums[p])
      if (c >= bestCos) (best = p), (bestCos = c)
    }
    if (best < 0) {
      best = sums.length
      sums.push(new Float32Array(v.length))
    }
    const s = sums[best]
    for (let d = 0; d < v.length; d++) s[d] += v[d]
    ids[i] = best
  }
  return ids
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) (dot += a[i] * b[i]), (nb += b[i] * b[i])
  return dot / Math.sqrt(nb || 1) // a is unit length
}

// ── Models ──────────────────────────────────────────────────────────────────

async function fetchCached(url: string): Promise<ArrayBuffer> {
  const cache = typeof caches !== 'undefined' ? await caches.open(CACHE).catch(() => null) : null
  const hit = await cache?.match(url)
  if (hit) return hit.arrayBuffer()
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Face model download failed: ${res.status}`)
  const buf = await res.arrayBuffer()
  void cache?.put(url, new Response(buf.slice(0))).catch(() => {})
  return buf
}

export interface FaceModels {
  det: ort.InferenceSession
  rec: ort.InferenceSession
}

export async function loadFaceModels(host?: string): Promise<FaceModels> {
  ort.env.wasm.wasmPaths ||= 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'
  const at = (u: string) => (host ? u.replace('https://huggingface.co', host.replace(/\/+$/, '')) : u)
  const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], logSeverityLevel: 3 }
  const [det, rec] = await Promise.all([
    fetchCached(at(YUNET)).then((b) => ort.InferenceSession.create(b, opts)),
    fetchCached(at(SFACE)).then((b) => ort.InferenceSession.create(b, opts))
  ])
  return { det, rec }
}

// ── Per image ───────────────────────────────────────────────────────────────

const detCanvas = new OffscreenCanvas(DET, DET)
const detCtx = detCanvas.getContext('2d', { willReadFrequently: true })!
const recCanvas = new OffscreenCanvas(112, 112)
const recCtx = recCanvas.getContext('2d', { willReadFrequently: true })!

/** Planar float tensor from a canvas; YuNet wants BGR, SFace RGB, both 0–255 unnormalized. */
function toTensor(ctx: OffscreenCanvasRenderingContext2D, size: number, bgr: boolean): ort.Tensor {
  const { data } = ctx.getImageData(0, 0, size, size)
  const px = size * size
  const t = new Float32Array(3 * px)
  const [c0, c2] = bgr ? [2, 0] : [0, 2]
  for (let i = 0; i < px; i++) {
    t[i] = data[i * 4 + c0]
    t[px + i] = data[i * 4 + 1]
    t[2 * px + i] = data[i * 4 + c2]
  }
  return new ort.Tensor('float32', t, [1, 3, size, size])
}

/**
 * Faces in `bmp` as packed floats, FACE_STRIDE per face (see there); empty when there are none.
 * `bmp` should be ~1280 px on its long side: detection runs at 640, crops come from the full bitmap.
 */
export async function findFaces(m: FaceModels, bmp: ImageBitmap): Promise<Float32Array> {
  const scale = DET / Math.max(bmp.width, bmp.height)
  detCtx.fillStyle = '#000'
  detCtx.fillRect(0, 0, DET, DET)
  detCtx.drawImage(bmp, 0, 0, bmp.width * scale, bmp.height * scale) // letterbox to the top left
  const out = await m.det.run({ input: toTensor(detCtx, DET, true) })
  const raw = Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.data as Float32Array]))
  const faces = decodeYunet(raw).filter((f) => Math.min(f.box[2], f.box[3]) / scale >= MIN_FACE_PX)

  const packed = new Float32Array(faces.length * FACE_STRIDE)
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i]
    const kps = f.kps.map(([x, y]) => [x / scale, y / scale] as [number, number])
    recCtx.fillStyle = '#000'
    recCtx.fillRect(0, 0, 112, 112)
    recCtx.setTransform(...(similarity(kps) as [number, number, number, number, number, number]))
    recCtx.drawImage(bmp, 0, 0)
    recCtx.setTransform(1, 0, 0, 1, 0, 0)
    const emb = (await m.rec.run({ data: toTensor(recCtx, 112, false) })).fc1.data as Float32Array
    const [x, y, w, h] = f.box.map((v) => v / scale)
    packed.set([x / bmp.width, y / bmp.height, w / bmp.width, h / bmp.height, f.score], i * FACE_STRIDE)
    packed.set(l2normalize(emb), i * FACE_STRIDE + 5)
  }
  return packed
}
