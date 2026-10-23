// Embed a folder of images with each candidate model, the way the app does,
// and cache the vectors so metrics and projections can be re-run for free.
//
//   node bench/embed.mjs --data <dir with images/> --models eg2-phone,nomic-mean
//
// Writes <out>/<model>.f32 (row-major float32, one L2-normalised row per
// image) and <out>/<model>.json ({ names, dim, seconds }).
//
// Model settings mirror src/vlmTiers.ts and src/app.ts; keep them in sync.

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import sharp from 'sharp'
import ort from 'onnxruntime-node'
import { AutoConfig, AutoModel, AutoProcessor, RawImage, pipeline } from '@huggingface/transformers'

const { values: args } = parseArgs({
  options: {
    data: { type: 'string' },
    out: { type: 'string', default: 'bench/out' },
    models: { type: 'string', default: 'eg2-phone,eg2-desktop,smolvlm2,nomic-mean,nomic-cls' },
    limit: { type: 'string' },
    models_dir: { type: 'string', default: 'bench/models' }
  }
})

// src/vlmTiers.ts: VLM_FRAME_PX. The app hands the VLM tiers
// createImageBitmap(file, { resizeWidth: 512 }).
const VLM_FRAME_PX = 512

async function vlmFrame(path) {
  const { data, info } = await sharp(path)
    .rotate() // honour EXIF orientation, as createImageBitmap does
    .resize({ width: VLM_FRAME_PX })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true })
  return new RawImage(new Uint8ClampedArray(data), info.width, info.height, 4)
}

function l2(v) {
  let n = 0
  for (const x of v) n += x * x
  n = Math.sqrt(n) || 1
  return Float32Array.from(v, (x) => x / n)
}

// ── Models ──────────────────────────────────────────────────────────────────

/** EmbeddingGemma 2, as src/vlmWorker.ts loads and runs it. */
async function eg2(imageTokens, fixedSeqLen) {
  const repo = 'onnx-community/embeddinggemma-2-ONNX'
  const processor = await AutoProcessor.from_pretrained(repo)
  processor.image_processor.max_soft_tokens = imageTokens
  const config = await AutoConfig.from_pretrained(repo)
  config.audio_config = null
  const model = await AutoModel.from_pretrained(repo, { config, dtype: 'q4', device: 'cpu' })
  const opts = fixedSeqLen ? { padding: 'max_length', max_length: fixedSeqLen, truncation: true } : {}
  return {
    async embed(path) {
      const inputs = await processor(null, [[await vlmFrame(path)]], null, null, opts)
      const out = await model(inputs)
      return l2(out.sentence_embedding.data)
    }
  }
}

/**
 * SmolVLM2 vision tower + mean pool (src/vlmWorker.ts embedGroupSmolVlm).
 *
 * Runs onnx/vision_encoder_q4.onnx directly: AutoModel with model_file_name
 * 'vision_encoder' (the app's way) still builds the full model and demands
 * input_ids. The app loads q4f16 on WebGPU; the CPU EP has no fp16 kernels
 * for it, so the CPU stand-in is q4 (same quantised weights).
 */
async function smolvlm2() {
  const repo = 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'
  const processor = await AutoProcessor.from_pretrained(repo)
  const file = join(args.models_dir, 'smolvlm2-vision_encoder_q4.onnx')
  try {
    await readFile(file)
  } catch {
    const res = await fetch(`https://huggingface.co/${repo}/resolve/main/onnx/vision_encoder_q4.onnx`)
    await mkdir(args.models_dir, { recursive: true })
    await writeFile(file, Buffer.from(await res.arrayBuffer()))
  }
  const session = await ort.InferenceSession.create(file)
  const toOrt = (t, type) => new ort.Tensor(type, type === 'bool' ? Uint8Array.from(t.data, Number) : t.data, t.dims)
  return {
    async embed(path) {
      // The app calls processor(raw, ...), which Idefics3Processor reads as
      // (text, images) and crashes on (reported to #31); the image processor
      // is what it means to call.
      const inputs = await processor.image_processor(await vlmFrame(path), { do_image_splitting: false })
      const out = await session.run({
        pixel_values: toOrt(inputs.pixel_values, 'float32'),
        pixel_attention_mask: toOrt(inputs.pixel_attention_mask, 'bool')
      })
      const f = out.image_features
      const dim = f.dims[f.dims.length - 1]
      const tokens = f.data.length / dim
      const mean = new Float32Array(dim)
      for (let t = 0; t < tokens; t++) for (let d = 0; d < dim; d++) mean[d] += f.data[t * dim + d]
      return l2(mean)
    }
  }
}

/** nomic-embed-vision-v1.5 through the pipeline, as src/app.ts does (file in, pipeline resizes). */
async function nomic(pooling) {
  const extractor = await pipeline('image-feature-extraction', 'nomic-ai/nomic-embed-vision-v1.5', {
    device: 'cpu',
    dtype: 'fp32'
  })
  return {
    async embed(path) {
      const image = await RawImage.read(path)
      // The pipeline ignores { pooling, normalize } for image features and
      // returns every token's hidden state, [1, 197, 768].
      const out = await extractor(image)
      const [, tokens, dim] = out.dims
      if (pooling === 'cls') {
        // Nomic's model card: the CLS token (first position).
        return l2(out.data.slice(0, dim))
      }
      // src/embeddings.ts extractVector: mean over every token, then L2.
      const mean = new Float32Array(dim)
      for (let t = 0; t < tokens; t++) for (let d = 0; d < dim; d++) mean[d] += out.data[t * dim + d]
      return l2(mean)
    }
  }
}

const MODELS = {
  'eg2-phone': () => eg2(70, 296), // PHONE_OVERRIDES in src/vlmTiers.ts
  'eg2-desktop': () => eg2(280, 0),
  smolvlm2: () => smolvlm2(),
  'nomic-mean': () => nomic('mean'),
  'nomic-cls': () => nomic('cls')
}

// ── Run ─────────────────────────────────────────────────────────────────────

const imgDir = join(args.data, 'images')
let names = (await readdir(imgDir)).filter((n) => /\.(jpe?g|png|webp)$/i.test(n)).sort()
if (args.limit) names = names.slice(0, Number(args.limit))
await mkdir(args.out, { recursive: true })

for (const id of args.models.split(',')) {
  if (!MODELS[id]) throw new Error(`unknown model ${id}; known: ${Object.keys(MODELS).join(', ')}`)
  const t0 = performance.now()
  const m = await MODELS[id]()
  const loadS = (performance.now() - t0) / 1000
  let dim = 0
  const rows = []
  const t1 = performance.now()
  for (const [i, name] of names.entries()) {
    const v = await m.embed(join(imgDir, name))
    dim = v.length
    rows.push(v)
    if ((i + 1) % 100 === 0) {
      const per = (performance.now() - t1) / 1000 / (i + 1)
      console.log(`${id} ${i + 1}/${names.length} ${per.toFixed(2)}s/img`)
    }
  }
  const seconds = (performance.now() - t1) / 1000
  const buf = new Float32Array(rows.length * dim)
  rows.forEach((r, i) => buf.set(r, i * dim))
  await writeFile(join(args.out, `${id}.f32`), Buffer.from(buf.buffer))
  await writeFile(join(args.out, `${id}.json`), JSON.stringify({ names, dim, loadS, seconds }))
  console.log(`${id} done: ${names.length} images, dim ${dim}, load ${loadS.toFixed(1)}s, ${(seconds / names.length).toFixed(2)}s/img`)
}
