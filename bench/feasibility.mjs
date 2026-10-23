// Can native onnxruntime (CPU) run EmbeddingGemma 2's q4 export in Node?
import { AutoConfig, AutoModel, AutoProcessor, RawImage } from '@huggingface/transformers'

const repo = 'onnx-community/embeddinggemma-2-ONNX'
const t0 = performance.now()
const processor = await AutoProcessor.from_pretrained(repo)
processor.image_processor.max_soft_tokens = 70
const config = await AutoConfig.from_pretrained(repo)
config.audio_config = null
const model = await AutoModel.from_pretrained(repo, { config, dtype: 'q4', device: 'cpu' })
console.log(`load ${((performance.now() - t0) / 1000).toFixed(1)}s`)

const img = new RawImage(new Uint8ClampedArray(512 * 384 * 4).map((_, i) => (i % 4 === 3 ? 255 : (i * 7) % 256)), 512, 384, 4)
let t = performance.now()
const imgOut = await model(await processor(null, [[img]], null, null))
console.log(`image ${((performance.now() - t) / 1000).toFixed(1)}s dims=${imgOut.sentence_embedding.dims}`)
t = performance.now()
const txtOut = await model(await processor(['task: search result | query: a red sunset'], null, null, null))
console.log(`text ${((performance.now() - t) / 1000).toFixed(1)}s dims=${txtOut.sentence_embedding.dims}`)
