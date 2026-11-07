/**
 * What a file itself says about being AI-generated: the generator, model and prompt when it carries
 * them, and which field said so. Read from the bytes, nothing is uploaded. See AI_PROVENANCE_PLAN.md.
 *
 * Sources: PNG text chunks (A1111/Forge/Civitai, ComfyUI, NovelAI, InvokeAI, Midjourney), EXIF
 * UserComment/Software/ImageDescription (A1111 JPEG/WebP), XMP/IPTC DigitalSourceType, and C2PA
 * manifests (Gemini/Imagen, OpenAI, Adobe Firefly, Microsoft…), whose signatures are NOT verified.
 * SynthID watermarks cannot be read on a device.
 */

export interface Provenance {
  generator: string
  model?: string
  prompt?: string
  negative?: string
  evidence: string[] // where it came from, e.g. 'PNG text chunk "parameters"'
  settings?: string // the generator's own settings line, verbatim
}

export interface ExifFields {
  UserComment?: unknown
  Software?: unknown
  ImageDescription?: unknown
}

// IPTC digital source types meaning generated or AI-composited media. Not plain "algorithmicMedia":
// that is media made without training data (renders, procedural art).
const AI_SOURCE = /digitalsourcetype\/(trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|compositeSynthetic)\b/i
const KNOWN_TOOLS: [RegExp, string][] = [
  [/civitai/i, 'Civitai'],
  [/tensor\.?art/i, 'Tensor.Art'],
  [/novelai/i, 'NovelAI'],
  [/midjourney/i, 'Midjourney'],
  [/dall[·.\- ]?e|openai|chatgpt|gpt-image/i, 'OpenAI'],
  [/firefly/i, 'Adobe Firefly'],
  [/imagen|gemini|google/i, 'Google'],
  [/bing image creator|microsoft designer|microsoft/i, 'Microsoft'],
  [/replicate/i, 'Replicate'],
  [/flux|black ?forest/i, 'FLUX'],
  [/comfyui/i, 'ComfyUI'],
  [/invokeai/i, 'InvokeAI'],
  [/stable[ -]?diffusion|automatic1111|forge/i, 'Stable Diffusion WebUI']
]
const toolIn = (s: string) => KNOWN_TOOLS.find(([re]) => re.test(s))?.[1]

// ── PNG text chunks ──────────────────────────────────────────────────────────

/** tEXt, iTXt and zTXt chunks of a PNG, keyword → text. Empty for anything that is not a PNG. */
export async function pngText(buf: Uint8Array): Promise<Record<string, string>> {
  const out: Record<string, string> = {}
  if (buf.length < 8 || buf[0] !== 0x89 || buf[1] !== 0x50 || buf[2] !== 0x4e || buf[3] !== 0x47) return out
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
  const latin1 = new TextDecoder('latin1')
  const utf8 = new TextDecoder()
  for (let p = 8; p + 12 <= buf.length; ) {
    const len = dv.getUint32(p)
    const type = latin1.decode(buf.subarray(p + 4, p + 8))
    const data = buf.subarray(p + 8, p + 8 + len)
    p += 12 + len
    if (type === 'IEND') break
    if (type !== 'tEXt' && type !== 'iTXt' && type !== 'zTXt') continue
    const nul = data.indexOf(0)
    if (nul < 0) continue
    const key = latin1.decode(data.subarray(0, nul))
    try {
      if (type === 'tEXt') out[key] = latin1.decode(data.subarray(nul + 1))
      else if (type === 'zTXt') out[key] = latin1.decode(await inflate(data.subarray(nul + 2)))
      else {
        // iTXt: keyword\0 compressed(1) method(1) language\0 translated\0 text
        const compressed = data[nul + 1] === 1
        const lang = data.indexOf(0, nul + 3)
        const trans = data.indexOf(0, lang + 1)
        const text = data.subarray(trans + 1)
        out[key] = utf8.decode(compressed ? await inflate(text) : text)
      }
    } catch (_) {
      // a corrupt chunk loses that chunk only
    }
  }
  return out
}

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('deflate'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

// ── Generator formats ────────────────────────────────────────────────────────

/** A1111 / Forge / Civitai "parameters": prompt, optional "Negative prompt:" line(s), then a settings line. */
export function parseA1111(text: string): Pick<Provenance, 'prompt' | 'negative' | 'model' | 'settings'> {
  const lines = text.trim().split('\n')
  let at = lines.length - 1
  while (at >= 0 && !/^Steps: /.test(lines[at])) at--
  const settings = at >= 0 ? lines.slice(at).join('\n') : ''
  const head = (at >= 0 ? lines.slice(0, at) : lines).join('\n')
  const neg = head.indexOf('\nNegative prompt:')
  const negStart = head.startsWith('Negative prompt:') ? 0 : neg
  const prompt = (negStart >= 0 ? head.slice(0, negStart) : head).trim()
  const negative = negStart >= 0 ? head.slice(negStart).replace(/^\n?Negative prompt:\s*/, '').trim() : undefined
  const model = settings.match(/(?:^|, )Model: ([^,]+)/)?.[1]?.trim()
  return { prompt: prompt || undefined, negative: negative || undefined, model, settings: settings || undefined }
}

type ComfyNode = { class_type?: string; inputs?: Record<string, unknown> }

/** ComfyUI "prompt" (API graph): the checkpoint, and the text that reaches the sampler's positive/negative. */
export function parseComfy(json: string): Pick<Provenance, 'prompt' | 'negative' | 'model'> {
  const g = JSON.parse(json) as Record<string, ComfyNode>
  const nodes = Object.values(g)
  const model = nodes
    .map((n) => n.inputs?.ckpt_name ?? n.inputs?.unet_name ?? n.inputs?.model_name)
    .find((v): v is string => typeof v === 'string')
  // Follow a link ([nodeId, slot]) back until a node with literal text.
  const textOf = (ref: unknown, depth = 0): string | undefined => {
    if (typeof ref === 'string') return ref
    if (!Array.isArray(ref) || depth > 8) return undefined
    const n = g[String(ref[0])]
    const inp = n?.inputs ?? {}
    for (const k of ['text', 'text_g', 'prompt', 'positive', 'conditioning', 'string', 'value']) {
      if (k in inp) {
        const t = textOf(inp[k], depth + 1)
        if (t) return t
      }
    }
    return undefined
  }
  const sampler = nodes.find((n) => n.inputs && 'positive' in n.inputs)
  let prompt = sampler ? textOf(sampler.inputs!.positive) : undefined
  const negative = sampler ? textOf(sampler.inputs!.negative) : undefined
  // No sampler (custom pipelines): the longest text encoder input is the best guess.
  prompt ??= nodes
    .filter((n) => /TextEncode/i.test(n.class_type ?? ''))
    .map((n) => n.inputs?.text)
    .filter((t): t is string => typeof t === 'string')
    .sort((a, b) => b.length - a.length)[0]
  return { prompt, negative, model }
}

// ── C2PA ─────────────────────────────────────────────────────────────────────

/** A C2PA manifest's claim generator and whether it declares AI generation, from the raw bytes (unverified). */
export function c2paHint(buf: Uint8Array): { generator?: string; ai: boolean } | null {
  // JUMBF boxes carry the label "c2pa"; found in JPEG APP11, PNG caBX and WebP C2PA chunks alike.
  const text = new TextDecoder('latin1').decode(buf)
  if (!/jumb[\s\S]{0,64}c2pa|caBX|C2PA/.test(text)) return null
  if (!/c2pa\.(claim|actions|assertions)|claim_generator/.test(text)) return null
  let generator: string | undefined
  for (const key of ['claim_generator', 'name']) {
    // CBOR: key as a text string, then the value as a text string (0x60–0x77 short, 0x78 one-byte length).
    for (let i = text.indexOf(key); i >= 0 && !generator; i = text.indexOf(key, i + 1)) {
      let p = i + key.length
      const h = buf[p++]
      const len = h >= 0x60 && h <= 0x77 ? h - 0x60 : h === 0x78 ? buf[p++] : -1
      if (len > 1) {
        const v = new TextDecoder().decode(buf.subarray(p, p + len))
        if (/^[\x20-\x7e]+$/.test(v) && !/^c2pa/.test(v)) generator = v
      }
      if (key === 'name' && generator && !toolIn(generator)) generator = undefined
    }
    if (generator) break
  }
  return { generator, ai: AI_SOURCE.test(text) || /c2pa\.created[\s\S]{0,200}trainedAlgorithmic/.test(text) }
}

// ── All together ─────────────────────────────────────────────────────────────

const decodeUserComment = (v: unknown): string => {
  if (typeof v === 'string') return v
  if (!(v instanceof Uint8Array)) return ''
  const head = new TextDecoder('latin1').decode(v.subarray(0, 8))
  const body = v.subarray(8)
  if (head.startsWith('UNICODE')) {
    // piexif (A1111) writes UTF-16BE; some tools write LE. The one with fewer NULs in odd bytes wins.
    const be = new TextDecoder('utf-16be').decode(body)
    const le = new TextDecoder('utf-16le').decode(body)
    const bad = (s: string) => (s.match(/[Ā-￿]/g) ?? []).length
    return (bad(be) <= bad(le) ? be : le).replace(/\0+$/, '')
  }
  return new TextDecoder().decode(head.startsWith('ASCII') ? body : v).replace(/\0+$/, '')
}

/**
 * Everything the file says about being AI-generated, or null when it says nothing.
 * `exif` is the app's already-parsed EXIF (exifr), so the file is not parsed twice.
 */
export async function readProvenance(buf: Uint8Array, exif: ExifFields | null): Promise<Provenance | null> {
  const ev: string[] = []
  let p: Partial<Provenance> = {}
  let generator: string | undefined

  const png = await pngText(buf)
  if (png.parameters) {
    ev.push('PNG text chunk "parameters"')
    p = parseA1111(png.parameters)
    generator = toolIn(png.parameters) ?? 'Stable Diffusion WebUI'
  }
  if (png.prompt && !p.prompt) {
    try {
      p = parseComfy(png.prompt)
      ev.push('PNG text chunk "prompt" (ComfyUI graph)')
      generator = toolIn(png.prompt + (png.workflow ?? '')) === 'Tensor.Art' ? 'Tensor.Art' : 'ComfyUI'
    } catch (_) {
      // not JSON: some tools store a plain prompt here
      p.prompt = png.prompt
      ev.push('PNG text chunk "prompt"')
    }
  }
  if (png.invokeai_metadata) {
    try {
      const m = JSON.parse(png.invokeai_metadata)
      p = { prompt: m.positive_prompt, negative: m.negative_prompt, model: m.model?.name ?? m.model?.model_name }
      ev.push('PNG text chunk "invokeai_metadata"')
      generator = 'InvokeAI'
    } catch (_) {
      // malformed: other evidence may still identify it
    }
  }
  if (png.Comment && /novelai/i.test(png.Software ?? '')) {
    try {
      const m = JSON.parse(png.Comment)
      p = { prompt: m.prompt, negative: m.uc, model: png.Source }
      ev.push('PNG text chunks "Comment" + "Software: NovelAI"')
      generator = 'NovelAI'
    } catch (_) {
      // malformed Comment
    }
  }
  if (!generator && png.Description && /midjourney/i.test(png.Author + png.Description + (png.Software ?? ''))) {
    p.prompt = png.Description.replace(/\s*Job ID:.*$/s, '')
    ev.push('PNG text chunk "Description" (Midjourney)')
    generator = 'Midjourney'
  }

  const uc = decodeUserComment(exif?.UserComment)
  if (!generator && /\nSteps: |^Steps: /m.test(uc)) {
    ev.push('EXIF UserComment (A1111 parameters)')
    p = parseA1111(uc)
    generator = toolIn(uc) ?? 'Stable Diffusion WebUI'
  }
  const sw = [exif?.Software, png.Software].filter((s): s is string => typeof s === 'string').join(' ')
  const swTool = sw ? toolIn(sw) : undefined
  if (swTool) {
    ev.push(`Software: ${sw.trim()}`)
    generator ??= swTool
  }

  const text = new TextDecoder('latin1').decode(buf.subarray(0, Math.min(buf.length, 4 << 20)))
  const src = text.match(AI_SOURCE)
  if (src) ev.push(`IPTC/XMP DigitalSourceType: ${src[1]}`)
  const xmpTool = text.match(/CreatorTool[=>"']+([^"<]+)/)?.[1]
  if (xmpTool && toolIn(xmpTool)) {
    ev.push(`XMP CreatorTool: ${xmpTool}`)
    generator ??= toolIn(xmpTool)
  }
  if (!p.prompt && src) {
    const desc = text.match(/<dc:description>[\s\S]*?<rdf:li[^>]*>([^<]+)</)?.[1]
    if (desc) p.prompt = desc
  }

  const c2pa = c2paHint(buf)
  if (c2pa) {
    ev.push(`C2PA manifest${c2pa.generator ? ` by ${c2pa.generator}` : ''}${c2pa.ai ? ', declares AI generation' : ''} (signature not verified)`)
    if (c2pa.ai || (c2pa.generator && toolIn(c2pa.generator) && !/photoshop|lightroom|camera/i.test(c2pa.generator))) {
      generator ??= (c2pa.generator && toolIn(c2pa.generator)) || c2pa.generator || 'Unknown (C2PA)'
    }
  }

  if (!generator && !src) return null
  return { generator: generator ?? 'Unknown generator', ...p, evidence: ev }
}
