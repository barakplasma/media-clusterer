import { test, expect } from '@playwright/test'

// The checks that need no GPU. Both hide WebGPU, so they behave the same
// on a Mac runner, a Linux deploy job or a laptop.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    if (navigator.gpu) navigator.gpu.requestAdapter = async () => null
  })
})

test('viewer mode lays out photos without loading a model', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true })))
  await page.goto('/')
  const r = await page.evaluate(async () => {
    const d = window.__debug
    const names = ['cats.jpg', 'tiger.png', 'football.webp', 'butterfly.jpg']
    const files = await Promise.all(
      names.map(async (n, i) => {
        const blob = await (await fetch(`/e2e/fixtures/${n}`)).blob()
        const file = new File([blob], n, { type: blob.type, lastModified: 1e12 + i * 1000 })
        return { name: `album/${n}`, size: file.size, lastModified: file.lastModified, file, objectURL: null }
      })
    )
    await d.processFiles(files)
    return { phase: d.state.phase, points: d.state.points.length, vectors: d.state.vectors.length }
  })
  expect(r).toEqual({ phase: 'done', points: 4, vectors: 0 })
  await expect(page.locator('#search-input')).toBeDisabled()

  // R jumps to a random photo; E goes back to the one before it.
  await page.keyboard.press('r')
  const first = await page.evaluate(() => window.__debug.state.activeFileIndex)
  await page.keyboard.press('r')
  await page.keyboard.press('e')
  expect(await page.evaluate(() => window.__debug.state.activeFileIndex)).toBe(first)

})

test('date links: year and month filters round-trip through the URL', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true })))
  await page.goto('/')
  // A year or month filter has no day; building its URL threw (Bugsink MEDIA-CLUSTERER-27).
  const hashes = await page.evaluate(() => {
    const d = window.__debug
    return [
      { type: 'datetime', granularity: 'year', year: 2026 },
      { type: 'datetime', granularity: 'month', year: 2026, month: 4 },
      { type: 'datetime', granularity: 'minute', year: 2026, month: 4, day: 9, hour: 7, minute: 5 }
    ].map((s) => {
      d.updateURL(s)
      return { hash: location.hash, parsed: d.parseURLHash(location.hash) }
    })
  })
  expect(hashes.map((h) => h.hash)).toEqual(['#dt:2026', '#dt:2026-05', '#dt:2026-05-09T07:05'])
  expect(hashes.map((h) => h.parsed.granularity)).toEqual(['year', 'month', 'minute'])
  expect(hashes[1].parsed).toMatchObject({ year: 2026, month: 4 })
})

test('a saved WebGPU-only model falls back when there is no adapter', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mc_settings', JSON.stringify({ modelVariant: 'embeddinggemma2' })))
  await page.goto('/')
  // Chrome hides blocklisted GPUs; the modal says how to turn WebGPU on.
  await expect(page.locator('#webgpu-help-modal')).toBeVisible()
  await expect(page.locator('#webgpu-help-url')).toHaveText('chrome://flags/#enable-unsafe-webgpu')
  expect(await page.evaluate(() => window.__debug.state.settings.modelVariant)).toBe('nomic')
  // The saved choice survives a settings write.
  await page.locator('#density-slider').dispatchEvent('input')
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('mc_settings')!).modelVariant)).toBe('embeddinggemma2')
})

test('first visit: the setup wizard checks the device and waits before downloading', async ({ page }) => {
  const modelFetches: string[] = []
  page.on('request', (r) => {
    if (/huggingface\.co|\.onnx/.test(r.url())) modelFetches.push(r.url())
  })
  await page.goto('/')
  await expect(page.locator('#wizard-checks')).toContainText('Graphics (WebGPU)')
  // No adapter here, so the pick is the model that runs on the CPU.
  await expect(page.locator('#wizard-options [aria-checked="true"]')).toContainText('nomic')
  await expect(page.locator('#wizard-go')).toBeVisible()
  await page.waitForTimeout(1000)
  expect(modelFetches, 'no weights before the user starts').toEqual([])
  expect(await page.evaluate(() => localStorage.getItem('mc_settings')), 'first visit stays a first visit').toBeNull()
})

test('folder filter: type, size, video length, date taken and embedded status', async ({ page }) => {
  await page.addInitScript(() =>
    localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true, modelVariant: 'nomic', randomSampleSize: 0 }))
  )
  await page.goto('/')
  // Two cached embeddings: cats.jpg by the current model (nomic, no prefix), tiger.png by another one.
  await page.evaluate(async () => {
    const names = ['cats.jpg', 'tiger.png', 'football.webp', 'butterfly.jpg', 'cats.mp4', 'tiger.webm']
    const files = await Promise.all(
      names.map(async (n, i) => {
        const blob = await (await fetch(`/e2e/fixtures/${n}`)).blob()
        const file = new File([blob], n, { type: blob.type, lastModified: 1e12 + i * 1000 })
        return { name: `album/${n}`, size: file.size, lastModified: file.lastModified, file, objectURL: null }
      })
    )
    await new Promise<void>((resolve, reject) => {
      const req = indexedDB.open('photo-organizer-v1', 2)
      req.onerror = () => reject(req.error)
      req.onsuccess = () => {
        const tx = req.result.transaction('embeddings', 'readwrite')
        const store = tx.objectStore('embeddings')
        // Cache keys use the basename; older builds stored the folder path, which still counts.
        const key = (f: (typeof files)[number]) => `${f.name.split('/').pop()}:${f.size}:${f.lastModified}`
        store.put(new Float32Array(4), key(files[0]))
        store.put(new Float32Array(4), `@other-model/${files[1].name}:${files[1].size}:${files[1].lastModified}`)
        tx.oncomplete = () => resolve()
      }
    })
    ;(window as any).__picked = window.__debug.openFilterDialog(files)
  })
  const count = page.locator('#filter-count')
  const field = (n: string) => page.locator(`#filter-form [name="${n}"]`)
  await expect(count).toHaveText('6 of 6 files match')
  // "To" dates default to today in the native picker, and that counts as open-ended.
  const today = await page.evaluate(() => new Date().toLocaleDateString('en-CA'))
  await expect(field('modifiedTo')).toHaveValue(today)
  await expect(field('createdTo')).toHaveValue(today)

  await field('videos').uncheck()
  await expect(count).toHaveText('4 of 6 files match')
  await field('videos').check()

  await field('embedded').selectOption('current')
  await expect(count).toHaveText('1 of 6 files match')
  await field('embedded').selectOption('some-model')
  await expect(count).toHaveText('2 of 6 files match')
  await field('embedded').selectOption('none')
  await expect(count).toHaveText('4 of 6 files match')
  await field('embedded').selectOption('any')

  // Every fixture video lasts under an hour, so a one-hour minimum keeps only the photos.
  await field('minDurationSec').fill('3600')
  await expect(count).toHaveText('4 of 6 files match')
  await field('minDurationSec').fill('')
  await field('images').uncheck()
  await field('maxDurationSec').fill('3600')
  await expect(count).toHaveText('2 of 6 files match')
  await page.locator('#filter-reset').click()

  // No fixture was taken in 2100, and files without EXIF use their 2001 modified date.
  await field('createdFrom').fill('2100-01-01')
  await expect(count).toHaveText('0 of 6 files match')
  await expect(page.locator('#filter-load')).toBeDisabled()
  await page.locator('#filter-reset').click()

  await field('maxSizeMB').fill('0.000001')
  await expect(count).toHaveText('0 of 6 files match')
  await field('maxSizeMB').fill('')
  await field('videos').uncheck()
  await expect(count).toHaveText('4 of 6 files match')
  await page.locator('#filter-load').click()
  const picked = await page.evaluate(async () => ((await (window as any).__picked) as { name: string }[]).map((f) => f.name))
  expect(picked.sort()).toEqual(['album/butterfly.jpg', 'album/cats.jpg', 'album/football.webp', 'album/tiger.png'])
  // The choice is remembered for the next folder.
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('mc_fileFilter')!))
  expect(saved).toMatchObject({ videos: false, modifiedTo: '', createdTo: '' })
})

test('overnight: two models, a crashing file and a hanging file, then resume', async ({ page }) => {
  test.setTimeout(240_000) // each worker frame loads the whole app from the dev server
  // Init scripts run in every frame, so the hidden worker frames get the stand-in model too.
  await page.addInitScript(() => {
    if (window === window.top && !sessionStorage.getItem('mc_test_seeded')) {
      sessionStorage.setItem('mc_test_seeded', '1')
      localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true, modelVariant: 'nomic', randomSampleSize: 0 }))
    }
    const w = window as any
    w.__mcOvernightTest = { chunkSize: 2, stallMs: 2000, loadStallMs: 30000 }
    w.__mcTestEmbedder = async (model: string, f: { name: string }) => {
      if (f.name.endsWith('butterfly.jpg')) return new Promise(() => {}) // hangs the frame every time
      if (f.name.endsWith('football.webp') && !localStorage.getItem(`crashed:${model}`)) {
        localStorage.setItem(`crashed:${model}`, '1')
        throw new Error('simulated crash') // once per model
      }
      return new Float32Array(8).fill(model.length)
    }
    // The folder picker returns an OPFS folder holding the fixtures.
    w.showDirectoryPicker = async () => {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('overnight-fixtures', { create: true })
      for (const n of ['cats.jpg', 'tiger.png', 'football.webp', 'butterfly.jpg', 'cats.mp4', 'tiger.webm']) {
        const h = await dir.getFileHandle(n, { create: true })
        if ((await h.getFile()).size === 0) {
          const wr = await h.createWritable()
          await wr.write(await (await fetch(`/e2e/fixtures/${n}`)).blob())
          await wr.close()
        }
      }
      return dir
    }
  })
  let projectionWorkerLoaded = false
  page.on('request', (r) => {
    if (r.url().includes('projectionWorker')) projectionWorkerLoaded = true
  })
  await page.goto('/')

  const pickModels = async () => {
    await page.locator('#overnight-btn').click()
    await expect(page.locator('#filter-count')).toHaveText('6 of 6 files match')
    await page.locator('#filter-load').click()
    await expect(page.locator('#overnight-setup')).toBeVisible()
  }
  await pickModels()
  const boxes = page.locator('#overnight-models input:not(:disabled)')
  const ids = [await boxes.nth(0).inputValue(), await boxes.nth(1).inputValue()]
  for (const box of await page.locator('#overnight-models input:checked').all()) await box.uncheck()
  for (const id of ids) await page.locator(`#overnight-models input[value="${id}"]`).check()
  await page.locator('#overnight-start').click()

  await expect(page.locator('#overnight-close')).toBeVisible({ timeout: 200_000 })
  await expect(page.locator('#overnight-pct')).toHaveText('100.0%')
  const rows = page.locator('.on-model')
  for (let i = 0; i < 2; i++) {
    await expect(rows.nth(i)).toContainText('done')
    await expect(rows.nth(i)).toContainText('embedded 5 ')
    await expect(rows.nth(i)).toContainText('failed 1 ')
  }
  // Both the crash and the hang cost a fresh frame; the hanging file was narrowed down and skipped.
  const restarts = Number((await page.locator('#overnight-stats').textContent())!.match(/restarts (\d+)/)![1])
  expect(restarts).toBeGreaterThanOrEqual(4)
  // Every vector landed compact and under its model's namespace; the hanging file has none.
  const stored = await page.evaluate(async () => {
    const db: IDBDatabase = await new Promise((res, rej) => {
      const r = indexedDB.open('photo-organizer-v1', 2)
      r.onsuccess = () => res(r.result)
      r.onerror = () => rej(r.error)
    })
    const store = db.transaction('embeddings').objectStore('embeddings')
    const keys: string[] = await new Promise((res) => {
      const r = store.getAllKeys()
      r.onsuccess = () => res(r.result.map(String))
    })
    const first: Float32Array = await new Promise((res) => {
      const r = store.get(keys[0])
      r.onsuccess = () => res(r.result)
    })
    return { keys, bytes: first.buffer.byteLength }
  })
  expect(stored.bytes).toBe(32)
  expect(stored.keys.filter((k) => k.includes('butterfly'))).toEqual([])
  expect(stored.keys.length).toBe(10)
  // Nothing left but a file that hangs, so there is no job to resume.
  expect(await page.evaluate(() => localStorage.getItem('mc_overnight_job'))).toBeNull()
  // The finished run left a projected layout in the Resume slot, for one of the run's models.
  await expect(page.locator('#overnight-line')).toContainText('press Resume', { timeout: 30_000 })
  const slot = await page.evaluate(() => ({
    keys: JSON.parse(localStorage.getItem('po_fileKeys')!).length,
    points: JSON.parse(localStorage.getItem('po_projectedPoints')!).length,
    model: JSON.parse(localStorage.getItem('mc_settings')!).modelVariant
  }))
  expect(slot).toMatchObject({ keys: 5, points: 5 })
  expect(ids).toContain(slot.model)
  expect(projectionWorkerLoaded, 'the layout was projected in the worker (src/projectionWorker.ts)').toBe(true)

  // A second run finds everything cached except the file that never embeds.
  await page.locator('#overnight-close').click()
  await pickModels()
  for (const id of ids) {
    await expect(page.locator('#overnight-models label', { has: page.locator(`input[value="${id}"]`) })).toContainText('1 of 6 to embed')
  }
})

test('saved embeddings: a bundle in the folder is imported and matched by sha256; overnight sample toggle', async ({ page }) => {
  await page.addInitScript(() => {
    if (window === window.top && !sessionStorage.getItem('mc_test_seeded')) {
      sessionStorage.setItem('mc_test_seeded', '1')
      localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true, modelVariant: 'nomic', randomSampleSize: 2 }))
    }
    ;(window as any).showDirectoryPicker = async () => {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('bundle-fixtures', { create: true })
      const sizes: Record<string, number> = {}
      for (const n of ['cats.jpg', 'tiger.png', 'football.webp']) {
        const h = await dir.getFileHandle(n, { create: true })
        if ((await h.getFile()).size === 0) {
          const wr = await h.createWritable()
          await wr.write(await (await fetch(`/e2e/fixtures/${n}`)).blob())
          await wr.close()
        }
        sizes[n] = (await h.getFile()).size
      }
      // Written "on another device": same content (sha256), a different lastModified.
      const { encodeBundle, sha256Hex } = await import('/src/embeddingBundle.ts' as string)
      const key = (n: string) => `${n}:${sizes[n]}:1`
      const sha = async (n: string) => sha256Hex(await (await dir.getFileHandle(n)).getFile())
      const blob = encodeBundle({
        entries: [[key('cats.jpg'), new Float32Array(8).fill(1)], [key('tiger.png'), new Float32Array(8).fill(2)], ['@faces/' + key('cats.jpg'), new Float32Array(0)]],
        captions: [],
        sha256: { [key('cats.jpg')]: await sha('cats.jpg'), [key('tiger.png')]: await sha('tiger.png') }
      })
      const wr = await (await dir.getFileHandle('media-clusterer.mcemb', { create: true })).createWritable()
      await wr.write(blob)
      await wr.close()
      return dir
    }
  })
  await page.goto('/')
  await page.locator('#overnight-btn').click()
  await expect(page.locator('#filter-count')).toContainText('3 of 3 files match')
  await page.locator('#filter-load').click()
  const nomic = page.locator('#overnight-models label', { has: page.locator('input[value="nomic"]') })
  // Sampled 2 of 3: 0, 1 or 2 of the sampled files are among the imported ones, but never 3 to embed.
  await expect(page.locator('#overnight-sample')).toBeChecked()
  await expect(nomic).toContainText(/of 2 to embed|all embedded/)
  await page.locator('#overnight-sample').uncheck()
  await expect(nomic).toContainText('1 of 3 to embed')
  // The remote endpoint is offered even before it is configured, with its form right here.
  await expect(page.locator('#overnight-models input[value="openai"]')).toBeDisabled()
  await expect(page.locator('#overnight-remote #openai-base-url')).toBeAttached()
  await page.locator('#overnight-cancel').click()
  await expect(page.locator('#overnight-remote #openai-base-url')).not.toBeAttached()
})

test('AI provenance: a PNG with A1111/Civitai parameters shows generator, model and prompt in the viewer', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true })))
  await page.goto('/')
  await page.evaluate(async () => {
    // A real PNG from a canvas, with a tEXt "parameters" chunk inserted after IHDR, as A1111 writes it.
    const c = new OffscreenCanvas(64, 64)
    c.getContext('2d')!.fillRect(0, 0, 32, 32)
    const png = new Uint8Array(await (await c.convertToBlob({ type: 'image/png' })).arrayBuffer())
    const text = 'a cat in a hat\nNegative prompt: blurry\nSteps: 30, Sampler: Euler a, Model hash: abc, Model: dreamshaper_8, Civitai resources: []'
    const data = new TextEncoder().encode(`parameters\0${text}`)
    const chunk = new Uint8Array(12 + data.length)
    new DataView(chunk.buffer).setUint32(0, data.length)
    chunk.set(new TextEncoder().encode('tEXt'), 4)
    chunk.set(data, 8)
    const at = 8 + 25 // signature + IHDR chunk
    const file = new File([png.subarray(0, at), chunk, png.subarray(at)], 'gen.png', { type: 'image/png', lastModified: 1e12 })
    const plain = new File([png], 'plain.png', { type: 'image/png', lastModified: 1e12 })
    const d = window.__debug
    await d.processFiles([file, plain].map((f) => ({ name: f.name, size: f.size, lastModified: f.lastModified, file: f, objectURL: null })))
    d.openFileModal(d.state.files.findIndex((f: { name: string }) => f.name === 'gen.png'))
  })
  const badge = page.locator('#modal-ai-btn')
  await expect(badge).toHaveText('✨ AI · Civitai · dreamshaper_8 ▸')
  await badge.click()
  await expect(page.locator('#exif-dialog-body')).toContainText('a cat in a hat')
  await expect(page.locator('#exif-dialog-body')).toContainText('blurry')
  await expect(page.locator('#exif-dialog-body')).toContainText('PNG text chunk "parameters"')
  await page.locator('#exif-dialog-close').click()
  // An ordinary PNG gets no badge.
  await page.evaluate(() => window.__debug.openFileModal(window.__debug.state.files.findIndex((f: { name: string }) => f.name === 'plain.png')))
  await page.waitForTimeout(300)
  await expect(badge).toBeHidden()
})

test('AI provenance parsers: ComfyUI graph in compressed iTXt, A1111 in a UTF-16 UserComment, a C2PA manifest', async ({ page }) => {
  await page.goto('/device-test.html')
  const r = await page.evaluate(async () => {
    const { readProvenance } = await import('/src/aiProvenance.ts' as string)
    const enc = new TextEncoder()
    const cat = (...a: Uint8Array[]) => new Uint8Array(a.reduce((n: number[], x) => [...n, ...x], []))
    const chunk = (type: string, data: Uint8Array) => {
      const head = new Uint8Array(8)
      new DataView(head.buffer).setUint32(0, data.length)
      head.set(enc.encode(type), 4)
      return cat(head, data, new Uint8Array(4))
    }
    const deflate = async (s: string) =>
      new Uint8Array(await new Response(new Blob([enc.encode(s)]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer())
    const sig = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)
    const comfy = JSON.stringify({
      '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sdxl_base.safetensors' } },
      '6': { class_type: 'CLIPTextEncode', inputs: { text: 'a red fox, ünïcödé' } },
      '7': { class_type: 'CLIPTextEncode', inputs: { text: 'ugly' } },
      '3': { class_type: 'KSampler', inputs: { positive: ['6', 0], negative: ['7', 0] } }
    })
    const itxt = chunk('iTXt', cat(enc.encode('prompt\0'), Uint8Array.of(1, 0, 0, 0), await deflate(comfy)))
    const comfyR = await readProvenance(cat(sig, itxt, chunk('IEND', new Uint8Array())), null)

    const params = 'a cat\nNegative prompt: blurry\nSteps: 20, Model: juggernaut'
    const be = new Uint8Array(params.length * 2)
    for (let i = 0; i < params.length; i++) be[2 * i + 1] = params.charCodeAt(i)
    const ucR = await readProvenance(new Uint8Array(64), { UserComment: cat(enc.encode('UNICODE\0'), be) })

    const gen = 'Google Gemini 1.0'
    const c2pa = cat(enc.encode('..jumb..jumdc2pa..c2pa.claim..claim_generator'), Uint8Array.of(0x60 + gen.length), enc.encode(gen),
      enc.encode('..c2pa.actions..http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'))
    const c2paR = await readProvenance(c2pa, null)
    return { comfyR, ucR, c2paR }
  })
  expect(r.comfyR).toMatchObject({ generator: 'ComfyUI', prompt: 'a red fox, ünïcödé', negative: 'ugly', model: 'sdxl_base.safetensors' })
  expect(r.ucR).toMatchObject({ generator: 'Stable Diffusion WebUI', prompt: 'a cat', negative: 'blurry', model: 'juggernaut' })
  expect(r.c2paR.generator).toBe('Google')
  expect(r.c2paR.evidence.join(' ')).toContain('C2PA manifest by Google Gemini 1.0, declares AI generation (signature not verified)')
})
