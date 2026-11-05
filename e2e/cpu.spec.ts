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
        const key = (f: (typeof files)[number]) => `${f.name}:${f.size}:${f.lastModified}`
        store.put(new Float32Array(4), key(files[0]))
        store.put(new Float32Array(4), `@other-model/${key(files[1])}`)
        tx.oncomplete = () => resolve()
      }
    })
    ;(window as any).__picked = window.__debug.openFilterDialog(files)
  })
  const count = page.locator('#filter-count')
  const field = (n: string) => page.locator(`#filter-form [name="${n}"]`)
  await expect(count).toHaveText('6 of 6 files match')

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
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('mc_fileFilter')!).videos)).toBe(false)
})
