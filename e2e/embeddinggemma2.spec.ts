import { test, expect } from '@playwright/test'

const FIXTURES = ['cats.jpg', 'tiger.png', 'football.webp', 'butterfly.jpg', 'cats.mp4', 'tiger.webm']
// query -> fixture-name prefix the top result must have
const QUERIES: [string, string][] = [
  ['cats sleeping on a couch', 'cats'],
  ['a tiger', 'tiger'],
  ['people playing soccer', 'football'],
  ['a butterfly on a flower', 'butterfly'],
]

type Debug = {
  state: {
    phase: string
    files: { name: string }[]
    vectors: Float32Array[]
    points: unknown[]
    searchResults: Int32Array | null
    settings: { modelVariant: string }
    captions: (string | null)[]
  }
  embedFailed: Set<number>
  processFiles: (files: unknown[]) => Promise<void>
  updateURL: (s: unknown) => void
  parseURLHash: (h: string) => { granularity: string; year: number; month?: number }
  searchImages: (q: string) => Promise<void>
}
declare global {
  interface Window {
    __debug: Debug
  }
}

test('EmbeddingGemma 2 embeds photos and videos and finds them by text', async ({ page }, testInfo) => {
  const webnn = testInfo.project.name === 'webnn'
  page.on('console', (m) => {
    if (!m.text().includes('[vite]')) console.log(`[${m.type()}] ${m.text().slice(0, 300)}`)
  })
  page.on('pageerror', (e) => console.log('[pageerror]', e.message))
  await page.addInitScript((webnn) => {
    localStorage.setItem('mc_settings', JSON.stringify({ modelVariant: 'embeddinggemma2', enableTextSearch: true }))
  }, webnn)
  await page.goto('/')

  const adapter = await page.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter()
    return a && { vendor: a.info.vendor, fallback: a.info.isFallbackAdapter }
  })
  console.log('adapter', adapter)
  if (webnn) {
    expect(await page.evaluate(() => 'ml' in navigator), 'no WebNN (navigator.ml)').toBe(true)
  } else {
    expect(adapter, 'this browser has no WebGPU adapter').toBeTruthy()
    if (!process.env.SWIFTSHADER) expect(adapter!.fallback, 'software WebGPU adapter; needs a real GPU').toBe(false)
  }

  // The app starts loading the model on its own.
  const t0 = Date.now()
  await page.waitForFunction(() => window.__debug?.state.phase === 'model_ready', null, {
    timeout: 20 * 60_000,
    polling: 2000,
  })
  console.log(`model ready in ${((Date.now() - t0) / 1000).toFixed(0)}s`)
  // WebNN is preferred whenever offered, most native device first.
  if (webnn) await expect(page.locator('#device-badge')).toContainText('WebNN')

  const t1 = Date.now()
  const run = await page.evaluate(async (names) => {
    const d = window.__debug
    const files = await Promise.all(
      names.map(async (n, i) => {
        const blob = await (await fetch(`/e2e/fixtures/${n}`)).blob()
        const file = new File([blob], n, { type: blob.type, lastModified: 1e12 + i * 1000 })
        return { name: `e2e/${n}`, size: file.size, lastModified: file.lastModified, file, objectURL: null }
      })
    )
    await d.processFiles(files)
    return {
      phase: d.state.phase,
      dims: d.state.vectors.map((v) => v.length),
      failed: [...d.embedFailed].map((i) => d.state.files[i].name),
    }
  }, FIXTURES)
  console.log(`embedded ${FIXTURES.length} files in ${((Date.now() - t1) / 1000).toFixed(0)}s`, run)
  expect(run.failed).toEqual([])
  expect(run.phase).toBe('done')
  expect(run.dims).toEqual(FIXTURES.map(() => 768))

  for (const [q, want] of QUERIES) {
    const top = await page.evaluate(async (q) => {
      const d = window.__debug
      await d.searchImages(q)
      return Array.from(d.state.searchResults!.slice(0, 2), (i) => d.state.files[i].name)
    }, q)
    console.log(`search "${q}" ->`, top)
    expect(top[0], `top hit for "${q}"`).toContain(`e2e/${want}`)
  }
})

test('overnight embedding runs the real model in recycled worker frames', async ({ page }) => {
  page.on('pageerror', (e) => console.log('[pageerror]', e.message))
  await page.addInitScript((fixtures) => {
    // Viewer mode keeps the visible page from loading a model of its own.
    if (window === window.top)
      localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true, modelVariant: 'embeddinggemma2', randomSampleSize: 0 }))
    const w = window as any
    w.__mcOvernightTest = { chunkSize: 4 } // one frame each for the photos, the quick videos and the bonus videos
    w.showDirectoryPicker = async () => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('overnight-gpu', { create: true })
      for (const n of fixtures) {
        const h = await dir.getFileHandle(n, { create: true })
        if ((await h.getFile()).size === 0) {
          const wr = await h.createWritable()
          await wr.write(await (await fetch(`/e2e/fixtures/${n}`)).blob())
          await wr.close()
        }
      }
      return dir
    }
  }, FIXTURES)
  await page.goto('/')
  await page.locator('#overnight-btn').click()
  await page.locator('#filter-load').click()
  for (const box of await page.locator('#overnight-models input:checked').all()) await box.uncheck()
  await page.locator('#overnight-models input[value="embeddinggemma2"]').check()
  await page.locator('#overnight-start').click()
  await expect(page.locator('#overnight-close')).toBeVisible({ timeout: 20 * 60_000 })
  console.log(await page.locator('#overnight-run').innerText())
  // Photos, then the videos from 2 frames, then the bonus pass over those videos from all their frames.
  const rows = page.locator('.on-model')
  await expect(rows).toHaveCount(3)
  for (const [i, n] of [4, 2, 2].entries()) await expect(rows.nth(i)).toContainText(`embedded ${n} `)

  const vecs = await page.evaluate(async () => {
    const db: IDBDatabase = await new Promise((res, rej) => {
      const r = indexedDB.open('photo-organizer-v1', 2)
      r.onsuccess = () => res(r.result)
      r.onerror = () => rej(r.error)
    })
    const store = db.transaction('embeddings').objectStore('embeddings')
    const all: { key: string; v: Float32Array }[] = await new Promise((res) => {
      const out: { key: string; v: Float32Array }[] = []
      const r = store.openCursor()
      r.onsuccess = () => {
        const c = r.result
        if (!c) return res(out)
        out.push({ key: String(c.key), v: c.value })
        c.continue()
      }
    })
    return all
      .filter((e) => e.key.startsWith('@embeddinggemma2'))
      .map((e) => ({ key: e.key, len: e.v.length, owns: e.v.byteLength === e.v.buffer.byteLength, norm: Math.hypot(...e.v) }))
  })
  expect(vecs).toHaveLength(8) // 4 photos, 2 videos from 2 frames, the same 2 from all their frames
  expect(vecs.filter((v) => v.key.includes('-f2/'))).toHaveLength(2)
  for (const v of vecs) {
    expect(v).toMatchObject({ len: 768, owns: true })
    expect(v.norm, `vector ${v.key}`).toBeGreaterThan(0.99)
  }
})
