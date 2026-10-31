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
    if (webnn) localStorage.setItem('mc_webnn', '1')
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
