// Real Safari (safaridriver) smoke test for EmbeddingGemma 2 on WebGPU.
// Chrome carries the full suite (e2e/*.spec.ts); this checks the same path
// works in Safari, split into steps so a failure names the stage.
// Needs `sudo safaridriver --enable` and the app served at BASE_URL.
import { test, before, after } from 'node:test'
import assert from 'node:assert/strict'
import { Builder } from 'selenium-webdriver'

const BASE_URL = process.env.BASE_URL ?? 'http://127.0.0.1:5173'
const FIXTURES = ['cats.jpg', 'tiger.png', 'football.webp', 'cats.mp4']
let driver

// safaridriver caps a single script call, so start work in the page and poll for it.
async function run(fn, arg, timeoutMs) {
  await driver.executeScript(
    `window.__job = undefined; (${fn})(arguments[0]).then(r => window.__job = { r }, e => window.__job = { e: String(e && e.stack || e) })`,
    arg
  )
  const end = Date.now() + timeoutMs
  for (;;) {
    const job = await driver.executeScript('return window.__job')
    if (job) {
      if (job.e) throw new Error(job.e)
      return job.r
    }
    if (Date.now() > end) throw new Error(`timed out after ${timeoutMs / 1000}s`)
    await new Promise((r) => setTimeout(r, 2000))
  }
}

before(async () => {
  driver = await new Builder().forBrowser('safari').build()
  await driver.manage().setTimeouts({ script: 60_000 })
  await driver.get(BASE_URL)
  await driver.executeScript(
    "localStorage.setItem('mc_settings', JSON.stringify({ modelVariant: 'embeddinggemma2', enableTextSearch: true }))"
  )
  await driver.navigate().refresh()
  console.log('user agent:', await driver.executeScript('return navigator.userAgent'))
})
after(() => driver?.quit())

test('Safari exposes a hardware WebGPU adapter', async () => {
  const a = await run(async () => {
    const a = await navigator.gpu?.requestAdapter()
    return a && { vendor: a.info?.vendor, fallback: a.info?.isFallbackAdapter ?? a.isFallbackAdapter ?? null }
  }, null, 30_000)
  console.log('adapter', a)
  assert.ok(a, 'navigator.gpu.requestAdapter() returned nothing')
  assert.notEqual(a.fallback, true, 'software WebGPU adapter')
})

test('model loads', async () => {
  const t0 = Date.now()
  const phase = await run(
    () =>
      new Promise((res) => {
        const tick = () => {
          const p = window.__debug?.state.phase
          if (p === 'model_ready' || (p === 'idle' && document.getElementById('load-model-btn')?.hidden === false))
            res({ phase: p, status: document.getElementById('status')?.textContent })
          else setTimeout(tick, 1000)
        }
        tick()
      }),
    null,
    20 * 60_000
  )
  console.log(`model: ${JSON.stringify(phase)} after ${((Date.now() - t0) / 1000).toFixed(0)}s`)
  assert.equal(phase.phase, 'model_ready')
})

test('embeds photos and a video', async () => {
  const r = await run(
    async (names) => {
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
    },
    FIXTURES,
    20 * 60_000
  )
  console.log('embed', r)
  assert.deepEqual(r.failed, [])
  assert.equal(r.phase, 'done')
  assert.deepEqual(r.dims, FIXTURES.map(() => 768))
})

test('text search ranks the right file first', async () => {
  for (const [q, want] of [['a tiger', 'tiger'], ['people playing soccer', 'football']]) {
    const top = await run(
      async (q) => {
        const d = window.__debug
        await d.searchImages(q)
        return Array.from(d.state.searchResults.slice(0, 2), (i) => d.state.files[i].name)
      },
      q,
      5 * 60_000
    )
    console.log(`search "${q}" ->`, top)
    assert.ok(top[0].startsWith(`e2e/${want}`), `top hit for "${q}" was ${top[0]}`)
  }
})
