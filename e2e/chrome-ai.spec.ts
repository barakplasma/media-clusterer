import { test, expect, chromium, type Page } from '@playwright/test'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

// The Chrome AI model: Gemini Nano describes each photo, nomic-embed-text
// embeds the description, and text search runs against those vectors.
const FIXTURES = ['cats.jpg', 'tiger.png', 'football.webp', 'butterfly.jpg']
const QUERIES: [string, string][] = [
  ['a tiger', 'tiger'],
  ['people playing soccer', 'football'],
]

let page: Page
test.beforeAll(async ({}, info) => {
  const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), 'nano-')), {
    channel: 'chrome',
    headless: false,
    baseURL: info.project.use.baseURL,
    // Playwright's defaults block the component download that fetches the model.
    ignoreDefaultArgs: ['--disable-component-update', '--disable-background-networking'],
    args: [
      '--enable-features=PromptAPIForGeminiNano,PromptAPIForGeminiNanoMultimodalInput,OptimizationGuideOnDeviceModel:BypassPerfRequirement/true',
    ],
  })
  page = ctx.pages()[0] ?? (await ctx.newPage())
})
test.afterAll(() => page?.context().close())

test('Chrome AI (Gemini Nano) describes photos and finds them by text', async () => {
  page.on('console', (m) => {
    if (!m.text().includes('[vite]')) console.log(`[${m.type()}] ${m.text().slice(0, 300)}`)
  })
  // First visit in viewer mode, so nothing loads while Gemini Nano downloads.
  await page.addInitScript(() => {
    if (!localStorage.getItem('mc_settings')) localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true }))
  })
  await page.goto('/')
  const opts = { expectedInputs: [{ type: 'image' }, { type: 'text', languages: ['en'] }], expectedOutputs: [{ type: 'text', languages: ['en'] }] }
  const availability = () => page.evaluate((o) => ('LanguageModel' in self ? LanguageModel.availability(o) : 'no API'), opts)
  let avail = await availability()
  console.log('Gemini Nano availability:', avail)
  expect(avail, 'Prompt API missing or this machine is ineligible').not.toMatch(/no API|unavailable/)

  if (avail !== 'available') {
    // create() starts the download and needs a user gesture.
    const t0 = Date.now()
    await page.evaluate((o) => {
      const b = document.createElement('button')
      b.id = 'e2e-nano'
      b.textContent = 'download'
      b.style.cssText = 'position:fixed;top:0;left:0;z-index:99999'
      b.onclick = () => void LanguageModel.create(o)
      document.body.append(b)
    }, opts)
    // The first-visit About dialog would swallow the click.
    await page.evaluate(() => document.querySelectorAll('dialog[open]').forEach((d) => (d as HTMLDialogElement).close()))
    await page.click('#e2e-nano')
    await expect.poll(availability, { timeout: 10 * 60_000, intervals: [5_000] }).toBe('available')
    console.log(`Gemini Nano downloaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`)
    avail = 'available'
  }

  await page.evaluate(() => localStorage.setItem('mc_settings', JSON.stringify({ modelVariant: 'chrome-ai', enableTextSearch: true })))
  await page.reload()
  // Log the app's status while it loads, so a stall says where it stalled.
  const loadEnd = Date.now() + 8 * 60_000
  let phase = ''
  while (Date.now() < loadEnd) {
    const st = await page.evaluate(() => ({ phase: window.__debug?.state.phase ?? '', status: document.getElementById('status')?.textContent ?? '' }))
    console.log(`app: ${st.phase} · ${st.status}`)
    phase = st.phase
    if (phase === 'model_ready' || phase === 'idle') break
    await page.waitForTimeout(15_000)
  }
  expect(phase, 'Chrome AI model did not load (see app status above)').toBe('model_ready')

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
      captions: d.state.files.map((f, i) => `${f.name}: ${d.state.captions[i]}`),
    }
  }, FIXTURES)
  console.log('Chrome AI run', run)
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
