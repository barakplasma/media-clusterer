// Throwaway probe: can Chrome's built-in Gemini Nano (Prompt API) run on this runner?
import { chromium } from 'playwright-core'
import http from 'node:http'
import os from 'node:os'
import { execSync } from 'node:child_process'
import fs from 'node:fs'

const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a)
log('host', os.platform(), os.arch(), `${os.cpus().length} cpus`, `${(os.totalmem() / 2 ** 30).toFixed(1)} GB RAM`)
try { log('disk', execSync(os.platform() === 'win32' ? 'powershell -c "Get-PSDrive C,D -ErrorAction SilentlyContinue | ft Name,Free"' : 'df -h / /mnt 2>/dev/null').toString().trim()) } catch {}

const server = http.createServer((_, res) => { res.setHeader('content-type', 'text/html'); res.end('<!doctype html><button id=go>go</button>') })
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const url = `http://127.0.0.1:${server.address().port}/`

const profile = fs.mkdtempSync(os.tmpdir() + '/nano-')
const ctx = await chromium.launchPersistentContext(profile, {
  channel: 'chrome',
  headless: false,
  // Playwright's defaults switch off component updates and background
  // networking, which is how Chrome fetches the on-device model.
  ignoreDefaultArgs: ['--disable-component-update', '--disable-background-networking'],
  args: [
    '--enable-features=PromptAPIForGeminiNano,PromptAPIForGeminiNanoMultimodalInput,OptimizationGuideOnDeviceModel:BypassPerfRequirement/true',
  ],
})
const page = ctx.pages()[0] ?? (await ctx.newPage())
page.on('console', (m) => log('[page]', m.text().slice(0, 300)))
await page.goto(url)
log('chrome', ctx.browser()?.version() ?? '(persistent)', await page.evaluate(() => navigator.userAgent))

const opts = { expectedInputs: [{ type: 'image' }, { type: 'text', languages: ['en'] }] }
const avail = () => page.evaluate(async (o) => {
  if (!('LanguageModel' in self)) return 'no LanguageModel API'
  try { return { text: await LanguageModel.availability(), image: await LanguageModel.availability(o) } } catch (e) { return 'error ' + e.message }
}, opts)
let a = await avail()
log('availability', JSON.stringify(a))

if (typeof a === 'object' && [a.text, a.image].some((x) => x === 'downloadable' || x === 'downloading')) {
  // create() needs a user gesture to start the download.
  await page.evaluate((o) => {
    window.__dl = []
    document.getElementById('go').onclick = () => {
      window.__created = LanguageModel.create({ ...o, monitor(m) { m.addEventListener('downloadprogress', (e) => window.__dl.push(e.loaded)) } })
        .then((s) => { window.__session = s; return 'created' }, (e) => 'create failed: ' + e.message)
    }
  }, opts)
  await page.click('#go')
  const end = Date.now() + 30 * 60_000
  for (;;) {
    const st = await page.evaluate(async () => ({ last: window.__dl.at(-1) ?? null, done: !!window.__session, err: await Promise.race([window.__created, new Promise((r) => setTimeout(() => r(null), 10))]) }))
    log('download', JSON.stringify(st))
    if (st.done || (st.err && st.err !== 'created')) break
    if (Date.now() > end) { log('download timed out after 30 min'); break }
    await page.waitForTimeout(30_000)
  }
  a = await avail()
  log('availability after download', JSON.stringify(a))
}

if (typeof a === 'object' && a.image === 'available') {
  const out = await page.evaluate(async (o) => {
    const s = window.__session ?? (await LanguageModel.create(o))
    const c = new OffscreenCanvas(128, 128), g = c.getContext('2d')
    g.fillStyle = 'red'; g.fillRect(0, 0, 128, 128); g.fillStyle = 'white'; g.beginPath(); g.arc(64, 64, 40, 0, 7); g.fill()
    const t0 = performance.now()
    const r = await s.prompt([{ role: 'user', content: [{ type: 'text', value: 'Describe this image in one short sentence.' }, { type: 'image', value: await createImageBitmap(c) }] }])
    return { reply: r, ms: Math.round(performance.now() - t0) }
  }, opts)
  log('PROMPT', JSON.stringify(out))
  log('RESULT: Gemini Nano runs here')
} else {
  log('RESULT: Gemini Nano not usable here:', JSON.stringify(a))
}
await ctx.close()
server.close()
