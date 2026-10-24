// Drive real Chrome (not Playwright's Chromium: Gemini Nano ships only with
// Chrome) through DevTools to run chrome-ai-bench.html and save its results.
//
//   node bench/run-chrome-ai.mjs --url 'http://127.0.0.1:4173/chrome-ai-bench.html' --out chrome-ai.json \
//     [--chrome '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'] [--timeout-min 90]
//
// Logs the page's CHROMEAI lines. Exits 0 with { availability: 'unavailable' }
// written when this machine cannot run Gemini Nano, so a CI job can report
// that instead of failing.

import { spawn } from 'node:child_process'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseArgs } from 'node:util'

const { values: args } = parseArgs({
  options: {
    url: { type: 'string' },
    out: { type: 'string', default: 'chrome-ai.json' },
    chrome: { type: 'string', default: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' },
    'timeout-min': { type: 'string', default: '90' },
    port: { type: 'string', default: '9333' },
    // Extra flags, e.g. --chrome-args='--headless=new --no-sandbox' on a Linux box without a display.
    'chrome-args': { type: 'string', default: '' }
  }
})

const profile = await mkdtemp(join(tmpdir(), 'chrome-ai-'))
const chrome = spawn(
  args.chrome,
  [
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${args.port}`,
    '--no-first-run',
    '--no-default-browser-check',
    // The Prompt API with image input, and skip the device performance-class
    // gate (chrome://flags "Enabled BypassPerfRequirement").
    '--enable-features=PromptAPIForGeminiNano,PromptAPIForGeminiNanoMultimodalInput,OptimizationGuideOnDeviceModel:BypassPerfRequirement/true',
    ...args['chrome-args'].split(' ').filter(Boolean),
    'about:blank'
  ],
  { stdio: 'ignore' }
)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let target
for (let i = 0; i < 60 && !target; i++) {
  await sleep(500)
  target = await fetch(`http://127.0.0.1:${args.port}/json/list`)
    .then((r) => r.json())
    .then((l) => l.find((t) => t.type === 'page'))
    .catch(() => null)
}
if (!target) throw new Error('Chrome did not start with remote debugging')

const ws = new WebSocket(target.webSocketDebuggerUrl)
await new Promise((r) => (ws.onopen = r))
let id = 0
const pending = new Map()
ws.onmessage = (e) => {
  const m = JSON.parse(e.data)
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m)
    pending.delete(m.id)
  } else if (m.method === 'Runtime.consoleAPICalled') {
    const text = m.params.args.map((a) => a.value ?? a.description ?? '').join(' ')
    if (text.startsWith('CHROMEAI')) console.log(text)
  }
}
const call = (method, params = {}) =>
  new Promise((resolve) => {
    pending.set(++id, resolve)
    ws.send(JSON.stringify({ id, method, params }))
  })
const evaluate = async (expression, opts = {}) =>
  (await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true, ...opts })).result?.result?.value

await call('Runtime.enable')
await call('Page.navigate', { url: args.url })

const deadline = Date.now() + Number(args['timeout-min']) * 60_000
let started = false
while (Date.now() < deadline) {
  await sleep(2000)
  if (await evaluate('!!window.__benchDone')) break
  if (!started && (await evaluate('typeof window.__benchStart === "function"'))) {
    started = true
    // A real user gesture as far as Chrome is concerned: starts the download.
    // Don't await it: a stalled download must not hold the loop past its deadline.
    await evaluate('window.__benchStart()', { userGesture: true, awaitPromise: false })
  }
}
const result = (await evaluate('JSON.stringify(window.__benchResult ?? null)')) ?? 'null'
await writeFile(args.out, result)
const parsed = JSON.parse(result)
console.log(
  `availability ${parsed?.availability}; described ${parsed?.items?.filter((i) => i.description).length ?? 0}` +
    ` of ${parsed?.items?.length ?? 0}; wrote ${args.out}`
)
ws.close()
chrome.kill()
