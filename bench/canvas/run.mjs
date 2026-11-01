// node bench/canvas/run.mjs  (CHROMIUM_PATH=/opt/pw-browsers/chromium). Fresh page per run, 2 rounds, median.
import { createServer } from 'vite'
import { chromium } from 'playwright'
const server = await createServer({ root: process.cwd(), server: { port: 5198 }, logLevel: 'error' })
await server.listen()
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const names = process.env.NAMES ? process.env.NAMES.split(',') : ['logic-baseline', 'logic-grid', 'baseline-main', 'opt2d-main', 'opt2d-worker', 'opt2d-all', 'pixi-main', 'pixi-all']
const FLUSH = process.env.FLUSH === '1' // 1 = include forced raster (software here: not representative of a GPU)
const sizes = (process.env.SIZES || '5000,20000,50000').split(',').map(Number)
const ROUNDS = +(process.env.ROUNDS || 2)
const rows = []
for (const n of sizes) for (const name of names) {
  const runs = []
  for (let r = 0; r < ROUNDS; r++) {
    const page = await browser.newPage({ viewport: { width: 1300, height: 800 } })
    page.on('pageerror', (e) => console.error('pageerror', name, n, e.message))
    await page.goto('http://localhost:5198/bench/canvas/index.html')
    await page.waitForFunction(() => window.bench)
    runs.push(await page.evaluate(([a, b, c]) => window.bench(a, b, c), [name, n, FLUSH]))
    await page.close()
  }
  const med = (k) => runs.map((x) => x[k]).sort((a, b) => a - b)[Math.floor(runs.length / 2)]
  rows.push({ n, renderer: name, p50ms: +med('p50').toFixed(2), p95ms: +med('p95').toFixed(2), maxMs: +med('max').toFixed(1), fps: Math.round(1000 / med('p50')), drawn: med('drawn'), mainLateP95: +med('mainLateP95').toFixed(1), mainLateMax: +med('mainLateMax').toFixed(1) })
  console.log(JSON.stringify(rows.at(-1)))
}
console.table(rows)
await browser.close(); await server.close()
