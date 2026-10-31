// Usage: node bench/ui/run.mjs   (starts vite dev, runs both impls in fresh pages, alternating, 3 rounds)
import { createServer } from 'vite'
import { chromium } from 'playwright'
const server = await createServer({ server: { port: 5199 }, logLevel: 'error' })
await server.listen()
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] })
const out = { vanilla: [], solid: [] }
for (let r = 0; r < 3; r++) for (const name of r % 2 ? ['solid', 'vanilla'] : ['vanilla', 'solid']) {
  const page = await browser.newPage()
  await page.goto('http://localhost:5199/bench/ui/index.html')
  await page.waitForFunction(() => (window).bench)
  out[name].push(await page.evaluate((n) => window.bench(n), name))
  await page.close()
}
const med = (a, k) => a.map((x) => x[k]).sort((x, y) => x - y)[Math.floor(a.length / 2)]
for (const n of ['vanilla', 'solid']) console.log(n, Object.fromEntries(['syncMs50k', 'toastMs5k', 'p50', 'p95', 'max', 'frames', 'updates'].map((k) => [k, +med(out[n], k).toFixed(2)])))
await browser.close(); await server.close()
