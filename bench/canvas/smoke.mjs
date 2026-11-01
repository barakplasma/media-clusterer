// Headless smoke test of canvas-bench.html: node bench/canvas/smoke.mjs (CHROMIUM_PATH=...)
import { createServer } from 'vite'
import { chromium } from 'playwright'
const server = await createServer({ server: { port: 5197 }, logLevel: 'error' }); await server.listen()
const b = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const p = await b.newPage({ viewport: { width: 390, height: 800 } })
p.on('pageerror', (e) => { console.error('pageerror', e.message); process.exitCode = 1 })
await p.goto('http://localhost:5197/canvas-bench.html')
await p.uncheck('input[value="20000"]'); await p.check('input[value="5000"]'); await p.uncheck('#flush')
await p.click('#run')
await p.waitForFunction(() => document.getElementById('status').textContent.startsWith('done'), null, { timeout: 120000 })
const rows = await p.$$eval('#out tr', (t) => t.length - 1)
console.log('rows', rows, await p.textContent('#info'))
if (rows !== 6) process.exitCode = 1
await b.close(); await server.close()
