/**
 * Benchmark harness for the Chrome AI model (bench/README.md): describe each
 * image with Gemini Nano exactly as the app does, so bench/embed.mjs can embed
 * the descriptions and score them next to the other models.
 *
 * chrome-ai-bench.html?list=<url of a JSON array of image names>&base=<url prefix>
 *
 * Results land in window.__benchResult = { availability, items: [{ name,
 * description | error, ms }] } and window.__benchDone = true.
 *
 * If Gemini Nano still has to download, the download needs a user gesture:
 * the page waits for window.__benchStart(), which bench/run-chrome-ai.mjs
 * calls through DevTools with userGesture: true.
 */

import { ChromeAISessionManager, DEFAULT_DESCRIBE_PROMPT, getChromeAIAvailability } from './chromeAI'

declare const LanguageModel: {
  create(options: {
    expectedInputs: Array<{ type: string; languages?: string[] }>
    monitor?: (m: EventTarget) => void
  }): Promise<{ destroy(): void }>
}

const logEl = document.getElementById('log')!
const log = (msg: string) => {
  console.log(`CHROMEAI ${msg}`)
  logEl.textContent += msg + '\n'
}

interface Item {
  name: string
  description?: string
  error?: string
  ms: number
}
const w = window as Window & { __benchResult?: unknown; __benchDone?: boolean; __benchStart?: () => Promise<void> }

async function run(): Promise<void> {
  const params = new URLSearchParams(location.search)
  const availability = await getChromeAIAvailability()
  log(`availability ${availability} ua=${navigator.userAgent}`)
  const items: Item[] = []
  w.__benchResult = { availability, items }
  if (availability === 'unavailable') return
  if (availability !== 'available') {
    // Download Gemini Nano first; create() only starts it inside a user gesture.
    await new Promise<void>((resolve) => {
      w.__benchStart = async () => {
        const t0 = performance.now()
        const session = await LanguageModel.create({
          expectedInputs: [{ type: 'image' }, { type: 'text', languages: ['en'] }],
          monitor: (m) =>
            m.addEventListener('downloadprogress', (e) => log(`download ${Math.round((e as ProgressEvent).loaded * 100)}%`))
        })
        session.destroy()
        log(`model ready in ${((performance.now() - t0) / 1000).toFixed(0)}s`)
        resolve()
      }
      log('waiting for __benchStart() (needs a user gesture)')
    })
  }

  const names: string[] = await (await fetch(params.get('list') ?? 'bench-images/list.json')).json()
  const base = params.get('base') ?? 'bench-images/'
  const manager = new ChromeAISessionManager()
  for (const [i, name] of names.entries()) {
    const t0 = performance.now()
    try {
      const blob = await (await fetch(base + name)).blob()
      // src/app.ts embedAll (chrome-ai): a 128 px wide bitmap, the settings' prompt.
      const bitmap = await createImageBitmap(blob, { resizeWidth: 128, resizeQuality: 'medium' })
      items.push({ name, description: await manager.describe(bitmap, DEFAULT_DESCRIBE_PROMPT), ms: performance.now() - t0 })
    } catch (err) {
      items.push({ name, error: String((err as Error)?.message ?? err), ms: performance.now() - t0 })
    }
    if ((i + 1) % 25 === 0) log(`${i + 1}/${names.length}`)
  }
  manager.destroy()
  log(`done ${items.filter((x) => x.description).length}/${names.length} described`)
}

run()
  .catch((err) => log(`FAIL ${(err as Error)?.message ?? err}`))
  .finally(() => {
    w.__benchDone = true
  })
