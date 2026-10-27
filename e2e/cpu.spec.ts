import { test, expect } from '@playwright/test'

// The two checks that need no GPU. Both hide WebGPU, so they behave the same
// on a Mac runner, a Linux deploy job or a laptop.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    if (navigator.gpu) navigator.gpu.requestAdapter = async () => null
  })
})

test('viewer mode lays out photos without loading a model', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true })))
  await page.goto('/')
  const r = await page.evaluate(async () => {
    const d = window.__debug
    const names = ['cats.jpg', 'tiger.png', 'football.webp', 'butterfly.jpg']
    const files = await Promise.all(
      names.map(async (n, i) => {
        const blob = await (await fetch(`/e2e/fixtures/${n}`)).blob()
        const file = new File([blob], n, { type: blob.type, lastModified: 1e12 + i * 1000 })
        return { name: `album/${n}`, size: file.size, lastModified: file.lastModified, file, objectURL: null }
      })
    )
    await d.processFiles(files)
    return { phase: d.state.phase, points: d.state.points.length, vectors: d.state.vectors.length }
  })
  expect(r).toEqual({ phase: 'done', points: 4, vectors: 0 })
  await expect(page.locator('#search-input')).toBeDisabled()

  // R jumps to a random photo; E goes back to the one before it.
  await page.keyboard.press('r')
  const first = await page.evaluate(() => window.__debug.state.activeFileIndex)
  await page.keyboard.press('r')
  await page.keyboard.press('e')
  expect(await page.evaluate(() => window.__debug.state.activeFileIndex)).toBe(first)
})

test('a saved WebGPU-only model falls back when there is no adapter', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('mc_settings', JSON.stringify({ modelVariant: 'embeddinggemma2' })))
  await page.goto('/')
  // Chrome hides blocklisted GPUs; the modal says how to turn WebGPU on.
  await expect(page.locator('#webgpu-help-modal')).toBeVisible()
  await expect(page.locator('#webgpu-help-url')).toHaveText('chrome://flags/#enable-unsafe-webgpu')
  expect(await page.evaluate(() => window.__debug.state.settings.modelVariant)).toBe('nomic')
  // The saved choice survives a settings write.
  await page.locator('#density-slider').dispatchEvent('input')
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('mc_settings')!).modelVariant)).toBe('embeddinggemma2')
})
