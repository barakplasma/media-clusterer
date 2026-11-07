import { test, expect } from '@playwright/test'

// Faces against OpenCV's own reference: in the SFace demo image the face on the left matches only
// the right-most of the six people on the right (OpenCV scores it 0.79, the others 0.1–0.3).
// Downloads the two models (~10 MB) and the image from Hugging Face.
test('faces: YuNet + SFace reproduce the OpenCV zoo reference match', async ({ page }) => {
  test.setTimeout(180_000)
  await page.goto('/device-test.html') // any page of the app: it only needs the dev server's module graph
  const r = await page.evaluate(async () => {
    const faces = await import('/src/faces.ts' as string)
    const m = await faces.loadFaceModels()
    const img = await createImageBitmap(await (await fetch('https://huggingface.co/opencv/face_recognition_sface/resolve/main/example_outputs/demo.jpg')).blob())
    const half = async (x: number) => faces.findFaces(m, await createImageBitmap(img, x, 0, 512, 512, { resizeWidth: 1024, resizeHeight: 1024 }))
    const [left, right] = [await half(0), await half(512)]
    const S = faces.FACE_STRIDE
    const vec = (p: Float32Array, i: number) => p.subarray(i * S + 5, (i + 1) * S)
    const n = right.length / S
    const q = vec(left, 0)
    const sims = Array.from({ length: n }, (_, i) => {
      const v = vec(right, i)
      return { x: right[i * S], cos: v.reduce((s: number, a: number, d: number) => s + a * q[d], 0) }
    }).sort((a, b) => a.x - b.x)
    return { leftFaces: left.length / S, sims, same: faces.SAME_PERSON }
  })
  console.log(JSON.stringify(r))
  expect(r.leftFaces).toBe(1)
  expect(r.sims.length).toBeGreaterThanOrEqual(5)
  const rightMost = r.sims.at(-1)!
  expect(rightMost.cos).toBeGreaterThan(r.same)
  for (const s of r.sims.slice(0, -1)) expect(s.cos).toBeLessThan(r.same)
})

test('faces: opening a folder finds the same person in two photos and the People strip highlights them', async ({ page }) => {
  test.setTimeout(180_000)
  await page.addInitScript(() => {
    if (window === window.top && !sessionStorage.getItem('mc_test_seeded')) {
      sessionStorage.setItem('mc_test_seeded', '1')
      localStorage.setItem('mc_settings', JSON.stringify({ viewerOnly: true, findFaces: true, randomSampleSize: 0 }))
    }
    // A folder with the two halves of the reference image: the man on the left is also the right-most person on the right.
    ;(window as any).showDirectoryPicker = async () => {
      const root = await navigator.storage.getDirectory()
      const dir = await root.getDirectoryHandle('faces-fixtures', { create: true })
      const img = await createImageBitmap(await (await fetch('https://huggingface.co/opencv/face_recognition_sface/resolve/main/example_outputs/demo.jpg')).blob())
      for (const [name, x] of [['portrait.jpg', 0], ['group.jpg', 512]] as const) {
        const c = new OffscreenCanvas(1024, 1024)
        c.getContext('2d')!.drawImage(img, x, 0, 512, 512, 0, 0, 1024, 1024)
        const w = await (await dir.getFileHandle(name, { create: true })).createWritable()
        await w.write(await c.convertToBlob({ type: 'image/jpeg', quality: 0.95 }))
        await w.close()
      }
      return dir
    }
  })
  await page.goto('/')
  await page.locator('#open-btn').click()
  await page.locator('#filter-load').click()
  const chip = page.locator('#people-strip button')
  await expect(chip).toHaveCount(1, { timeout: 120_000 })
  await expect(chip).toHaveAttribute('aria-label', 'Person 1, 2 photos')
  await chip.click()
  await expect(chip).toHaveAttribute('aria-pressed', 'true')
  // Cached: a second open finds the same person without running the models again.
  const keys = await page.evaluate(async () => {
    const db: IDBDatabase = await new Promise((res) => {
      const r = indexedDB.open('photo-organizer-v1', 2)
      r.onsuccess = () => res(r.result)
    })
    return new Promise<string[]>((res) => {
      const r = db.transaction('embeddings').objectStore('embeddings').getAllKeys()
      r.onsuccess = () => res(r.result.map(String).filter((k) => k.startsWith('@faces/')))
    })
  })
  expect(keys).toHaveLength(2)
})
