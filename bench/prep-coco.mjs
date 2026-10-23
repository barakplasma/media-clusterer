// Build a category-labelled, captioned subset of COCO val2017 for the
// "does the model understand what is in the photo" half of the benchmark.
//
//   node bench/prep-coco.mjs --coco <dir with val2017/ and annotations/> --out data/coco
//
// Keeps images with one clear subject (the largest object covers >= 15% of
// the frame; categories with fewer than 5 such images are dropped), at most
// --per images per category, so every category counts
// about equally. Writes <out>/images/ (symlinks), <out>/labels.json
// ({ name: category }) and <out>/captions.json ([{ name, caption }], one
// caption per image: the first COCO lists).

import { mkdir, readFile, symlink, writeFile, rm } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'

const { values: args } = parseArgs({
  options: {
    coco: { type: 'string' },
    out: { type: 'string', default: 'data/coco' },
    per: { type: 'string', default: '6' },
    minArea: { type: 'string', default: '0.15' },
    minPer: { type: 'string', default: '5' }
  }
})

const ann = JSON.parse(await readFile(join(args.coco, 'annotations/instances_val2017.json'), 'utf8'))
const caps = JSON.parse(await readFile(join(args.coco, 'annotations/captions_val2017.json'), 'utf8'))
const catName = new Map(ann.categories.map((c) => [c.id, c.name]))
const images = new Map(ann.images.map((im) => [im.id, im]))

// Largest object per image.
const largest = new Map()
for (const a of ann.annotations) {
  if (a.iscrowd) continue
  const cur = largest.get(a.image_id)
  if (!cur || a.area > cur.area) largest.set(a.image_id, a)
}

const byCat = new Map()
for (const [imageId, a] of largest) {
  const im = images.get(imageId)
  if (a.area / (im.width * im.height) < Number(args.minArea)) continue
  const list = byCat.get(a.category_id) ?? []
  list.push(imageId)
  byCat.set(a.category_id, list)
}

const firstCaption = new Map()
for (const c of caps.annotations) if (!firstCaption.has(c.image_id)) firstCaption.set(c.image_id, c.caption.trim())

const per = Number(args.per)
const chosen = []
for (const [catId, ids] of [...byCat].sort((a, b) => a[0] - b[0])) {
  // A category with only a couple of images makes retrieval for it noise.
  if (ids.length < Number(args.minPer)) continue
  for (const id of ids.sort((a, b) => a - b).slice(0, per)) chosen.push([id, catName.get(catId)])
}

await rm(args.out, { recursive: true, force: true })
await mkdir(join(args.out, 'images'), { recursive: true })
const labels = {}
const captions = []
for (const [id, cat] of chosen) {
  const name = images.get(id).file_name
  await symlink(resolve(args.coco, 'val2017', name), join(args.out, 'images', name))
  labels[name] = cat
  captions.push({ name, caption: firstCaption.get(id) })
}
await writeFile(join(args.out, 'labels.json'), JSON.stringify(labels, null, 1))
await writeFile(join(args.out, 'captions.json'), JSON.stringify(captions, null, 1))
const counts = Object.values(labels).reduce((m, c) => m.set(c, (m.get(c) ?? 0) + 1), new Map())
console.log(`${chosen.length} images over ${counts.size} categories (min ${Math.min(...counts.values())}, max ${Math.max(...counts.values())})`)
