// Download the benchmark's COCO subset (bench/data/coco-subset.json) without
// the full 1 GB of COCO: just its ~870 images, plus labels.json, captions.json
// and list.json in the layout embed.mjs, metrics.mjs and chrome-ai-bench.html read.
//
//   node bench/fetch-coco-subset.mjs --out data/coco
//
// Then, for the Chrome AI run, serve the folder as dist/bench-images/.

import { mkdir, readFile, writeFile, access } from 'node:fs/promises'
import { join } from 'node:path'
import { parseArgs } from 'node:util'

const { values: args } = parseArgs({ options: { out: { type: 'string', default: 'data/coco' } } })
const { items } = JSON.parse(await readFile(new URL('./data/coco-subset.json', import.meta.url), 'utf8'))
await mkdir(join(args.out, 'images'), { recursive: true })
let fetched = 0
for (const { name } of items) {
  const file = join(args.out, 'images', name)
  if (await access(file).then(() => true, () => false)) continue
  const res = await fetch(`http://images.cocodataset.org/val2017/${name}`)
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`)
  await writeFile(file, Buffer.from(await res.arrayBuffer()))
  if (++fetched % 100 === 0) console.log(`${fetched} images`)
}
await writeFile(join(args.out, 'labels.json'), JSON.stringify(Object.fromEntries(items.map((i) => [i.name, i.category]))))
await writeFile(join(args.out, 'captions.json'), JSON.stringify(items.map(({ name, caption }) => ({ name, caption }))))
await writeFile(join(args.out, 'images', 'list.json'), JSON.stringify(items.map((i) => i.name)))
console.log(`${items.length} images in ${args.out}`)
