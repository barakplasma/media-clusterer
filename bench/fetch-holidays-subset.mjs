// Download the benchmark's Holidays subset (bench/data/holidays-subset.json:
// the first 100 scenes, 274 photos) from the Kaggle mirror, which needs no
// login. Writes images/, groundtruth.json and images/list.json in the layout
// embed.mjs, metrics.mjs and chrome-ai-bench.html read.
//
//   node bench/fetch-holidays-subset.mjs --out data/holidays
//
// The zip is 2.8 GB; only the 274 images are kept and the zip is deleted.

import { execFileSync } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { parseArgs } from 'node:util'

const { values: args } = parseArgs({ options: { out: { type: 'string', default: 'data/holidays' } } })
const { scenes } = JSON.parse(await readFile(new URL('./data/holidays-subset.json', import.meta.url), 'utf8'))
const names = Object.values(scenes).flatMap(({ query, similar }) => [query, ...similar])

await mkdir(join(args.out, 'images'), { recursive: true })
const zip = join(args.out, 'holidays.zip')
const res = await fetch('https://www.kaggle.com/api/v1/datasets/download/vadimshabashov/inria-holidays')
if (!res.ok) throw new Error(`Kaggle download: HTTP ${res.status}`)
await pipeline(Readable.fromWeb(res.body), createWriteStream(zip))
// -j drops the zip's images/ folder; -o overwrites a previous run.
execFileSync('unzip', ['-q', '-o', '-j', zip, ...names.map((n) => `images/${n}`), '-d', join(args.out, 'images')], { stdio: 'inherit' })
await rm(zip)

await writeFile(join(args.out, 'groundtruth.json'), JSON.stringify(scenes))
await writeFile(join(args.out, 'images', 'list.json'), JSON.stringify(names))
console.log(`${names.length} images in ${args.out}`)
