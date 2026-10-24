// Turn benchmark results (bench/metrics.mjs --json) into src/benchmarkRanks.ts,
// the order the Settings lists use.
//
//   node bench/rank.mjs bench/results/holidays-274.json bench/results/coco-216.json > src/benchmarkRanks.ts
//
// Models: average rank across the image-similarity scores (each dataset's
// mAP, plus k-means NMI where the dataset has categories); ties go to the
// higher mean mAP. Projections: average rank across every model and every
// map score (trustworthiness@10, 2D mAP, 1-NN, 2D NMI). Average ranks keep
// one metric's scale from drowning the others.

import { readFile } from 'node:fs/promises'

// Benchmark ids -> app ModelVariant. Where the app has one setting for
// several benchmarked ones, the app's own setting is the one that counts.
const VARIANT = {
  'eg2-phone': 'embeddinggemma2', // phones get the phone profile...
  'eg2-desktop': null, // ...desktops the full budget; reported, ranked via eg2-phone (the worse of the two)
  smolvlm2: 'smolvlm2-vision',
  'nomic-mean': 'nomic', // src/embeddings.ts mean-pools
  'nomic-cls': null, // model-card pooling, for comparison only
  'sapiens2-fp16': 'sapiens2-fp16',
  'sapiens2-int8': 'sapiens2-int8',
  'chrome-ai': 'chrome-ai' // Gemini Nano descriptions, embedded with nomic-embed-text
}
// Never benchmarked here: a remote endpoint depends on whatever model it serves.
const UNRANKED_TAIL = ['openai']

const files = process.argv.slice(2)
const datasets = await Promise.all(files.map(async (f) => ({ file: f, rows: JSON.parse(await readFile(f, 'utf8')) })))

/** rank[i] = 1 for the best of `values` (higher is better), ties share the mean rank. */
function ranks(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
  const out = new Array(values.length)
  for (let i = 0; i < order.length; ) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    for (let k = i; k <= j; k++) out[order[k][1]] = (i + j) / 2 + 1
    i = j + 1
  }
  return out
}

// ── Models ──────────────────────────────────────────────────────────────────
const ids = Object.keys(VARIANT).filter((id) => VARIANT[id] && datasets.every((d) => d.rows.some((r) => r.model === id)))
const columns = []
const notes = new Map(ids.map((id) => [id, []]))
for (const d of datasets) {
  const name = d.file.replace(/^.*\//, '').replace(/\.json$/, '')
  const pick = (id, key) => d.rows.find((r) => r.model === id)?.[key]
  columns.push(ranks(ids.map((id) => pick(id, 'mAP'))))
  for (const id of ids) notes.get(id).push(`${name} mAP ${pick(id, 'mAP').toFixed(3)}`)
  if (ids.every((id) => pick(id, 'nmi') !== undefined)) {
    columns.push(ranks(ids.map((id) => pick(id, 'nmi'))))
    for (const id of ids) notes.get(id).push(`NMI ${pick(id, 'nmi').toFixed(3)}`)
  }
}
const meanMap = (id) =>
  datasets.reduce((t, d) => t + d.rows.find((r) => r.model === id).mAP, 0) / datasets.length
const avgRank = (id) => columns.reduce((t, c) => t + c[ids.indexOf(id)], 0) / columns.length
const modelOrder = [...ids].sort((a, b) => avgRank(a) - avgRank(b) || meanMap(b) - meanMap(a))
// Sapiens2's other precisions sit right after whichever one was run (same
// model, same quality within rounding).
const SAPIENS = ['sapiens2-fp16', 'sapiens2-int8', 'sapiens2-fp32']
const modelLines = []
for (const id of modelOrder) {
  modelLines.push(`  '${VARIANT[id]}', // avg rank ${avgRank(id).toFixed(2)}: ${notes.get(id).join(', ')}`)
  if (SAPIENS.includes(VARIANT[id])) {
    for (const v of SAPIENS) if (!modelOrder.some((m) => VARIANT[m] === v)) modelLines.push(`  '${v}', // not run: same model as ${VARIANT[id]}`)
  }
}
// Speed: seconds per photo, averaged over the datasets. Chrome AI's results
// time is only nomic embedding its descriptions; Gemini Nano writing them
// (CI runner, not in the results files) took ~14.5 s on COCO, 22-33 s on Holidays.
const EXTRA_S = { 'chrome-ai': 20 }
const secs = (id) =>
  datasets.reduce((t, d) => t + d.rows.find((r) => r.model === id).sPerImage, 0) / datasets.length + (EXTRA_S[id] ?? 0)
const speedOrder = [...ids].sort((a, b) => secs(a) - secs(b))
const badgeLines = modelOrder.map(
  // Quote only keys that need it, as prettier does.
  (id, i) => `  ${/^\w+$/.test(VARIANT[id]) ? VARIANT[id] : `'${VARIANT[id]}'`}: { speed: ${speedOrder.indexOf(id) + 1}, quality: ${i + 1} }, // ${secs(id).toFixed(2)} s/photo`
)

// Known app models with no results yet keep a place ahead of the never-run tail.
const ranked = new Set(modelLines.map((l) => l.match(/'([^']+)'/)[1]))
for (const v of [...new Set([...Object.values(VARIANT).filter(Boolean), 'sapiens2-fp32'])])
  if (!ranked.has(v)) modelLines.push(`  '${v}', // not benchmarked yet`)
for (const v of UNRANKED_TAIL) modelLines.push(`  '${v}', // not benchmarked`)

// ── Projections ─────────────────────────────────────────────────────────────
const methods = [...new Set(datasets.flatMap((d) => d.rows.flatMap((r) => Object.keys(r.projections ?? {}))))]
const projCols = []
for (const d of datasets) {
  for (const r of d.rows) {
    const p = r.projections ?? {}
    for (const key of ['trust10', 'mAP2d', 'nn1', 'nmi2d']) {
      // A projection that failed or skipped a score ranks last on it.
      const vals = methods.map((m) => (p[m] && !p[m].error && p[m][key] !== undefined ? p[m][key] : -Infinity))
      if (vals.some((v) => v !== -Infinity)) projCols.push(ranks(vals))
    }
  }
}
const projAvg = (m) => projCols.reduce((t, c) => t + c[methods.indexOf(m)], 0) / projCols.length
const projOrder = [...methods].sort((a, b) => projAvg(a) - projAvg(b))

console.log(`// Generated by bench/rank.mjs from ${files.map((f) => f.replace(/^.*\//, '')).join(', ')}.
// Do not edit by hand; re-run the benchmark (bench/README.md) and this script.

import type { ModelVariant, ProjectionMethod } from './types'

/** Best first by image-similarity quality. Unbenchmarked models go last. */
export const MODEL_RANK: readonly ModelVariant[] = [
${modelLines.join('\n').replace(/,( \/\/[^\n]*)?$/, '$1')}
]

/** Rank on the benchmark, 1 = best: quality as in MODEL_RANK, speed by seconds per photo. */
export const MODEL_BADGES: Partial<Record<ModelVariant, { speed: number; quality: number }>> = {
${badgeLines.join('\n').replace(/,( \/\/[^\n]*)?$/, '$1')}
}

/** Best first by how well the 2D map keeps real neighbours (average rank). */
export const PROJECTION_RANK: readonly ProjectionMethod[] = [
${projOrder.map((m) => `  '${m}', // avg rank ${projAvg(m).toFixed(2)}`).join('\n').replace(/,( \/\/[^\n]*)?$/, '$1')}
]`)
