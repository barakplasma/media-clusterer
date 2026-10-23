// Score cached embeddings (bench/embed.mjs) on INRIA Holidays, and score
// each 2D projection the app offers on how well it keeps those neighbours.
//
//   node bench/metrics.mjs --data <holidays dir> --emb <embed.mjs --out dir> \
//     [--models a,b] [--projections UMAP,TSNE,PCA] [--json results.json]
//
// Embedding: Holidays protocol (each scene's query ranks every other image;
// the scene's other images are relevant) -> mAP and Recall@1, plus an
// all-images-as-queries mAP, which is less noisy.
// Projection: druidjs with src/app.ts runProjection's settings, then
//   - trustworthiness@10: are 2D neighbours real neighbours (1 = perfect)
//   - 2D mAP: the Holidays protocol on 2D distances (do scenes stay together)
//   - 1-NN: share of images whose nearest 2D point is from the same scene

import { readFile, readdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import * as druid from '@saehrimnir/druidjs'

const { values: args } = parseArgs({
  options: {
    data: { type: 'string' },
    emb: { type: 'string' },
    models: { type: 'string' },
    projections: { type: 'string', default: 'UMAP,TSNE,PCA,ISOMAP,TriMap,MDS,LLE' },
    json: { type: 'string' }
  }
})

const gt = JSON.parse(await readFile(join(args.data, 'groundtruth.json'), 'utf8'))

async function loadEmbeddings(id) {
  const meta = JSON.parse(await readFile(join(args.emb, `${id}.json`), 'utf8'))
  const buf = await readFile(join(args.emb, `${id}.f32`))
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
  const rows = meta.names.map((_, i) => flat.subarray(i * meta.dim, (i + 1) * meta.dim))
  return { ...meta, rows }
}

// ── Ranking metrics over any similarity ─────────────────────────────────────

/** Average precision of `ranked` (indices, best first) against a relevant set. */
function averagePrecision(ranked, relevant) {
  let hits = 0
  let sum = 0
  for (let r = 0; r < ranked.length; r++) {
    if (relevant.has(ranked[r])) {
      hits++
      sum += hits / (r + 1)
      if (hits === relevant.size) break
    }
  }
  return relevant.size ? sum / relevant.size : 0
}

/**
 * Holidays mAP/R@1 for a similarity function sim(i, j) (higher = closer).
 * Returns the official query protocol and the all-queries variant.
 */
function retrieval(names, sim) {
  const index = new Map(names.map((n, i) => [n, i]))
  const scene = new Int32Array(names.length).fill(-1)
  const scenes = []
  for (const { query, similar } of Object.values(gt)) {
    const members = [query, ...similar].map((n) => index.get(n)).filter((i) => i !== undefined)
    if (members.length < 2) continue
    for (const i of members) scene[i] = scenes.length
    scenes.push(members)
  }
  const rank = (q) => {
    const others = []
    for (let j = 0; j < names.length; j++) if (j !== q) others.push([j, sim(q, j)])
    others.sort((a, b) => b[1] - a[1])
    return others.map(([j]) => j)
  }
  let apQ = 0
  let r1Q = 0
  for (const members of scenes) {
    const [q, ...rel] = members
    const ranked = rank(q)
    apQ += averagePrecision(ranked, new Set(rel))
    r1Q += rel.includes(ranked[0]) ? 1 : 0
  }
  let apAll = 0
  let nAll = 0
  for (const members of scenes) {
    for (const q of members) {
      apAll += averagePrecision(rank(q), new Set(members.filter((m) => m !== q)))
      nAll++
    }
  }
  return { mAP: apQ / scenes.length, R1: r1Q / scenes.length, mAPall: apAll / nAll, scene }
}

// ── Projection metrics ──────────────────────────────────────────────────────

/** Indices of the k nearest neighbours of every point under dist(i, j). */
function knn(n, k, dist) {
  const out = []
  for (let i = 0; i < n; i++) {
    const d = []
    for (let j = 0; j < n; j++) if (j !== i) d.push([j, dist(i, j)])
    d.sort((a, b) => a[1] - b[1])
    out.push(d)
  }
  return out
}

/** Trustworthiness (Venna & Kaski): penalises 2D neighbours that were far in high-D. */
function trustworthiness(highSorted, lowSorted, k) {
  const n = highSorted.length
  let penalty = 0
  for (let i = 0; i < n; i++) {
    const highRank = new Map(highSorted[i].map(([j], r) => [j, r + 1]))
    for (const [j] of lowSorted[i].slice(0, k)) {
      const r = highRank.get(j)
      if (r > k) penalty += r - k
    }
  }
  return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * penalty
}

function project(rows, method) {
  // src/app.ts processFiles: nNeighbors = clamp(files - 1, 2, 15)
  const nNeighbors = Math.max(2, Math.min(15, rows.length - 1))
  const matrix = druid.Matrix.from(rows.map((v) => Float64Array.from(v)))
  switch (method) {
    case 'TSNE':
      return new druid.TSNE(matrix, { d: 2, perplexity: Math.min(30, Math.floor(rows.length / 3)) }).transform()
    case 'PCA':
      return new druid.PCA(matrix, { d: 2 }).transform()
    case 'ISOMAP':
      return new druid.ISOMAP(matrix, { d: 2, neighbors: nNeighbors }).transform()
    case 'LLE':
      return new druid.LLE(matrix, { d: 2, neighbors: nNeighbors }).transform()
    case 'MDS':
      return new druid.MDS(matrix, { d: 2 }).transform()
    case 'SAMMON':
      return new druid.SAMMON(matrix, { d: 2 }).transform()
    case 'TriMap':
      return new druid.TriMap(matrix, { d: 2 }).transform()
    case 'UMAP':
      return new druid.UMAP(matrix, { d: 2, n_neighbors: nNeighbors, local_connectivity: 1 }).transform()
    default:
      throw new Error(`unknown projection ${method}`)
  }
}

// ── Run ─────────────────────────────────────────────────────────────────────

const ids =
  args.models?.split(',') ??
  (await readdir(args.emb)).filter((f) => f.endsWith('.json') && f !== 'results.json').map((f) => f.slice(0, -5)).sort()
const results = []
for (const id of ids) {
  const e = await loadEmbeddings(id)
  const dot = (i, j) => {
    const a = e.rows[i]
    const b = e.rows[j]
    let s = 0
    for (let d = 0; d < a.length; d++) s += a[d] * b[d]
    return s
  }
  const emb = retrieval(e.names, dot)
  const row = {
    model: id,
    images: e.names.length,
    sPerImage: +(e.seconds / e.names.length).toFixed(2),
    mAP: +emb.mAP.toFixed(4),
    R1: +emb.R1.toFixed(4),
    mAPall: +emb.mAPall.toFixed(4),
    projections: {}
  }
  console.log(`${id}: mAP ${row.mAP} R@1 ${row.R1} mAP(all) ${row.mAPall} (${row.sPerImage}s/img)`)

  const n = e.names.length
  const highSorted = knn(n, n, (i, j) => 1 - dot(i, j))
  for (const method of args.projections.split(',')) {
    const t0 = performance.now()
    let pts
    try {
      pts = project(e.rows, method).to2dArray()
    } catch (err) {
      row.projections[method] = { error: String(err.message ?? err) }
      console.log(`  ${method}: failed ${err.message ?? err}`)
      continue
    }
    const seconds = (performance.now() - t0) / 1000
    const d2 = (i, j) => Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
    const lowSorted = knn(n, n, d2)
    const T = trustworthiness(highSorted, lowSorted, 10)
    const flat = retrieval(e.names, (i, j) => -d2(i, j))
    let nn = 0
    let counted = 0
    for (let i = 0; i < n; i++) {
      if (emb.scene[i] < 0) continue
      counted++
      if (emb.scene[lowSorted[i][0][0]] === emb.scene[i]) nn++
    }
    row.projections[method] = {
      seconds: +seconds.toFixed(1),
      trust10: +T.toFixed(4),
      mAP2d: +flat.mAP.toFixed(4),
      nn1: +(nn / counted).toFixed(4)
    }
    console.log(`  ${method}: trust@10 ${T.toFixed(3)} 2D-mAP ${flat.mAP.toFixed(3)} 1-NN ${(nn / counted).toFixed(3)} (${seconds.toFixed(1)}s)`)
  }
  results.push(row)
}
if (args.json) await writeFile(args.json, JSON.stringify(results, null, 2))
