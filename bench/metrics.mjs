// Score cached embeddings (bench/embed.mjs) on INRIA Holidays, and score
// each 2D projection the app offers on how well it keeps those neighbours.
//
//   node bench/metrics.mjs --data <holidays dir> --emb <embed.mjs --out dir> \
//     [--models a,b] [--projections UMAP,TSNE,PCA] [--json results.json]
//
// Embedding: Holidays protocol (each scene's query ranks every other image;
// the scene's other images are relevant) -> mAP and Recall@1, plus an
// all-images-as-queries mAP, which is less noisy.
// COCO subset (labels.json): every image is a query, same category is
// relevant; k-means NMI against the categories; caption search R@K.
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
    projections: { type: 'string', default: 'UMAP,TSNE,PCA,ISOMAP,TriMap,MDS,LLE,SAMMON' },
    json: { type: 'string' }
  }
})

// Holidays: groundtruth.json (scenes, each with a designated query).
// COCO subset (bench/prep-coco.mjs): labels.json (category per image), no
// designated query, so every image is a query; plus captions.json.
const readJson = async (f) => JSON.parse(await readFile(join(args.data, f), 'utf8')).valueOf()
let groupsByName
let hasQueries = true
try {
  groupsByName = Object.values(await readJson('groundtruth.json')).map(({ query, similar }) => [query, ...similar])
} catch {
  hasQueries = false
  const byLabel = new Map()
  for (const [name, label] of Object.entries(await readJson('labels.json'))) byLabel.set(label, [...(byLabel.get(label) ?? []), name])
  groupsByName = [...byLabel.values()]
}
let captions = null
try {
  captions = await readJson('captions.json')
} catch {
  // Not captioned.
}

// Score only the images in --data, so vectors cached for a larger set of the
// same images can be reused on a subset.
const present = new Set(await readdir(join(args.data, 'images')))

async function loadEmbeddings(id) {
  const meta = JSON.parse(await readFile(join(args.emb, `${id}.json`), 'utf8'))
  const buf = await readFile(join(args.emb, `${id}.f32`))
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
  const keep = meta.names.map((n, i) => [n, i]).filter(([n]) => present.has(n))
  if (keep.length < present.size - (present.has('list.json') ? 1 : 0)) throw new Error(`${id}: vectors cover ${keep.length} of the ${present.size} images in --data`)
  const rows = keep.map(([, i]) => flat.subarray(i * meta.dim, (i + 1) * meta.dim))
  return { ...meta, names: keep.map(([n]) => n), rows }
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
  for (const group of groupsByName) {
    const members = group.map((n) => index.get(n)).filter((i) => i !== undefined)
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
  for (const members of hasQueries ? scenes : []) {
    const [q, ...rel] = members
    const ranked = rank(q)
    apQ += averagePrecision(ranked, new Set(rel))
    r1Q += rel.includes(ranked[0]) ? 1 : 0
  }
  let apAll = 0
  let r1All = 0
  let nAll = 0
  for (const members of scenes) {
    for (const q of members) {
      const ranked = rank(q)
      apAll += averagePrecision(ranked, new Set(members.filter((m) => m !== q)))
      r1All += members.includes(ranked[0]) ? 1 : 0
      nAll++
    }
  }
  // Without designated queries (COCO), the all-queries protocol is the protocol.
  return hasQueries
    ? { mAP: apQ / scenes.length, R1: r1Q / scenes.length, mAPall: apAll / nAll, scene, k: scenes.length }
    : { mAP: apAll / nAll, R1: r1All / nAll, mAPall: apAll / nAll, scene, k: scenes.length }
}

// ── Clustering against the labels ───────────────────────────────────────────

/** Seeded k-means++ (Lloyd), so runs are comparable. Returns a cluster per row. */
function kmeans(rows, k, iters = 60, seed = 1) {
  let s = seed
  const rand = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648)
  const d2 = (a, b) => {
    let t = 0
    for (let i = 0; i < a.length; i++) t += (a[i] - b[i]) ** 2
    return t
  }
  const centers = [Float64Array.from(rows[Math.floor(rand() * rows.length)])]
  const best = rows.map((r) => d2(r, centers[0]))
  while (centers.length < k) {
    const total = best.reduce((a, b) => a + b, 0)
    let pick = rand() * total
    let idx = 0
    while (idx < rows.length - 1 && (pick -= best[idx]) > 0) idx++
    centers.push(Float64Array.from(rows[idx]))
    for (let i = 0; i < rows.length; i++) best[i] = Math.min(best[i], d2(rows[i], centers[centers.length - 1]))
  }
  const assign = new Int32Array(rows.length)
  for (let it = 0; it < iters; it++) {
    let moved = 0
    for (let i = 0; i < rows.length; i++) {
      let bi = 0
      let bd = Infinity
      for (let c = 0; c < k; c++) {
        const d = d2(rows[i], centers[c])
        if (d < bd) [bd, bi] = [d, c]
      }
      if (assign[i] !== bi) moved++
      assign[i] = bi
    }
    const sums = centers.map((c) => new Float64Array(c.length))
    const counts = new Int32Array(k)
    rows.forEach((r, i) => {
      counts[assign[i]]++
      for (let d = 0; d < r.length; d++) sums[assign[i]][d] += r[d]
    })
    sums.forEach((sum, c) => counts[c] && centers[c].set(sum.map((v) => v / counts[c])))
    if (!moved && it) break
  }
  return assign
}

/** Normalised mutual information between two labelings (arithmetic mean). */
function nmi(a, b) {
  const n = a.length
  const count = (xs) => xs.reduce((m, x) => m.set(x, (m.get(x) ?? 0) + 1), new Map())
  const ca = count(a)
  const cb = count(b)
  const joint = count(a.map((x, i) => `${x}|${b[i]}`))
  const H = (c) => -[...c.values()].reduce((t, v) => t + (v / n) * Math.log(v / n), 0)
  let I = 0
  for (const [key, v] of joint) {
    const [x, y] = key.split('|')
    I += (v / n) * Math.log((v * n) / (ca.get(Number(x)) * cb.get(Number(y))))
  }
  const denom = (H(ca) + H(cb)) / 2
  return denom ? I / denom : 0
}

/** NMI of k-means (k = number of labels) against the labels, over labelled rows. */
function clusterNmi(rows, scene, k) {
  const idx = []
  for (let i = 0; i < rows.length; i++) if (scene[i] >= 0) idx.push(i)
  const assign = kmeans(idx.map((i) => rows[i]), k)
  return nmi(Array.from(assign), idx.map((i) => scene[i]))
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
  // Category datasets: does k-means (k = categories) recover the categories?
  // (Skipped for Holidays: 500 tiny scenes make k-means slow and NMI uninformative.)
  if (!hasQueries) {
    row.nmi = +clusterNmi(e.rows, emb.scene, emb.k).toFixed(4)
    console.log(`  k-means NMI ${row.nmi}`)
  }

  // Caption search: each caption should find its own image.
  const textFile = join(args.emb, `${id}.text.f32`)
  if (captions && (await readFile(textFile).catch(() => null))) {
    const tb = await readFile(textFile)
    const tflat = new Float32Array(tb.buffer, tb.byteOffset, tb.byteLength / 4)
    const index = new Map(e.names.map((n, i) => [n, i]))
    const hits = { 1: 0, 5: 0, 10: 0 }
    let catHit = 0
    captions.forEach((c, qi) => {
      const q = tflat.subarray(qi * e.dim, (qi + 1) * e.dim)
      const target = index.get(c.name)
      const scores = e.rows.map((r, j) => {
        let t = 0
        for (let d = 0; d < r.length; d++) t += r[d] * q[d]
        return [j, t]
      })
      scores.sort((a, b) => b[1] - a[1])
      const rankOf = scores.findIndex(([j]) => j === target)
      for (const k of [1, 5, 10]) if (rankOf < k) hits[k]++
      if (emb.scene[scores[0][0]] === emb.scene[target]) catHit++
    })
    const n = captions.length
    row.text = {
      R1: +(hits[1] / n).toFixed(4),
      R5: +(hits[5] / n).toFixed(4),
      R10: +(hits[10] / n).toFixed(4),
      categoryAt1: +(catHit / n).toFixed(4)
    }
    console.log(`  text search R@1 ${row.text.R1} R@5 ${row.text.R5} R@10 ${row.text.R10} category@1 ${row.text.categoryAt1}`)
  }

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
    // The app clusters the 2D points, so this is the clustering users see.
    if (!hasQueries) row.projections[method].nmi2d = +clusterNmi(pts, emb.scene, emb.k).toFixed(4)
    console.log(`  ${method}: trust@10 ${T.toFixed(3)} 2D-mAP ${flat.mAP.toFixed(3)} 1-NN ${(nn / counted).toFixed(3)} (${seconds.toFixed(1)}s)`)
  }
  results.push(row)
}
if (args.json) await writeFile(args.json, JSON.stringify(results, null, 2))
