# Embedding benchmark

Which model gives the best image similarity, and which 2D projection keeps it
best, measured on labelled data instead of by eye.

- `embed.mjs` embeds a folder of images with each candidate the way the app
  does (512 px wide for the VLM tiers, EmbeddingGemma 2's phone and desktop
  token budgets, nomic's mean pooling and the model card's CLS pooling) and
  caches the vectors.
- `metrics.mjs` scores them on INRIA Holidays (mAP, Recall@1) and scores each
  projection the app offers (trustworthiness@10, 2D scene mAP, 1-NN).

It runs in Node on CPU (transformers.js + onnxruntime-node, the same model
files the browser downloads), which is far faster than SwiftShader WebGPU.

```bash
# Just the scored Holidays subset (first 100 scenes, 274 photos):
node bench/fetch-holidays-subset.mjs --out data/holidays-300

# Or all of INRIA Holidays (1,491 photos, 500 scenes): the Kaggle mirror needs no login
curl -L -o holidays.zip https://www.kaggle.com/api/v1/datasets/download/vadimshabashov/inria-holidays
unzip holidays.zip -d data/holidays     # images/ + groundtruth.json

# The scored set is the first 100 scenes (274 photos): enough to separate
# the models, a few hours on a laptop CPU. metrics.mjs scores only the images
# in --data, so a folder with just those (symlinks + their groundtruth) works.
node bench/embed.mjs --data data/holidays-300 --out bench/out
node bench/metrics.mjs --data data/holidays-300 --emb bench/out --json bench/out/holidays.json
```

## Results (2026-10-23)

Higher is better everywhere. Node CPU, the app's settings. Holidays: 274 photos (first 100 scenes). COCO: 216 photos (4 per category, 54 categories).

### Models

| Model                                                   | Holidays mAP | COCO category mAP | COCO k-means NMI | Caption search R@1 | CPU s/image                                    |
|---------------------------------------------------------|--------------|-------------------|------------------|--------------------|------------------------------------------------|
| EmbeddingGemma 2, desktop (280 tokens)                  | 0.941        | 0.547             | 0.766            | 0.91               | 5.73                                           |
| EmbeddingGemma 2, phone (70 tokens)                     | 0.933        | 0.564             | 0.790            | 0.88               | 1.71                                           |
| nomic, mean pooling (app)                               | 0.911        | 0.560             | 0.797            | 0.54               | 0.33                                           |
| nomic, CLS pooling                                      | 0.882        | –                 | –                | –                  | 0.32                                           |
| SmolVLM2 (576-d vision encoder)                         | 0.895        | 0.505             | 0.758            | –                  | 1.73                                           |
| Chrome AI (Gemini Nano descriptions → nomic-embed-text) | 0.744        | 0.649             | 0.833            | 0.66               | ~14.5 COCO, 22–33 Holidays (Gemini, CI runner) |
| Sapiens2-0.1B int8                                      | 0.739        | 0.149             | 0.643            | –                  | 3.97                                           |

nomic CLS was not run on COCO (comparison only). Chrome AI's descriptions came from GitHub ubuntu-latest (Chrome with Gemini Nano via bench/run-chrome-ai.mjs); it is the best at grouping by content (COCO) and near the bottom at near-duplicate scene matching (Holidays), since a description drops the visual detail that tells two similar scenes apart. Caption search needs a text side: SmolVLM2 and Sapiens2 have none.

### Projections (mean over every model run on that dataset)

| Projection | Holidays 2D mAP | COCO 2D mAP | Holidays trust@10 | COCO 2D NMI |
|------------|-----------------|-------------|-------------------|-------------|
| TSNE       | 0.874           | 0.516       | 0.953             | 0.819       |
| TriMap     | 0.555           | 0.357       | 0.960             | 0.759       |
| UMAP       | 0.393           | 0.206       | 0.919             | 0.706       |
| ISOMAP     | 0.361           | 0.156       | 0.846             | 0.680       |
| PCA        | 0.353           | 0.147       | 0.845             | 0.680       |
| MDS        | 0.352           | 0.146       | 0.844             | 0.683       |
| SAMMON     | 0.314           | 0.092       | 0.821             | 0.660       |
| LLE        | 0.036           | 0.050       | 0.570             | 0.599       |

Raw numbers: `bench/results/`.

## COCO: what is in the photo, and text search

`bench/data/coco-subset.json` lists 322 COCO val2017 images with one clear
subject (54 categories, up to 6 each) and one caption each.

```bash
node bench/fetch-coco-subset.mjs --out data/coco      # ~50 MB of images, no full COCO download
node bench/embed.mjs --data data/coco --out bench/out-coco
node bench/metrics.mjs --data data/coco --emb bench/out-coco --json bench/out-coco/results.json
```

Scores: category mAP (every image is a query, same category is relevant),
k-means NMI against the categories (also on each 2D projection, since the app
clusters the 2D points), and caption search R@1/5/10 for models with a text
side (EmbeddingGemma 2, nomic, Chrome AI).

## Chrome AI (Gemini Nano)

Gemini Nano runs only inside real Chrome on eligible hardware (Chrome's
requirements: 22 GB free disk, and a GPU with more than 4 GB of VRAM or
16 GB of RAM and 4 cores; the runner also sets BypassPerfRequirement). Only
the describing needs Chrome; the embedding of the descriptions runs here.

```bash
# on a Mac (or any machine with Chrome and Gemini Nano):
node bench/fetch-coco-subset.mjs --out dist/bench-images-src
npm run build && cp -R dist/bench-images-src/images dist/bench-images
npx vite preview --port 4173 &
node bench/run-chrome-ai.mjs --url http://127.0.0.1:4173/chrome-ai-bench.html --out chrome-ai.json
# back anywhere:
node bench/embed.mjs --data data/coco --out bench/out-coco --models chrome-ai --descriptions chrome-ai.json
```

`chrome-ai.json` records `availability`; on an ineligible machine it is
`unavailable` and the run exits cleanly, so CI can report it.

CPU stand-ins: SmolVLM2's vision encoder runs as q4 (the app uses q4f16 on
WebGPU; the CPU EP has no fp16 kernels for it).
