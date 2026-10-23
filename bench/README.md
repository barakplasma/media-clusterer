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
# INRIA Holidays (1,491 photos, 500 scenes): the Kaggle mirror needs no login
curl -L -o holidays.zip https://www.kaggle.com/api/v1/datasets/download/vadimshabashov/inria-holidays
unzip holidays.zip -d data/holidays     # images/ + groundtruth.json

node bench/embed.mjs --data data/holidays --out bench/out
node bench/metrics.mjs --data data/holidays --emb bench/out --json bench/out/results.json
```

## COCO: what is in the photo, and text search

`bench/data/coco-subset.json` lists 870 COCO val2017 images with one clear
subject (54 categories, up to 20 each) and one caption each.

```bash
node bench/fetch-coco-subset.mjs --out data/coco      # ~140 MB of images, no full COCO download
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
