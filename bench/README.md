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

CPU stand-ins: SmolVLM2's vision encoder runs as q4 (the app uses q4f16 on
WebGPU; the CPU EP has no fp16 kernels for it).
