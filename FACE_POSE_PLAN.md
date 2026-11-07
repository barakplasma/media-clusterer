# Plan: faces, poses and fast detectors

Research notes for a second per-photo signal next to the whole-image embedding: who is in the photo
(faces), what people are doing (pose), or what objects are in it (detection). Written 2026-11-07.
Nothing here is built yet.

## Candidates

| Model                                | Task                                                           | Size                           | License                           | Browser path                                                                     | Notes                                                                                                      |
|--------------------------------------|----------------------------------------------------------------|--------------------------------|-----------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **YuNet** (OpenCV zoo)               | face detection + 5 landmarks                                   | ~0.3 MB ONNX                   | MIT                               | raw onnxruntime-web, like `src/sapiens2.ts`                                      | Very fast; landmarks are what face alignment needs                                                         |
| **SFace** (OpenCV zoo)               | face embedding, 128-d                                          | ~37 MB fp32 ONNX (int8 ~10 MB) | Apache-2.0                        | raw onnxruntime-web                                                              | Takes the aligned 112×112 crop from YuNet                                                                  |
| **EdgeFace-XS**                      | face embedding, 512-d                                          | ~7 MB                          | check the repo (research release) | needs our own ONNX export                                                        | Better accuracy per MB than SFace; licence must be checked before use                                      |
| **MediaPipe Pose Landmarker (lite)** | 33-keypoint pose                                               | ~3–6 MB                        | Apache-2.0                        | `@mediapipe/tasks-vision` (WASM + GPU delegate)                                  | Best licence and size for pose                                                                             |
| **YOLO26n / YOLO26n-pose**           | detection / pose                                               | ~5–10 MB                       | **AGPL-3.0** (Ultralytics)        | Transformers.js 3.8+ on WebGPU (`onnx-community/yolo26n-pose-ONNX`)              | Fastest general detector. AGPL is fine for this open source app but it binds anyone who forks and hosts it |
| **Cactus Mirage**                    | detection, OCR, segmentation, classification, embedding in one | ~22 MB (2-bit)                 | Apache-2.0                        | **none yet**: ships for the Cactus engine (native, `pip install cactus-compute`) | Interesting for the remote path, not for the tab                                                           |

### Cactus Mirage, in more detail

Released 2026-11-05. One 22 MB 2-bit model does detection, OCR, segmentation and embeddings, and it
writes structured text that a small language model can read. It runs on Cactus's native engine, with no
ONNX or WebGPU build, so it can't run in the browser today. Two ways it could still be useful:

1. **Remote mode.** Run it on the k3s box or a vast.ai instance behind a small OpenAI-compatible wrapper,
   and it plugs into the existing `openai` model variant.
2. **Watch for a WASM/WebGPU build** of the Cactus engine; then it could become a local tier.

## Recommendation

Start with **faces: YuNet + SFace**. Both are tiny, permissively licensed ONNX files that run on the
onnxruntime-web stack `src/sapiens2.ts` already uses. Face clusters ("everyone named X") are the feature
a whole-image embedding can't give you. After that, add **MediaPipe Pose** if pose search is wanted.
YOLO is only worth it if object detection itself becomes a feature: the AGPL licence and the extra
download cost more than the speed gains.

```mermaid
flowchart LR
  img[Photo thumbnail] --> det[YuNet<br/>faces + 5 landmarks]
  det -->|aligned 112x112 crops| emb[SFace<br/>128-d per face]
  emb --> store[(IndexedDB<br/>@faces/ prefix,<br/>one key per face)]
  store --> clus[Cluster faces<br/>HDBSCAN / cosine]
  clus --> ui[People filter<br/>name a cluster once]
  img --> whole[Existing image embedder] --> map[Map + text search]
```

## Open questions

- Store one vector per face under `@faces/<file key>#<i>`, so the bundle export carries it unchanged?
- Should overnight mode run faces as one more "model" in its list? That would fit the existing queue.
- Do face embeddings ever leave the device in remote mode? (Proposal: never. They are biometric data.)

## Sources

- [Cactus Mirage blog post](https://cactuscompute.com/blog/mirage), [model card](https://huggingface.co/Cactus-Compute/mirage), [engine](https://github.com/cactus-compute/cactus)
- [YOLO26 WebGPU demo (Transformers.js)](https://huggingface.co/spaces/ipepe/YOLO26-WebGPU/blob/main/main.js), [Ultralytics ONNX export](https://docs.ultralytics.com/integrations/onnx), [onnx-community/yolo26n-pose-ONNX](https://huggingface.co/onnx-community/yolo26n-pose-ONNX)
- [FaceX browser quickstart (EdgeFace-XS)](https://github.com/facex-engine/facex), [EdgeFace overview](https://www.emergentmind.com/topics/edgeface)
