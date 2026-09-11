# Lessons from Aura, applied to media-clusterer

Cross-repository notes written 2026-09-11, after adding a second and third in-browser VLM to
[Aura](https://github.com/barakplasma/Aura) — the sibling project that runs the same
Transformers.js/WebGPU stack against a live camera instead of a photo library.

The two codebases made the same bet (a small VLM in the browser, `@huggingface/transformers`
`^4.2.0`, a Web Worker, WebGPU with a WASM fallback) and then hit different walls first. Aura
shipped SmolVLM2-256M to real use on a phone and found out what a 256M model does under a real
prompt; media-clusterer got further on tiering, offline recovery and secret hygiene. This
document is the first direction only: what Aura learned that changes something here.

Nothing below is implemented by the PR that introduces this file. Items are ordered by expected
value, not by effort. Where an item affects a decision already recorded, the ADR is named.

---

## L1 · SmolVLM2-256M cannot follow a caption prompt, and here the damage is silent

**Status: blocks [VIDEO_LM_PLAN.md](VIDEO_LM_PLAN.md) M4 as written.**

Aura gave SmolVLM2-256M a five-line instruction ("write the spoken announcement … return a JSON
object") and the model paraphrased the instruction back instead of answering it. The literal
output, read aloud to a room:

> The alert condition was just met. The operator instruction for the response was "What they
> look like".

This is not a prompt-engineering miss. A 256M-parameter decoder has enough capacity to continue
the text it was given and not enough to work out that it was addressed. Aura's fix was to stop
asking: a separate `promptProfile` per model, where the 256M row gets a one-line positional
prompt ("YES 80 a person is at the door") and an echo filter on the output, and the models that
can actually follow an instruction get the real prompts.

**Why this is worse here than it was there.** In Aura a prompt echo is embarrassing — it gets
spoken and the operator hears it. In media-clusterer the caption is not the end of the pipeline:
under `enableTextSearch` it is embedded with `nomic-embed-text` and becomes `state.searchVectors`
(VIDEO_LM_PLAN.md, "Search needs a second vector array"). A caption that is a paraphrase of
`DEFAULT_DESCRIBE_PROMPT` (`src/chromeAI.ts:52-56`) is near-identical for *every* item in the
library, so every search vector collapses toward the same point. The failure mode is not a funny
caption; it is a search index that returns arbitrary results and looks like it is working.

M4 ships tiers B and C — both SmolVLM2 — and M6 embeds their captions. So:

- Before M4 writes a caption to IndexedDB, run the tier's own prompt against ~20 real images and
  read the output. If the captions restate the prompt, the tier is not caption-capable, whatever
  its model card says.
- Keep an echo guard on the caption path regardless of tier: drop a caption that contains a
  distinctive fragment of the prompt that produced it. Aura's is `stripPromptEcho()` in
  `lib/monitor.js` — about fifteen lines, no dependencies.
- Consider making caption-capability a tier field that M2's measurement gate *sets*, rather than
  one the plan asserts up front. `selectable` already encodes "the code can deliver what the label
  claims"; this is the same idea one level down.

## L2 · Replace SmolVLM2 with LFM2.5-VL — the vision tower is a drop-in and the decoder works

ADR-0001 chose SmolVLM2 in August 2026, when it was the only sub-1B video-native VLM with an ONNX
export. That is no longer true, and the alternative is better on every axis this repo cares about.

[`LiquidAI/LFM2.5-VL-450M-ONNX`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX) is
LiquidAI's own export, with their own WebGPU recipe in the README (fp16 vision encoder + q4
decoder; q8 is explicitly not WebGPU-compatible). Transformers.js 4.2.0 — the version already in
`package.json` — registers `lfm2_vl` → `Lfm2VlForConditionalGeneration`, so no version bump.

What matters for the decision recorded in ADR-0001:

| | SmolVLM2-256M | LFM2.5-VL-450M |
| --- | --- | --- |
| `vision_config.hidden_size` | 768 | **768** (SigLIP2) |
| Vision encoder alone | 55 MB (q4f16) | 57 MB (q4) / 180 MB (fp16) |
| Decoder | 134 MB → paraphrases prompts | LFM2-350M → follows them |
| Export maintained by | community | the model's authors |

The 768 is the load-bearing number: ADR-0001's whole "pooled features are a drop-in" argument
survives the swap, and `assertVisionDim()` (`src/vlmTiers.ts`) already exists to catch it if the
graph disagrees. A tier-A-equivalent row for LFM2.5-VL is ~57 MB — the same order as the
SmolVLM2 tier A that is live today — and it comes with a decoder that makes tiers B/C real.

**Two things that will bite, both found the hard way in Aura:**

1. **Weights live in `.onnx_data` sidecars.** `decoder_model_merged_q4.onnx` is 172 KB; the
   481 MB sits in `decoder_model_merged_q4.onnx_data`. Any code that sizes a download by
   summing `.onnx` files — see L3 — reports a fraction of a percent of the truth. The repo's
   `config.json` declares `transformers.js_config.use_external_data_format`, so
   `from_pretrained` handles the fetch itself; it is only the *accounting* that breaks.
   `modelDownloadUrls()` (`src/modelFallback.ts:42`) has the same problem in a worse place: a
   user following the offline-fallback modal's instructions would download the graph stubs and
   none of the weights.
2. **`do_image_splitting` is not a per-call option here.** Its config default is `true` with
   `max_tiles: 10`, and `Lfm2VlImageProcessor._call` accepts only `return_row_col_info` — the
   flag is read off the processor's own config. Setting it in the `processor(...)` kwargs, the
   way the SmolVLM path does, silently does nothing and you pay for ten tiles plus a thumbnail
   per frame. Assign `processor.image_processor.do_image_splitting = false` after
   `from_pretrained` instead.

[`onnx-community/FastVLM-0.5B-ONNX`](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX)
(`llava_qwen2`, also registered in 4.2.0) is the other credible option — Apple built it for
time-to-first-token on live video. Less interesting here than in Aura: this repo embeds a static
library in the background, where throughput beats latency, and `vision_encoder_q4` is a no-op
quantization at 505 MB because FastViTHD's convolutions don't quantize, making the honest
download ~1.1 GB.

## L3 · Model-download progress needs the Hub file tree, fetched upfront

`src/vlm.ts:29-49` aggregates progress by summing the per-file `loaded`/`total` pairs it has seen
so far. That is monotonic *within* a file and not across the handoff between them: transformers.js
reports one file at a time, so when a finished 2 MB config hands off to a 481 MB decoder the
denominator jumps and the percentage falls. Aura shipped exactly this and users reported the bar
going backwards.

The fix is to know the grand total before the first byte arrives — one call to
`https://huggingface.co/api/models/<id>/tree/main?recursive=true`, filtered to the files this
dtype config will actually fetch, summed once. Aura's version is `lib/model-size.js` (~90 lines,
pure, `fetchImpl` injected so it is unit-testable) plus `lib/download-progress.js` for the
running sum. Both port with nothing but the import paths changed — though note that Aura is
GPL-3.0 and this repo declares no license at all, so copying files across is a licensing decision
before it is a technical one. Reimplementing from this description avoids the question entirely;
the logic is not the hard part, knowing that it is needed is.

Two details worth copying verbatim:

- The estimate is **best-effort and never blocks the load**. Offline, blocked Hub, unknown repo —
  every failure resolves to `null` and the bar degrades to the old per-file behaviour. A progress
  bar must never be able to prevent the thing it is measuring.
- It must count `.onnx_data` and `.onnx_data_N` chunks against the `.onnx` that owns them, or
  every external-data model (L2) is sized at a thousandth of itself. Aura's `externalDataParent()`
  is a three-line regex.

## L4 · Build the measurement surface before the model swap, not after

VIDEO_LM_PLAN.md M2 calls itself "the measurement gate" and says to "compare cluster quality
against `sapiens2-fp16` on the same folder". There is no tool in the repo that does that, so in
practice the comparison is someone squinting at two canvases and forming an impression.

Aura hit the same problem and built `lib/eval.js` + an EVAL screen: pin a handful of sample
images, define the expected answer, then run the (image × model × prompt-variant) matrix and
score it. It changed how model changes get made — a swap becomes a number instead of an argument,
and a prompt change that helps one model and hurts another becomes visible rather than
theoretical.

The equivalent here is cheaper than Aura's, because the ground truth is already available:

- Pick a folder, embed it under two variants, and report a clustering-agreement metric
  (adjusted Rand index against the folder structure, or against a hand-labelled subset).
  `src/compute.ts` already has k-means and the cosine machinery.
- Persist the sample set and the last run, the way `lib/eval-store.js` does — a measurement you
  have to reconstruct by hand is one you will run once.
- This is the concrete unblocker for ADR-0001's open risk row ("Pooled SigLIP clusters worse than
  `sapiens2-fp16`"), and it has to exist before L2's swap is decidable.

## L5 · The worker facade needs an abort path and an injectable factory

Two gaps in `src/vlm.ts`, both of which Aura hit in production:

**No way to stop work in flight.** `VlmRequest` (`src/types.ts:73-76`) is `load | embed | dispose`.
A user who opens a folder of 2,000 items and immediately navigates away waits for the whole batch,
because `dispose` tears the worker down rather than interrupting it, and on the WASM fallback a
batch is minutes. Aura's answer is an `abort` message plus an `InterruptableStoppingCriteria`
subclass whose flag the worker flips between tokens — generation stops within one token instead of
running to completion. For tier A (encode-only) the equivalent is a checked flag between frames in
`embed()`'s loop; it matters more once M4 makes each item a `generate()` call.

**The facade's own logic is untested.** `ensureWorker()` constructs `new Worker(new URL(...))`
unconditionally, so the id-correlation, the progress aggregation and the crash-cleanup path can
only be exercised in a browser — and they are exactly the parts where an error means a promise
that never settles. Aura exports a `_setWorkerFactory()` test hook and drives the whole protocol
against a fake worker object under `node --test`, including out-of-order replies and a mid-flight
crash. Under `vitest` here it is the same handful of lines.

While in there: `src/vlm.ts` handles the worker's `error` event but not `messageerror`. A
structured-clone failure (a transferred bitmap that is already neutered, say) fires the latter,
and today it settles nothing.

## L6 · Decode greedily

When M4 adds `model.generate()`, pass `do_sample: false`. Two reasons, and the second is the one
that is easy to miss: a caption that changes between runs makes the cache key lie (the same file
gets a different caption, and under `enableTextSearch` a different search vector), and it makes
L4's numbers unreproducible — you cannot tell a real regression from sampling noise.

## L7 · Ship the PWA shell

media-clusterer has no service worker and no manifest (`public/` holds only `_headers`). For an
app whose entire pitch is "your photos never leave the device", that means the one thing it can't
do is run without a network — even though the model weights are already sitting in the Cache API
after the first load.

Aura's `scripts/sw-template.js` generates a shell-caching service worker with one hard rule worth
restating here: **the worker must never intercept anything but same-origin `GET`s.** Aura's
provider calls and webhooks go straight to the network; the equivalent exemptions here are the
Hub/mirror fetches and, under ADR-0002, the remote inference endpoint. A service worker that
caches a POST to an inference endpoint is a data-leak-shaped bug.

Note the interaction with `excludeOrtWasm` (`vite.config.ts:22-31`): ORT's WASM binaries come from
jsDelivr at runtime, so an offline boot needs them cached too, or served same-origin. Aura copies
them into `public/ort/` and gives them a runtime-cache rule for exactly this reason.

## L8 · A tier is only auto-selectable if you'd hand it to someone who never opened settings

`src/vlmTiers.ts` has `selectable`, meaning "the code can deliver what this label claims" — a good
distinction, and Aura copied it. Aura then found it needs a second one: `autoSelectable`, meaning
"`pickVlmTier()` may choose this without being asked". FastVLM is selectable and not
auto-selectable, because a 1.1 GB download is a decision a person should make.

Two findings for `pickVlmTier()` when M6 writes it, both of which cost Aura a debugging session:

- **`navigator.deviceMemory` is capped at 8 by the spec.** A 16 GB Pixel 10 and an 8 GB laptop
  report the same number, so `8` has to read as "comfortable", never as a ceiling to be careful
  about. `computeOptimalBatchSize()` (`src/hardware.ts:9-33`) already defaults it to `2` when
  absent, which is the right instinct for a batch size and the wrong one for a tier choice —
  Firefox and Safari don't implement the property at all, and treating "absent" as "tiny" would
  hand every non-Chrome browser the smallest tier regardless of hardware.
- **The real constraint is the GPU buffer limit, not RAM.** ORT allocates each ONNX initializer
  into a GPU buffer, so a model whose largest weight file exceeds the adapter's
  `maxStorageBufferBindingSize` fails at session creation — and Android adapters report far
  smaller limits than desktop ones. `requestAdapter().limits` is already probed for other reasons
  in `src/sapiens2.ts:270-321`.

## L9 · Drop the stored API key when the endpoint's origin changes

For ADR-0002 / REMOTE_INFERENCE_PLAN.md. AGENT.md's secrets rule covers where the key may not go
(`state`, `mc_settings`, a URL, `console.error`) but not when it should stop existing. The failure
is mundane and easy: a user configures OpenRouter, pastes their key, later switches the base URL to
a local vLLM — and the next request sends their OpenRouter credential to whatever is listening on
that host.

Aura's rule is that changing the base URL to a different origin clears the stored key, with
`sameOrigin()` (`lib/monitor.js`) returning `false` for unparseable input so a half-typed URL can
never read as a match and skip the clear. `normalizeBaseUrl()` (`src/openaiCompat.ts:90-120`)
already does the parsing; this is a comparison on top of it.

---

## Ordering

L1 and L2 are one piece of work: the reason to move off SmolVLM2 is that its decoder can't caption,
and L1 is what happens if M4 ships before that is faced. L4 should land before either, because it
is what makes the swap arguable rather than assertable. L3 and L5 are independent and small. L6 is
a line. L7 and L9 are their own tracks.

| Step | Item | Why here |
| --- | --- | --- |
| 1 | L4 — measurement surface | Nothing about a model swap is decidable without it |
| 2 | L3 — Hub-tree progress sizing | Prerequisite for L2: external-data models break the current bar |
| 3 | L2 + L1 — LFM2.5-VL tier, echo guard | Supersedes ADR-0001's model choice; write a new ADR, don't edit it |
| 4 | L5 — abort + test seam | Independent; grows more valuable as generation gets slower |
| 5 | L8 — tier picker | Needs more than one credible tier to choose between, i.e. needs L2 |
| 6 | L6, L7, L9 | Independent of the above and of each other |

## What went the other way

For the record, three things this repo does that Aura does not and should: the offline/corporate-proxy
fallback (`src/modelFallback.ts` — upload the weights from disk, or point at a mirror), `isDownloadError()`'s
careful refusal to treat a bare `TypeError` as a network failure, and the ADR discipline in `docs/adr/`.
Aura has a `docs/PRD-*.md` convention that records what was built but not what was rejected.
