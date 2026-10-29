# H3 Ref2VA: portable SVDQuant NVFP4 PTQ recipe

This is an experimental, inference-only **original Ref2VA / all208 / RTN /
rank32** recipe. It adds runnable collection, export, and inference entry points
to this branch; it is not a bundled serving engine or an upstream SGLang backend.
New-path GPU/E2E validation is pending. Do not substitute historical compatibility
results for a run of these commands on your model and runtime.

## 1. Dependencies and scope

Use a dedicated environment with this ModelOpt branch installed (`pip install -e .`
from its repository root) and the dependencies in `requirements-modelopt-overlay.txt`.
Install the pinned SGLang checkout and its dependencies in the **same environment**.
Do not let installation silently replace the required Torch build.

The NVFP4 export/runtime gates currently require:

- B200/SM100, Python 3.12, Torch `2.11.0+cu130`, CUDA 13.0.
- A separately built and authorized TRT shared library exposing
  `trtllm.fp4_quantize`, `trtllm.nvfp4_quantize_smooth`, and
  `trtllm.nvfp4_svdquant_gemm`, compatible with that exact environment.
- The original, full local H3 Ref2VA model partition, including text/VAE/audio
  components. The transformer must retain the original 50+2 block architecture.
- FFmpeg and FFprobe on PATH; enough GPU memory for **both retained BF16 weights
  and packed weights**. This integration does not measure compressed residency.

The custom runtime is an external prerequisite, **not** supplied by `pip install
modelopt` or the stock TRT library. Obtain its matching build and approved SHA256
from the runtime provider. This branch does not contain its source/build recipe;
without it, collection and CPU tests are available, but actual NVFP4 GEMM is not.
Only load trusted libraries and local artifacts. Hashes detect mutation, not provenance.

Supported launch boundary: TP1 × UlyssesN, no offload, serial requests, no CUDA
graph/compile integration. GPU counts 1/2/4/8 are selectable, not all certified.
Start with 2 B200s. This recipe pins 5 seconds, short edge 768, 50 requested sampler
steps and **49 actual DiT calls per request**; any deviation fails the request gate.
The output check requires 124 frames, 1344×768 (or 768×1344), with decodable audio.

208 projections are quantized: QKV/out/FC1/FC2 in 50 main blocks and 2 token-refiner
blocks. Other projections, encoders, VAE/audio codecs, and attention computation
remain outside this policy. The smoothing/SVD/RTN math is described in README.md.

## 2. Prepare an isolated SGLang checkout and your data

Run all following Python commands from this example directory. Use absolute paths
for model/data/run directories. Example shell variables below are placeholders:

```bash
cd examples/diffusers/h3_svdquant_ptq
export PYTHONPATH="$PWD:../../..:$PYTHONPATH"
export H3_PTQ_MODEL=/data/models/h3-ref2va
export H3_PTQ_SGLANG=/data/src/sglang-h3-ptq
export H3_PTQ_DATA=/data/my-h3-ptq-inputs
export H3_PTQ_RUN=/data/runs/h3-ptq-new-run
export H3_PTQ_LIBRARY=/data/runtime/libth_common.so
export H3_PTQ_LIBRARY_SHA=REPLACE_WITH_APPROVED_LIBRARY_SHA256

git clone https://github.com/sgl-project/sglang.git "$H3_PTQ_SGLANG"
git -C "$H3_PTQ_SGLANG" checkout 0ba46c88e5b16c739be0b698ec7405a147a05cb4
python scripts/h3_ptq.py prepare-sglang --sglang "$H3_PTQ_SGLANG"
```

Install that checkout using its own installation instructions, preserving the ABI
above. The patch changes only `post_load_weights()` to install opt-in hooks; without
`H3_MODELOPT_PTQ_CONFIG` normal inference is unaffected. A modified H3 source or
different revision is rejected rather than patched over.

Copy `configs/calibration.example.json` to `$H3_PTQ_DATA/calibration.json` and
`configs/evaluation.example.json` to `$H3_PTQ_DATA/evaluation.json`. Supply your own
images and edit the prompts/relative image paths. The files contain **no images**
and will fail until references exist. Image paths must stay under the manifest
directory. Materialize model files as regular files, not HF cache symlinks.

Use distinct calibration and evaluation prompts **and image contents**. Every
launch checks both manifests and rejects reuse. The two calibration rows are a
smoke-test template, not a claim that two examples calibrate every domain well.
Use representative authorized calibration data; keep evaluation data held out.

## 3. Collect BF16 calibration inputs

Terminal A, foreground server:

```bash
python scripts/h3_ptq.py serve --mode collect --gpus 2 \
  --model "$H3_PTQ_MODEL" --sglang "$H3_PTQ_SGLANG" \
  --requests "$H3_PTQ_DATA/calibration.json" \
  --other-requests "$H3_PTQ_DATA/evaluation.json" \
  --output "$H3_PTQ_RUN/collect"
```

Terminal B, with the same environment/variables and example working directory:

```bash
python scripts/h3_requests.py --launch "$H3_PTQ_RUN/collect/launch.json" \
  --requests "$H3_PTQ_DATA/calibration.json" --output "$H3_PTQ_RUN/calibration-videos"
```

After the client succeeds, stop **your foreground server** with Ctrl-C to release
GPU memory. Do not send extra requests. On any error, inspect the logs, stop that
server, and start a fresh run directory; partial outputs never count as completed.
The client has bounded startup/request waits but does not kill your server for you.
Only one client may access this local server; it is not a multi-user service.

Collection observes real BF16 inputs without replacing GEMMs. It handles the
request-static token refiner before the first DiT forward, as well as nested refiner
execution. Each rank records FP32 per-channel absolute maxima, row/call counts,
layer order, source identity and request hash. It does **not** collect a GPTQ
Hessian or activation reservoir; those are unnecessary for this RTN-only recipe.

Merge after both requests pass (2 × 49 = 98 calls on each rank):

```bash
python scripts/h3_ptq.py merge --calibration "$H3_PTQ_RUN/collect" \
  --world-size 2 --expected-forwards 98 --output "$H3_PTQ_RUN/calibration"
```

For more calibration rows, set `--expected-forwards` to `49 × number_of_rows`.
Merge requires every rank, all208 layers, matching identities/order, and finite
statistics. Main-block projections must be observed once per DiT call. The result
is `absmax.safetensors` plus `complete.json`; missing completion means failure.

## 4. Factorize, roundtrip through ModelOpt, export and verify

Use one B200 with enough memory after stopping the calibration server:

```bash
python scripts/h3_ptq.py export --model "$H3_PTQ_MODEL" \
  --calibration "$H3_PTQ_RUN/calibration" --output "$H3_PTQ_RUN/packed" \
  --library "$H3_PTQ_LIBRARY" --library-sha256 "$H3_PTQ_LIBRARY_SHA"
python scripts/h3_ptq.py verify --model "$H3_PTQ_MODEL" --artifact "$H3_PTQ_RUN/packed"
```

This reads one selected weight at a time, applies H3 QKV row reordering, uses
observed layer order and seeds `1101 + ordinal`, and runs real ModelOpt
`import_frozen → save → restore → export`. No full dense converted model is retained
on the GPU. The exporter uses the compatible `nested_rcp_approx` QTensor path.

For every layer it checks packed bytes against the **actual TRT packer**, then
actual custom-GEMM outputs/repeatability at M=1/128/129 using deterministic synthetic
inputs. These checks are not held-out video quality or equivalence to every old
runtime behavior. A failure stops export before the final manifest is written.

The artifact contains 208 safetensors payloads, calibration statistics and
`manifest.json`. It records source shards/config hashes, recipe, seed/order,
calibration request hash, per-layer checks and runtime library identity. This is a
new `h3-modelopt-portable-ptq-v1` format, not the old customer-project artifact format.
No old project, legacy Python package, or historical acceptance receipt is needed.

## 5. Generate matched BF16 and NVFP4 videos

Start a **fresh** BF16 server in terminal A:

```bash
python scripts/h3_ptq.py serve --mode bf16 --gpus 2 \
  --model "$H3_PTQ_MODEL" --sglang "$H3_PTQ_SGLANG" \
  --requests "$H3_PTQ_DATA/evaluation.json" --other-requests "$H3_PTQ_DATA/calibration.json" \
  --output "$H3_PTQ_RUN/bf16-server"
```

Terminal B:

```bash
python scripts/h3_requests.py --launch "$H3_PTQ_RUN/bf16-server/launch.json" \
  --requests "$H3_PTQ_DATA/evaluation.json" --output "$H3_PTQ_RUN/bf16-videos"
```

Stop BF16, then start the NVFP4 server in terminal A:

```bash
python scripts/h3_ptq.py serve --mode kernel --gpus 2 \
  --model "$H3_PTQ_MODEL" --sglang "$H3_PTQ_SGLANG" \
  --requests "$H3_PTQ_DATA/evaluation.json" --other-requests "$H3_PTQ_DATA/calibration.json" \
  --artifact "$H3_PTQ_RUN/packed" --library "$H3_PTQ_LIBRARY" \
  --output "$H3_PTQ_RUN/nvfp4-server"
```

Terminal B:

```bash
python scripts/h3_requests.py --launch "$H3_PTQ_RUN/nvfp4-server/launch.json" \
  --requests "$H3_PTQ_DATA/evaluation.json" --output "$H3_PTQ_RUN/nvfp4-videos"
```

The adapter replaces `quant_method.apply` for all208 unsharded linears, keeps
bias, and uses BF16 low-rank down projection plus fused NVFP4 residual/rank-up GEMM.
BF16 weights are deliberately retained for integration safety. No automatic
fallback to BF16, TP sharding, offload or concurrent requests is supported.

The client saves each MP4, media metadata/hash, per-rank forward checks and API
wall time (server load excluded; polling overhead included). It decodes video and
audio and writes `complete.json` only after every request passes. This is not a
kernel benchmark or a complete module-latency breakdown.

## 6. Acceptance and remaining gates

Review the **same evaluation manifest** side by side: reference identity, motion,
temporal consistency, prompt adherence and audio. Define numerical/perceptual
thresholds before evaluation. File/decoded-media validity and exact packing are
separate from human quality acceptance; none imply identical BF16 and NVFP4 videos.

Source identity covers model index and selected transformer shards/config/index,
not all text/VAE/audio files. Keep the entire model directory and runtime immutable
across BF16/PTQ runs. Input mutation checks assume a trusted local environment,
not adversarial writers. Run files may contain local paths and media metadata;
do not publish them or customer data with source code.

Before calling this portable path validated, rerun all208 export and held-out
BF16/NVFP4 generation on the stated ABI. Optional parity against the previous
implementation requires a separate real-kernel/video comparison using identical
source/calibration/seed/order; the historical README workflow remains available.
No speedup, memory saving or quality equivalence is claimed by this new recipe yet.
