# VDN H3 eight-step Frozen PTQ recipe

This code-only example reproduces the **recipe**, not a distributed customer
checkpoint or a guarantee of identical video quality on arbitrary inputs.
This branch packages only Frozen and its shared PTQ helpers, not the original
Ref2VA/SGLang workflow, private experiments or generated artifacts.
The model here is the VDN H3 hybrid checkpoint, rendered through VDN's native
inference code with eight model evaluations (NFE).

## What Frozen means

- Fit only the canonical BF16 rank-32 SVDQuant compensation matrix **B** offline.
  Keep A, smoothing, activation scales, residual packed weights/scales, alpha,
  rank, layout and the existing NVFP4 kernel unchanged.
- Use original BF16 projection weights at **zero-based NFE indices 0 and 5**.
  These evaluations are not skipped. The other six evaluations use the
  reconstructed NVFP4 overlay.
- Apply that rule to all 208 selected logical groups / 312 physical projections
  (50 main blocks plus two token-refiner blocks). QKV remains one joint rank-32
  decomposition. BF16 Q, K and V retain their separate original GEMMs.
- **No sensitive-layer fallback** is added to the Frozen recipe. Attention,
  VDN-specific branches/gates, AdaLN, encoders and decoders stay outside the
  quantized projection policy. Native checkpoint LoRAs are merged before
  calibration and are not optimized by the B fit.

The offline fit uses real-kernel outputs, not FakeQuant: 129 sampled rows per
group/evaluation, balanced video/reference/audio/text rows (33/32/32/32; token
refiners use 129 text rows), ridge values 0.01/0.1/1 and shrink factors 0.25/0.5/1.
The BF16-rounded B change is limited to 10% of the original B Frobenius norm.
Candidates require at least 0.5% mean relative-MSE improvement and no modality
regression above 1% on fitting and development inputs. Full-sequence block
replay at NFEs 0/3/7 provides a separate gate; attention inputs are not sliced.
This is GEMM-level ridge fitting plus block validation, not gradient-based
block training.

**Never recompute `residual = W - B @ A` after correcting B.** That would cancel
the intended error correction. `export-frozen` preserves the original residual,
executes ModelOpt save/restore and compares every packed field against the
expected B-only update, plus real TRT packing and kernel repeat checks.

## Dependencies and scope

Use a dedicated environment. The packaged workflow was revalidated on B200 with
the following dependencies; see the bounded reproduction scope below:

| Component | Version / source |
| --- | --- |
| GPU / ABI | B200 SM100, Python 3.12, Torch 2.11.0+cu130, CUDA 13.0 |
| VDN source | `OpenVDN/vdn-minimax-h3`, commit `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` |
| Checkpoint | `OpenVDN/vdn-minimax-h3`, revision `2ce18deb1d1c6b115d6bb3732f3750fdbac95173`, `stage-dmd-step-250` plus `h3-base` |
| Diffusers | `3a2f35d4efa4c059c8bfb3bc0d6c906264895c81` plus VDN's AdaLN FP32 SiLU patch |
| Transformers / PEFT / Accelerate | 5.12.1 / 0.20.0 / 1.14.0 |
| Triton / FlashAttention 4 | 3.6.0 / 4.0.0b19 |

Obtain/install VDN and its dependencies separately. On a **fresh, dedicated**
checkout at that VDN revision, its `scripts/setup_diffusers.sh` installs the
required Diffusers base plus `diffusers_patches/`. Do not run setup against a
shared dirty Diffusers checkout. Both BF16 and Frozen must use the same patch.
Use the checkpoint's `h3-base` for DiT/VAE and an authorized local MiniMax H3
snapshot containing `processor/` and `text_encoder/` for text encoding.

Install this ModelOpt branch as described in step 0 below. The separately
supplied, ABI-compatible TRT library must provide `fp4_quantize`,
`nvfp4_quantize_smooth` and `nvfp4_svdquant_gemm`. Its matching shared-library
dependencies must be on your library search path. No internal binary or kernel
source is included. A stock ModelOpt or TensorRT-LLM install alone is insufficient.
`portable_runtime.PackedLinear` preserves the original PTQ operator sequence;
the external legacy Python package is not required by these entry points.

Only trusted local model files, Python checkouts and libraries should be used.
Model assembly may invoke the upstream checkpoint loader. Do not point it at
untrusted downloads. This example does not silently fetch model weights.

Supported workflow: one GPU, dedicated serial eager process, 768x1344 portrait,
124 frames, 24 fps, three image references at short-edge768, eight NFEs,
video/audio shifts12/3. It does not certify multi-GPU, concurrent requests,
CUDA graphs, compilation, offload or compressed-residency deployment. Original
BF16 projection weights remain resident for fallback. Budget approximately
37.3 GiB for those projection weights in addition to the quantized overlay;
that number is not a complete peak-memory estimate.

## Reproduce from calibration to videos

### 0. Prepare an isolated environment and external dependencies

Start from a dedicated Python 3.12 environment with the Torch/CUDA ABI in the
table above. Install the pinned VDN dependencies and Diffusers patch first, then
install this branch from its repository root:

```bash
python -m pip install -e .
python -m pip install omegaconf safetensors scipy pillow pytest
python -m pip check
cd examples/diffusers/h3_svdquant_ptq
```

These commands are not an automatic installer for VDN, model weights, FA4 or the
external kernel library. Obtain those separately and check the versions in the
table. Do not upgrade Torch underneath an already-built kernel library. Ensure
VDN's video/audio encoding dependencies (including FFmpeg) are available.

Commands run from `examples/diffusers/h3_svdquant_ptq` in the installed branch.
All paths below are placeholders. Keep `H3_RUN` and input media **outside the
repository**. The commands generate private JSON receipts and tensors at runtime;
none are inputs shipped in Git or archived customer test logs.

```bash
export H3_VDN=/path/to/vdn-checkout
export H3_BASE=/path/to/h3-base
export H3_CHECKPOINT=/path/to/stage-dmd-step-250
export H3_TEXT=/path/to/snapshot-with-text-encoder-and-processor
export H3_LIBRARY=/path/to/compatible/libth_common.so
export H3_LIBRARY_SHA256=the-reviewed-sha256-of-your-library
export H3_RUN=/path/to/private/new-frozen-run
export H3_INPUT=/path/to/private/inputs-and-caches
export PYTHONPATH="$PWD:$H3_VDN${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 1. Encode separate fitting, development and evaluation inputs

Prepare your own UTF-8 prompt files and three ordered reference images for each
request. Do not use customer/evaluation frames to fit B or select the recipe.
Encode each request once; the BF16/Frozen comparison must share its exact cache.
For the fitting request, for example:

```bash
python -m vdn_frozen.encode \
  --vdn-root "$H3_VDN" --text-source "$H3_TEXT" --base "$H3_BASE" \
  --prompt-file "$H3_INPUT/fit.txt" \
  --refs "$H3_INPUT/fit1.png" "$H3_INPUT/fit2.png" "$H3_INPUT/fit3.png" \
  --output "$H3_INPUT/fit-cache"
```

Encode the separate development and evaluation requests as well:

```bash
python -m vdn_frozen.encode \
  --vdn-root "$H3_VDN" --text-source "$H3_TEXT" --base "$H3_BASE" \
  --prompt-file "$H3_INPUT/dev.txt" \
  --refs "$H3_INPUT/dev1.png" "$H3_INPUT/dev2.png" "$H3_INPUT/dev3.png" \
  --output "$H3_INPUT/dev-cache"

python -m vdn_frozen.encode \
  --vdn-root "$H3_VDN" --text-source "$H3_TEXT" --base "$H3_BASE" \
  --prompt-file "$H3_INPUT/eval.txt" \
  --refs "$H3_INPUT/eval1.png" "$H3_INPUT/eval2.png" "$H3_INPUT/eval3.png" \
  --output "$H3_INPUT/eval-cache"
```

The output of each encoding command is `prompt.pt`; reference ordering follows the
prompt's Picture1/2/3 numbering. The wrapper calls VDN's existing encoding
helpers with the portrait metadata used by this recipe, without changing VDN.
The reference VAE posterior sampling seed is42; generation seeds are separate.

### 2. Bind inputs on CPU, calibrate, and export the original PTQ baseline

```bash
python -m vdn_frozen.cli init \
  --vdn-root "$H3_VDN" --base "$H3_BASE" --checkpoint "$H3_CHECKPOINT" \
  --library "$H3_LIBRARY" --library-sha256 "$H3_LIBRARY_SHA256" \
  --fit-cache "$H3_INPUT/fit-cache/prompt.pt" --fit-seed 1121 \
  --dev-cache "$H3_INPUT/dev-cache/prompt.pt" --dev-seed 1122 \
  --run "$H3_RUN"

python -m vdn_frozen.cli calibrate --run "$H3_RUN"
python -m vdn_frozen.cli export-baseline --run "$H3_RUN"
```

`init` hashes local source weights/code/library and conditioning without a GPU.
Subsequent stages revalidate source content. Calibration collects pooled absmax
over both BF16 eight-NFE trajectories and saves exact teacher latents.
Baseline export uses the existing ModelOpt compatibility recipe and seed
`1101 + logical_group_ordinal`, actual ModelOpt roundtrips and TRT byte checks.
This is fresh VDN calibration: original Ref2VA packed artifacts are not accepted.

### 3. Reconstruct B, validate full blocks, and export Frozen

```bash
python -m vdn_frozen.cli reconstruct --run "$H3_RUN"
python -m vdn_frozen.cli export-frozen --run "$H3_RUN"
```

Reconstruction checks exact BF16 teacher replay, real-kernel per-group candidates,
development losses, full-block replay and repeatable complete NVFP4 trajectories.
The exporter refuses an empty fit or disagreement between group and block gates;
do not manufacture a passing receipt or disable the gate. A different calibration
dataset may legitimately reject this candidate. No fixed accepted-group count is
hard-coded, and no parameters are selected using evaluation outputs.

Outputs are a `baseline/` packed overlay, private reconstruction tensors/receipts,
and a `frozen/` B-corrected overlay. The latter is not a whole-model checkpoint:
the original VDN model and non-quantized components are still required.

### 4. Render the matched comparison

```bash
python -m vdn_frozen.cli render --run "$H3_RUN" \
  --prompt-cache "$H3_INPUT/eval-cache/prompt.pt" --seed 42 \
  --mode bf16 --output "$H3_RUN/evaluation/bf16"

python -m vdn_frozen.cli render --run "$H3_RUN" \
  --prompt-cache "$H3_INPUT/eval-cache/prompt.pt" --seed 42 \
  --mode frozen --output "$H3_RUN/evaluation/frozen"
```

Each output includes `video.mp4` with audio, final latents and a local receipt.
Frozen requires exactly two BF16 and six NVFP4 calls per logical group: 416 and
1248 calls respectively, not an estimate based only on configured flags. All
eight model evaluations run. Fresh output directories are mandatory; incomplete
or existing results are not silently overwritten.

Compare full video/audio, reference appearance, motion and temporal continuity.
Latent/RGB distance gains do not guarantee subjective quality or improvement in
every modality. The original experiment had held-out regressions, so this recipe
is not automatically promoted to a general default. Timing recorded here includes
audit overhead/cold effects and is not a production speedup benchmark.

### 5. Optional sequential reproduction script

Instead of running steps 1-4 manually, set the environment variables above and
run the following from this example directory with **fresh** output paths:

```bash
bash scripts/reproduce_frozen.sh
```

The script encodes `fit`, `dev` and `eval`, then runs initialization, calibration,
baseline export, reconstruction, Frozen export and both matched renders. It
stops on the first failure; it does not overwrite or silently resume stages.
For existing trusted caches made with the same encoding settings, skip only
encoding with `bash scripts/reproduce_frozen.sh --from-cache`. That mode still
performs fresh calibration and reconstruction; `H3_RUN` must not already exist.
Use `--help` for the required input file names. To resume a failed attempt,
inspect the failure and use the individual stage commands with new output paths
where required; do not delete or fabricate validation receipts.

## Validation and publication boundary

On 2026-11-05, the packaged `init`, `calibrate`, `export-baseline`, `reconstruct`,
`export-frozen` and both `render` modes were rerun on one B200. The run used the
same two fitting/development conditioning caches and one separate three-reference
evaluation cache (seed42, 768x1344, 124 frames) as the original local experiment.
It regenerated the artifacts rather than importing historical success receipts.

- Calibration statistics and both BF16 teachers matched the original files.
- All 208 baseline and Frozen packed payloads matched exactly. The same 207 B
  updates were selected, and all 52 full-block gates passed.
- BF16 and Frozen final latents and MP4 files matched their respective original
  outputs byte for byte. Independent CPU checks also compared all 124 decoded
  RGB frames and decoded audio samples: both were exact for each mode.
- Frozen executed eight NFEs, with two BF16 and six NVFP4 calls for each of the
  208 groups, BF16 indices0/5 and no group fallback.
- Before this minimal branch was extracted, the original PTQ example and Frozen
  CPU tests passed together (101 tests), including the required legacy oracle;
  applicable pre-commit checks passed. This branch keeps only the relevant subset
  of shared-helper tests and all Frozen tests, not the unrelated SGLang tests.

This establishes integration parity for those fixed inputs, not broad quality
acceptance, a performance claim or support for other environments. The encoding
entry point was not GPU-rerun in this reproduction: it started from existing,
trusted conditioning caches. No input prompts, images, caches, generated videos
or test receipts are distributed with this example. Rerun the full sequence on
your own inputs in a matching environment before deployment.

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  python -m pytest -c /dev/null -p no:cacheprovider tests vdn_frozen/tests -q -rs
python -m vdn_frozen.cli --help
python -m vdn_frozen.encode --help
```

Only source, scripts, tests and documentation belong in this directory. Do not
commit generated JSON/JSONL, logs, prompts, images, videos, calibration tensors,
packed weights, source inventories, credentials or runtime binaries. The local
`.gitignore` is a guardrail, not a substitute for reviewing the staged file list.
No customer examples or previously generated accuracy reports are bundled.
Optional historical-oracle tests skip when the separately maintained legacy
implementation is unavailable. The Frozen tests do not need that implementation.
