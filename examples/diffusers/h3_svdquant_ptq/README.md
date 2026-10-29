# H3 SVDQuant NVFP4 PTQ compatibility example

This experimental, inference-only example reproduces an existing H3 SVDQuant
recipe through ModelOpt modules, checkpoint state, and a custom NVFP4 QTensor
export path. It is **not** the default ModelOpt SVDQuant algorithm or a
self-contained H3 inference engine. No ModelOpt core files are modified.

**Start here: [portable calibration → ModelOpt export → SGLang inference recipe](RECIPE.md).**
The branch now includes collection hooks, rank merging, a direct ModelOpt exporter,
a minimal opt-in SGLang patch, request templates, and a serial video runner.
These new entry points do not require the old customer-project directory or legacy
Python package. They still require separately supplied compatible TRT NVFP4 kernels.
The new integration is experimental: CPU contracts can be tested here, but new-path
GPU generation/quality validation is still pending. Historical GPU results below
must not be read as acceptance of this new integration.

## Version and scope

This PTQ-only branch is based on commit
`9392dfeabfde8695e9f58421c551ea8004fc3a1a` of
[`tiffany940107/Model-Optimizer`](https://github.com/tiffany940107/Model-Optimizer).
Only this example directory is added; upstream workflow changes and unrelated
experiments are not included.

The original 61 CPU tests plus 13 new portable-pipeline tests passed against this
base (74 total), including the separately supplied, hash-bound legacy oracle.
Full pre-commit checks are run on the added files.
Historical GPU measurements below used the earlier Jingyu branch commit
`881cacdbeed36ab3ca16c4cb8bc56606a42329c9`; GPU and full-generation equivalence
have **not** been rerun on this fork-main base. The measured compatibility recipe
is preserved, but CPU success alone does not transfer that GPU acceptance.

The selector targets the original H3 Ref2VA architecture:

| Projection | Main blocks | Token-refiner blocks | Total |
| --- | ---: | ---: | ---: |
| Fused QKV | 50 | 2 | 52 |
| Attention output | 50 | 2 | 52 |
| MLP FC1 | 50 | 2 | 52 |
| MLP FC2 | 50 | 2 | 52 |
| Total | 200 | 8 | 208 |

AdaLN, input/output/time/condition projections, text encoders, VAE, audio codecs,
and attention QK/softmax/PV computation are outside this policy. Exclusion is not
a claim that those operations cannot be quantized.

The recipe uses calibration-based SmoothQuant-style smoothing (alpha 0.5,
clamp 1/16 to 16), FP32 randomized SVD (rank 32, oversample 8, two iterations),
BF16 low-rank compensation, and an NVFP4 residual. NVFP4 uses E2M1 data,
E4M3 block scales, block size 16, and the legacy static-6 scale convention.
Weights use round-to-nearest, **not GPTQ**. No block-aware refinement or optimizer
training is performed. The numerical contract is in `compat_contract_v1.json`.

## What is implemented

- `h3_modelopt_ptq.py`: registered `h3_svdquant_legacy_v1` mode, actual ModelOpt
  `SVDQuantLinear`/`TensorQuantizer` modules, factorization, and save/restore.
- `packed_replay.py` and `streaming_roundtrip.py`: bounded-memory, per-layer
  checkpoint roundtrips; packed-only replay is a storage gate, not a forward model.
- `modelopt_multiplier_compat.py` and `compat_rcp_triton.py`: custom QTensor export,
  including the measured `nested_rcp_approx` compatibility arithmetic. This does
  not replace the deployment GEMM or claim default ModelOpt output equivalence.
- `recipe_comparison.py` and `native_breakdown.py`: native-versus-compatible
  packing, scale layout conversion, and numerical attribution.
- `artifact_binding.py` and `binding_cache.py`: source/calibration/artifact hashes
  and fail-closed launch checks. Do not bypass them to accept unrelated artifacts.
- `sglang_bridge.py`: validated launch-environment bridge only; it does not install
  SGLang hooks or implement a standalone SGLang quantization backend.
- `portable_*.py`, `scripts/h3_ptq.py`, `scripts/h3_requests.py`, and `patches/`:
  the new independent calibration/export/runtime path documented in [RECIPE.md](RECIPE.md).
  `configs/` contains templates, not calibration data or customer assets.

## Setup and CPU tests

Use a separate environment and install this repository with its required
dependencies. The dependency inventory in `requirements-modelopt-overlay.txt`
does not pin a complete GPU environment and intentionally does not replace Torch.
From the repository root:

```bash
cd examples/diffusers/h3_svdquant_ptq
export PYTHONPATH="$PWD:../../..:$PYTHONPATH"
python -m pytest -c /dev/null -p no:cacheprovider tests -q -rs
```

Legacy-oracle tests require separately supplied, authorized legacy source.
Without it, those tests explicitly skip; this is **not** a full parity result.
To require the entire suite:

```bash
export H3_LEGACY_ROOT=/path/to/authorized/legacy-share
export H3_REQUIRE_LEGACY_TESTS=1
python -m pytest -c /dev/null -p no:cacheprovider tests -q -rs
```

The expected file is `$H3_LEGACY_ROOT/svdquant_h3/core.py`; its hash must match
the contract. No local workspace layout is assumed. Tests exercise real upstream
CPU pack/dequant and checkpoint APIs, but CPU success is not CUDA GEMM evidence.
Only load trusted ModelOpt/Torch checkpoints and trusted legacy Python source.

## GPU and real-layer workflows

Historical compatibility runs used B200/SM100, Python 3.12, Torch
`2.11.0+cu130`, CUDA 13.0, and the matching separately supplied legacy TRT runtime.
The scripts enforce that ABI for runtime replay. **Do not load that runtime with
a different Python/PyTorch/CUDA ABI.** No binary, internal kernel source, or runtime library
is distributed here; obtaining those dependencies is a prerequisite for GPU replay.

Add the authorized legacy package to `PYTHONPATH`, then inspect the entry points:

```bash
export PYTHONPATH="$H3_LEGACY_ROOT:$PYTHONPATH"
python scripts/check_trt_replay.py --help
python scripts/check_multiplier_compat.py --help
python scripts/roundtrip_legacy.py --help
python scripts/preflight_regeneration.py --help
python scripts/compare_regenerated.py --help
python scripts/summarize_regeneration.py --help
```

The real-layer sequence is: independently calibrate the source model with the
legacy calibration/export tools; verify its source receipts; run packed-only
roundtrip; prepare regeneration; regenerate with native and compatible ModelOpt
packers; compare factors, packed bytes, and actual GEMM outputs; then evaluate
matched full generation in a separately integrated inference environment.
`compare_regenerated.py --compat-pack nested_rcp_approx` selects the compatibility
variant. Reuse original calibration layer order and absolute seed ordinals.

The whole-artifact driver intentionally requires the original source-bound
208-layer recipe and its legacy project receipts. It is not a generic converter
for arbitrary H3 checkpoints. Runtime activation quantization, GEMM, and low-rank
epilogue remain the legacy implementation. Replacing that backend requires a new
numerical and performance evaluation.

## Historical compatibility workflow (requires external legacy tools)

This section documents reproduction against the older integration. For a new
standalone handoff, use [RECIPE.md](RECIPE.md); do not mix its new manifest format
with the historical receipt-bound artifacts below.

The workflow is:

```text
Pinned BF16 Ref2VA + independent calibration requests
    -> BF16 trajectory collection -> merged activation statistics
    -> legacy SVDQuant export + GPU validation (comparison reference)
    -> ModelOpt smoothing/SVD + save/restore + compatibility NVFP4 packing
    -> 208-layer factor / packed-tensor / same-kernel validation
    -> existing SGLang H3 hooks + the same TRT NVFP4 GEMM
    -> matched BF16 / legacy PTQ / ModelOpt-compatible PTQ videos
```

**This is a compatibility recipe, not a self-contained one-command pipeline.**
The calibration collector, original exporter, patched H3 runtime, and complete
generation wrappers are external dependencies. The commands below distinguish
those dependencies from tools included in this branch. They are a source-checked
recipe, not a new execution or fresh GPU result on this branch.

### 1. Prepare the model, runtime, and independent inputs

Use the **Ref2VA** partition of `MiniMaxAI/MiniMax-H3`, revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. The whole-artifact driver checks
this identity; an FL2VA artifact or arbitrary checkpoint is not interchangeable.
Prepare the complete model components, not just DiT weights.

You also need an authorized copy of the original Ref2VA integration project,
containing the following external components:

- `svdquant_h3/`, the hooked SGLang checkout, and compatible custom TRT library.
- `scripts/run_case.py`, `validate_calibration.py`, and `export_ref2va.py`.
- For regenerated-candidate inference: `check_regenerated_compat.py` and
  `run_regenerated_case.py`, with their helper dependencies.
- Source-download receipts, runtime inventory, and independently prepared
  calibration/evaluation manifests and media. These are not bundled here.

These historical wrappers enforce a specific project layout and request
contract; they are **not generic public H3 CLI tools**. Obtain the authorized
integration separately, or implement an equivalent reviewed integration and
validate its receipts. Do not manufacture success receipts or disable source
checks to make an arbitrary directory pass. An ordinary ModelOpt installation
or stock SGLang installation alone is insufficient.

Commands below run from this example directory. All `/path/to/...` values are
placeholders. Use fresh output paths and only trusted inputs. In the matching
GPU environment, set:

```bash
export H3_EXAMPLE="$PWD"
export H3_LEGACY_PROJECT=/path/to/authorized-ref2va-integration
export H3_RUN=/path/to/new-run
export H3_CALIBRATION="$H3_RUN/calibration"
export H3_SOURCE_RECEIPT="$H3_RUN/source-preflight.json"
export H3_GOLDEN="$H3_RUN/legacy-export"
export H3_REGEN="$H3_RUN/modelopt-regeneration"
export H3_TRT_LIBRARY="$H3_LEGACY_PROJECT/runtime/trtllm/thop/libth_common.so"
export PYTHONPATH="$H3_EXAMPLE:$H3_EXAMPLE/../../..:$H3_LEGACY_PROJECT:$H3_LEGACY_PROJECT/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H3_RUN"
```

The external project expects the model under `models/MiniMax-H3/Ref2VA/`,
`results/download_complete.json`, `results/runtime-pip-freeze.txt`,
`calibration_manifest.json`, and a separate held-out `case_manifest.json`.
Prepare these through its original setup/validation tools before continuing.
Provide working `ffmpeg`/`ffprobe` and the runtime's dependent libraries.
The historical GPU path is B200/SM100, Python 3.12, Torch 2.11.0+cu130, CUDA 13.0;
the regeneration driver rejects other GPU architectures or ABIs.

### 2. Collect BF16 calibration trajectories (external integration)

Keep calibration prompts and reference media separate from held-out evaluation
inputs. Run the unquantized model with collection hooks; do not calibrate on
already quantized outputs. The historical pilot used two independent requests,
not a representative large calibration dataset.

```bash
python "$H3_LEGACY_PROJECT/scripts/run_case.py" \
  --root "$H3_LEGACY_PROJECT" --mode collect \
  --calibration-manifest "$H3_LEGACY_PROJECT/calibration_manifest.json" \
  --gpus 2 --port 30010 --output "$H3_CALIBRATION"

python "$H3_LEGACY_PROJECT/scripts/validate_calibration.py" \
  --root "$H3_LEGACY_PROJECT" --output "$H3_CALIBRATION" --ranks 2
```

This wrapper uses TP1 x Ulysses2, disables automatic warmup, and serializes
requests. Its contract is **50 requested sampling steps and 49 observed DiT
forwards per request per rank**. Two requests therefore collect 98 forwards on
each rank, not 196 independent trajectories. Changing the sampler requires
revisiting the contract, not merely changing this count.

The collector records per-input-channel `activation_absmax` for all 208 target
linears and bounded activation samples: five timestep bins for main blocks,
one for token refiners, with up to 512 sampled rows per bin. Expected outputs
include `calibration.rank0.pt`, `calibration.rank1.pt`, their sample directories,
generation contracts, and `calibration_validated.json`. Every rank must write
its own files using its actual distributed rank. Do not share a collecting
server with unrelated requests.

### 3. Build the legacy PTQ reference (external integration)

First hash the source model on CPU; then perform export and kernel validation
in the matching B200 environment:

```bash
python "$H3_LEGACY_PROJECT/scripts/export_ref2va.py" \
  --root "$H3_LEGACY_PROJECT" --preflight-only \
  --source-receipt "$H3_SOURCE_RECEIPT"

python "$H3_LEGACY_PROJECT/scripts/export_ref2va.py" \
  --root "$H3_LEGACY_PROJECT" --calibration "$H3_CALIBRATION" \
  --source-receipt "$H3_SOURCE_RECEIPT" --output "$H3_GOLDEN"
```

This merges rank statistics using channelwise maxima and preserves rank0's
layer insertion order. The reference recipe is global rank32 SVD, smoothing
alpha 0.5, static-6 NVFP4 RTN; it does not enable GPTQ or block-aware refinement.
Outputs include `merged-calibration.pt`, 208 packed layer files, `manifest.json`,
and source/calibration/kernel provenance. The wrapper validates all 208 layers
at row counts 1, 128, and 129, including repeat checks.

The legacy export is required here because this branch verifies compatibility
against a measured reference. It is not an assertion that PTQ mathematically
requires two separate exports.

### 4. Regenerate and export through ModelOpt (included)

For each original BF16 weight `W[N,K]`, with calibration channel absmax `a[K]`:

1. Compute FP32 smoothing
   `s = clamp(sqrt(max(abs(W), dim=0) / a), 1/16, 16)`, flooring both spans
   at `1e-8`. Form `W_s = W.float() / s` before rounding `s` to BF16.
2. Run FP32 randomized SVD with rank32, oversample8, and two iterations.
   Form `B[N,32]`, `A[32,K]`, and `R = W_s - B @ A` in FP32; then cast
   `s`, `A`, `B`, and `R` to BF16. Use seed `1101 + absolute_layer_ordinal`.
3. Import those factors into actual ModelOpt `SVDQuantLinear` modules and
   execute ModelOpt save/restore. This is not a call to the default AWQ/full-SVD
   recipe. Checkpoint QKV weights require the original head-interleaved to
   runtime Q/K/V-major row permutation before factorization.
4. Pack `R` as E2M1 NVFP4 with E4M3 scales per block of 16. The compatibility
   `MultiplierNVFP4QTensor` variant is `nested_rcp_approx`; convert scales to
   the legacy TRT swizzled layout. Keep BF16 low-rank factors and smoothing.
   Preserve the calibrated input multiplier and residual multiplier, each
   using the legacy `2688 / amax` convention with a `1e-12` floor.

The conceptual linear is
`Y ~= NVFP4(X*s) @ NVFP4(R).T + ((X*s) @ A.T) @ B.T`.
The actual runtime folds smoothing into the BF16 rank-down matrix and combines
rank-up with the GEMM epilogue. This expression is explanatory, not a
bitwise-equivalent replacement: cast order, scale arithmetic, and accumulation
order matter. See [the numerical contract](compat_contract_v1.json).

Create a fully hashed CPU binding receipt, then regenerate all layers on GPU:

```bash
python scripts/preflight_regeneration.py \
  --legacy-project "$H3_LEGACY_PROJECT" --legacy-artifact "$H3_GOLDEN" \
  --calibration "$H3_CALIBRATION" --source-receipt "$H3_SOURCE_RECEIPT" \
  --output "$H3_RUN/preflight.json" --binding-receipt "$H3_RUN/binding.json"

python scripts/compare_regenerated.py \
  --legacy-project "$H3_LEGACY_PROJECT" --legacy-artifact "$H3_GOLDEN" \
  --calibration "$H3_CALIBRATION" --source-receipt "$H3_SOURCE_RECEIPT" \
  --binding-receipt "$H3_RUN/binding.json" --trt-library "$H3_TRT_LIBRARY" \
  --compat-pack nested_rcp_approx --output "$H3_REGEN"

python scripts/summarize_regeneration.py \
  --input "$H3_REGEN" --output "$H3_RUN/regeneration-summary.json"
```

For a smoke run, use a separate new output directory and add `--start 0 --stop 1`
to regeneration. That result is a slice, not a full-model acceptance. Production
comparison requires all 208 layers; sliced runs keep absolute seed ordinals.

The driver processes one projection at a time and writes:

| Output | Meaning |
| --- | --- |
| `layers.jsonl`, `complete.json` | Per-layer measurements and full/slice completion scope |
| `compat_modelopt_candidate/` | Actual compatibility-QTensor export; intended candidate for this recipe |
| `native_modelopt_candidate/` | Separate default ModelOpt pack comparison; not presumed equivalent |

Before inference, require 208 layers, full coverage, matched source/calibration
hashes, and
`complete.json: compat_candidate.observed_packed_and_output_exact == true`.
Check `compat_candidate.full_208_layers` as well. The top-level `all-208-exact`
status alone describes the legacy-factor/legacy-packer gate, **not** acceptance
of the native candidate or full generation.

Optionally, `scripts/roundtrip_legacy.py` accepts the same four binding paths
and a fresh `--output` to test packed-only ModelOpt checkpoint roundtrip.
That preserves supplied bytes; it does not regenerate quantization and is not
a substitute for this step. The included `sglang_bridge` accepts only that
packed-roundtrip output, not the regenerated candidate directory.

### 5. Run BF16 and matched PTQ inference (external integration)

Keep the original BF16 model and non-target components available. These packed
files are a **hybrid overlay**, not a complete standalone H3 checkpoint. The
existing hooked SGLang runtime performs activation smoothing/quantization,
NVFP4 residual GEMM, and BF16 low-rank compensation using the same custom TRT
backend as the reference. Attention computation and communication are unchanged.

The historical wrappers below use two B200s, TP1 x Ulysses2. Full packed weights
are not TP-sharded: do not enable TP2 or assume offload, arbitrary concurrency,
other GPU architectures, or compilation modes are supported. Kernel mode retains
original BF16 weights, so packed weight size is not the model's peak memory.

First create a fresh candidate launch receipt on CPU. This external validator
checks actual old/new tensor payloads, all regeneration records, and the runtime
binding; do not copy the reference's validation receipt onto the candidate:

```bash
python "$H3_LEGACY_PROJECT/scripts/check_regenerated_compat.py" \
  --root "$H3_LEGACY_PROJECT" --modelopt-project "$H3_EXAMPLE" \
  --legacy-artifact "$H3_GOLDEN" --calibration "$H3_CALIBRATION" \
  --source-receipt "$H3_SOURCE_RECEIPT" --binding-receipt "$H3_RUN/binding.json" \
  --regeneration "$H3_REGEN" --output "$H3_RUN/launch.json"

python "$H3_LEGACY_PROJECT/scripts/run_case.py" \
  --root "$H3_LEGACY_PROJECT" --mode bf16 \
  --gpus 2 --port 30010 --output "$H3_RUN/bf16"

python "$H3_LEGACY_PROJECT/scripts/run_regenerated_case.py" \
  --root "$H3_LEGACY_PROJECT" --role legacy --mode kernel \
  --launch-receipt "$H3_RUN/launch.json" --weights "$H3_GOLDEN" \
  --gpus 2 --port 30010 --output "$H3_RUN/legacy-ptq"

python "$H3_LEGACY_PROJECT/scripts/run_regenerated_case.py" \
  --root "$H3_LEGACY_PROJECT" --role candidate --mode kernel \
  --launch-receipt "$H3_RUN/launch.json" \
  --weights "$H3_REGEN/compat_modelopt_candidate" \
  --gpus 2 --port 30010 --output "$H3_RUN/modelopt-ptq"
```

Run these sequentially, allowing each owned server to exit before reusing its
port. The wrappers use the prepared evaluation manifest; they do not accept
arbitrary prompts through these command lines. Both PTQ arms use the same
wrapper, kernel, input reference ordering, prompt, seed, sampler, resolution,
frame count, and audio configuration. Never label a kernel-mode run as BF16.

### 6. Evaluate compatibility, quality, and performance separately

- **Compatibility:** compare the legacy PTQ and regenerated PTQ runs. Decode
  every video frame and the entire audio stream; check shapes, timestamps,
  frame count, RGB and PCM differences, not just file size or selected frames.
  A changed MP4 container alone does not imply changed decoded content.
- **Quality:** compare both PTQ arms against BF16 on held-out inputs. Inspect
  reference identity, motion, temporal consistency, prompt adherence, and audio.
  Numerical closeness or equality between the two PTQ arms is not BF16 quality
  acceptance. Keep calibration data out of this evaluation.
- **Performance:** exclude calibration, factorization, hashing, model loading,
  and first-use compilation from warm inference timing. Record cold runs
  separately; warm up the exact request configuration and repeat measurements.
  The wrapper commands above establish comparison runs, not a steady-state
  speedup benchmark.

Record source revisions, all artifact/receipt hashes, runtime/library versions,
GPU count, parallelism, actual DiT calls, frame count, resolution, video/audio
metrics, and peak-memory measurement scope with each result. The historical
matched case was 124 frames, 768x1344, 24 FPS, seed42, and 49 observed DiT calls
per rank. It does not certify other prompts, durations, or configurations.

## Evidence and limitations

Historical fixed-environment checks before packaging established:

- 208 real layers: regenerated compatibility factors and packed payloads matched.
- 624 real same-runtime GEMM comparisons (three row counts per layer): exact.
- One matched 124-frame Ref2VA evaluation: decoded RGB and audio PCM matched
  between original PTQ and the compatibility export.

These are bounded historical observations, not fresh GPU tests of this branch,
generalization to all prompts, perceptual equivalence to BF16, or a speedup claim.
Native/default ModelOpt quantization had measured numerical differences and is
reported separately from compatibility mode. Full-generation data and raw reports
are intentionally not published; readers must generate their own validation evidence.

This directory contains only code, tests, and documentation. No prompts, reference
media, generated media, model weights, calibration tensors, credentials, cluster
scripts, or unrelated experiments are included.
