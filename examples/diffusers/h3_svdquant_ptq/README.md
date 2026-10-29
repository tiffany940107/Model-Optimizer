# H3 SVDQuant NVFP4 PTQ compatibility example

This experimental, inference-only example reproduces an existing H3 SVDQuant
recipe through ModelOpt modules, checkpoint state, and a custom NVFP4 QTensor
export path. It is **not** the default ModelOpt SVDQuant algorithm, QAD training,
or a self-contained H3 inference engine. No ModelOpt core files are modified.

## Version and scope

The tested base is ModelOpt commit
`881cacdbeed36ab3ca16c4cb8bc56606a42329c9` from the
[`jingyu-ml/Model-Optimizer` fork](https://github.com/jingyu-ml/Model-Optimizer).
This branch intentionally preserves that base; compatibility with current
upstream main has not been established.

The selector targets the original H3 Ref2VA architecture, **not** the separate
FastH3 V2 QAD implementation:

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
The scripts enforce that ABI for runtime replay. **Do not load that runtime in a
Torch 2.12 QAD environment.** No binary, internal kernel source, or runtime library
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
scripts, or unrelated QAD/attention experiments are included.
