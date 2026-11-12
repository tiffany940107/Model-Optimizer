# `vdn_frozen`: code map and quick start

Implementation of the VDN H3 eight-step Frozen NVFP4 PTQ recipe. The recipe,
dependency versions, validation scope and publication boundary are documented in
the parent [README](../README.md); read that first. This file only maps the code.

## Entry points

Run from `examples/diffusers/h3_svdquant_ptq` with `PYTHONPATH` including this
directory and the VDN checkout (see parent README, step 0).

| Command | Purpose |
| --- | --- |
| `python -m vdn_frozen.encode` | Encode a prompt and three references into a `prompt.pt` cache |
| `python -m vdn_frozen.cli init` | Bind inputs/sources/library by hash (CPU only) |
| `python -m vdn_frozen.cli calibrate` | BF16 eight-NFE teachers and pooled absmax |
| `python -m vdn_frozen.cli export-baseline` | Original rank-32 SVDQuant NVFP4 overlay |
| `python -m vdn_frozen.cli reconstruct` | Fit B per group, gate on dev and full-block replay |
| `python -m vdn_frozen.cli export-frozen` | B-corrected overlay; the original residual is kept |
| `python -m vdn_frozen.cli render --mode bf16\|frozen` | Matched BF16 / Frozen video |
| `bash scripts/reproduce_frozen.sh [--from-cache]` | All stages in sequence; needs the `H3_*` environment variables, see `--help` |

## Modules

| Module | Role |
| --- | --- |
| `cli.py` | Stage commands and run-directory state |
| `encode.py` | Prompt/reference encoding using VDN's own helpers |
| `context.py`, `environment.py` | Model assembly, source and environment checks |
| `policy.py` | Selected groups (208 logical, 312 physical) and BF16 steps 0 and 5 |
| `quantization.py` | Baseline calibration and packing |
| `fit.py`, `reconstruct.py` | Real-kernel ridge fit of B and candidate gating |
| `block_replay.py` | Full-block replay gate at NFEs 0/3/7 |
| `export_reconstructed.py` | Frozen overlay export and packed-field checks |
| `runtime.py`, `pipeline.py` | Per-step BF16/NVFP4 dispatch and rendering |

## Tests

```bash
python -m pytest -c /dev/null -p no:cacheprovider vdn_frozen/tests -q
```

The tests are CPU-only. They do not replace a GPU rerun on your own inputs.
Do not commit generated receipts, tensors, media or runtime binaries.
