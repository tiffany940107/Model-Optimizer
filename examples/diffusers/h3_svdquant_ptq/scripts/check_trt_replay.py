# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic GPU gate: legacy TRT -> ModelOpt state -> same TRT kernel.

Does not run calibration, load customer images, or prove full-model quality.
Uses the established legacy runtime as the sole NVFP4 GEMM backend.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import svdquant_h3.core as legacy_core
import torch
from h3_modelopt_ptq import FrozenFactors, export_frozen, factorize_legacy, import_frozen
from native_breakdown import layer_breakdown
from recipe_comparison import pack_modelopt_for_trt
from svdquant_h3.core import ArtifactFakeSVDQuantLinearMethod, TRTNVFP4SVDQuantLinearMethod
from torch import nn

import modelopt.torch.opt as mto


def payload(factors):
    """Return the original TRT runtime's tensor dictionary."""
    return {
        **factors.packed,
        "pre_quant_scale": factors.pre_quant_scale,
        "svdquant_lora_a": factors.lora_a,
        "svdquant_lora_b": factors.lora_b,
    }


def errors(actual, reference):
    """Measure output error without accepting a numerical equivalence claim."""
    diff = actual.float() - reference.float()
    return {
        "max_abs": diff.abs().max().item(),
        "relative_l2": (diff.norm() / reference.float().norm().clamp_min(1e-12)).item(),
    }


@torch.no_grad()
def main():
    """Run the explicit synthetic frozen-state GPU gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-modelopt",
        action="store_true",
        help="Also measure native ModelOpt quantizer error; may JIT CUDA/Triton",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Separate true native packing/activation/low-rank/fused epilogue errors",
    )
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 12)
        or torch.__version__ != "2.11.0+cu130"
        or torch.version.cuda != "13.0"
    ):
        raise RuntimeError(
            "Legacy TRT library requires Python 3.12 / torch 2.11.0+cu130 / CUDA 13.0; do not use QAD runtime"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Initial legacy TRT runtime gate is SM100 only")
    torch.ops.load_library(str(args.trt_library.resolve()))
    torch.manual_seed(1101)
    original = nn.Sequential(nn.Linear(256, 256, bias=False)).to("cuda", torch.bfloat16)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    stats = x.float().abs().amax(dim=0)
    factors = factorize_legacy(original[0].weight, stats, seed=42)
    ws = (2688 / factors.residual.float().abs().amax().clamp_min(1e-12)).reshape(1)
    xs = (2688 / (stats * factors.pre_quant_scale.float()).amax().clamp_min(1e-12)).reshape(1)
    packed, scales = torch.ops.trtllm.fp4_quantize(
        factors.residual.contiguous(), ws, 16, False, True
    )
    factors = FrozenFactors(
        **{
            **vars(factors),
            "packed": {
                "weight": packed.view(torch.uint8),
                "weight_scale": scales.view(torch.uint8),
                "weight_global_scale": ws,
                "input_scale": xs,
                "alpha": 1 / (ws * xs),
            },
        }
    )
    baseline = TRTNVFP4SVDQuantLinearMethod(payload(factors))
    reference = ArtifactFakeSVDQuantLinearMethod(payload(factors))
    baseline.apply(None, x)
    expected = baseline.apply(None, x)
    model = import_frozen(original, {"0": factors}, require_full_h3=False)
    checkpoint = args.output / "modelopt.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(
        nn.Sequential(nn.Linear(256, 256, bias=False)).bfloat16(), checkpoint
    ).cuda()
    reexport = payload(export_frozen(restored[0]))
    for name, tensor in payload(factors).items():
        if not torch.equal(tensor, reexport[name]):
            raise RuntimeError(f"ModelOpt replay changed {name}")
    replay = TRTNVFP4SVDQuantLinearMethod(reexport)
    replay.apply(None, x)
    actual = replay.apply(None, x)
    if not torch.isfinite(actual).all() or not torch.equal(expected, actual):
        raise RuntimeError("Frozen ModelOpt/TRT replay not finite/bitwise identical")
    report = {
        "status": "frozen-replay-passed",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": [128, 256, 256],
        "trt_library_sha256": hashlib.sha256(args.trt_library.read_bytes()).hexdigest(),
        "replay_vs_legacy_trt": errors(actual, expected),
        "legacy_trt_vs_packed_reference": errors(expected, reference.apply(None, x)),
        "scope": "synthetic one-layer frozen state; no H3 E2E or recalibration quality claim",
    }
    if args.native_modelopt or args.breakdown:
        native = restored(x)
        if not torch.isfinite(native).all():
            raise RuntimeError("Native ModelOpt output nonfinite")
        report["native_modelopt_vs_legacy_trt"] = errors(native, expected)
        report["native_modelopt_compatibility"] = "measurement only; not declared equivalent"
        if args.breakdown:
            native_payload, _ = pack_modelopt_for_trt(restored[0])
            native_trt = TRTNVFP4SVDQuantLinearMethod(native_payload).apply(None, x)
            report["native_breakdown"] = layer_breakdown(
                restored[0],
                legacy_core,
                reexport,
                native_payload,
                x,
                old_output=expected,
                native_output=native_trt,
                modelopt_output=native,
            )
    (args.output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
