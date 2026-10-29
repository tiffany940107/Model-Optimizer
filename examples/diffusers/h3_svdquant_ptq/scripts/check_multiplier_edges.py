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

"""Deterministic GPU edge probes for ModelOpt multiplier packing, not video quality."""

import argparse
import json
import sys
from pathlib import Path

import torch
from artifact_binding import digest
from modelopt_multiplier_compat import MultiplierNVFP4QTensor
from native_breakdown import byte_comparison
from recipe_comparison import swizzle_scales


def edge_inputs():
    """Build deterministic CPU BF16 tensors, including explicit unsupported inputs."""
    shape = (128, 64)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    zeros[:, ::2] = -0.0
    ties = torch.tensor(
        [0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.25, -0.75, -1.75, -3.5, -5, -6],
        dtype=torch.bfloat16,
    ).repeat(128, 4)
    near = torch.nextafter(ties, torch.full_like(ties, float("inf")))
    near[:, 15::16] = -6
    near[:, 9::16] = 6
    generator = torch.Generator().manual_seed(20261028)
    random = torch.randn(shape, generator=generator).bfloat16()
    cases: list[tuple[str, torch.Tensor, torch.Tensor, str | None]] = [
        ("signed_zero", zeros, torch.tensor([448.0]), None),
        ("e2m1_ties", ties, torch.tensor([448.0]), None),
        ("bf16_neighbors", near, torch.tensor([448.0]), None),
    ]
    for exponent in (-12, -6, 0, 6, 12):
        weight = (random.float() * (10.0**exponent)).bfloat16()
        multiplier = (2688 / weight.float().abs().amax().clamp_min(1e-12)).reshape(1)
        cases.append((f"random_scale_1e{exponent}", weight, multiplier, None))
    underflow = torch.full(shape, 6.0, dtype=torch.bfloat16)
    underflow[:, :16] = 1e-20
    cases.append(("nonzero_scale_underflow", underflow, torch.tensor([448.0]), "underflow"))
    nonfinite = random.clone()
    nonfinite[0, 0] = float("nan")
    cases.append(("nonfinite_input", nonfinite, torch.tensor([448.0]), "finite"))
    return cases


@torch.no_grad()
def main():
    """Compare bytes with the true TRT packer as a reference only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 12)
        or torch.__version__ != "2.11.0+cu130"
        or torch.version.cuda != "13.0"
        or torch.cuda.get_device_capability() != (10, 0)
    ):
        raise RuntimeError("Requires original Python3.12/torch2.11+cu130 B200 runtime")
    library_hash = digest(args.trt_library)
    if library_hash != "fe27188368a94eade998e54532e1f6de04bf9f38537ebf0bac528e3ebcdf50e2":
        raise ValueError("Unexpected golden library")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.ops.load_library(str(args.trt_library.resolve()))
    results = []
    for name, weight, multiplier, expected_rejection in edge_inputs():
        weight, multiplier = weight.cuda(), multiplier.cuda()
        record = {"case": name, "shape": list(weight.shape), "multiplier": multiplier.item()}
        try:
            packed, scales, _, normalized = MultiplierNVFP4QTensor.quantize(weight, multiplier)
        except ValueError as error:
            if expected_rejection is None or expected_rejection not in str(error):
                raise
            record.update(status="explicitly-unsupported-rejected", reason=str(error), passed=True)
        else:
            if expected_rejection is not None:
                raise RuntimeError(f"Unsupported case was silently accepted: {name}")
            gold_data, gold_scales = torch.ops.trtllm.fp4_quantize(
                weight, multiplier, 16, False, True
            )
            candidate_scales = swizzle_scales(scales, *weight.shape)
            data_result = byte_comparison(packed._quantized_data, gold_data.view(torch.uint8))
            scale_result = byte_comparison(candidate_scales, gold_scales.view(torch.uint8))
            record.update(
                status="measured",
                data=data_result,
                scales=scale_result,
                passed=data_result["exact"] and scale_result["exact"],
                normalized_finite=bool(torch.isfinite(normalized).all()),
            )
        results.append(record)
        print(json.dumps(record), flush=True)
    passed = all(row["passed"] for row in results)
    report = {
        "status": "edge-probe-passed-not-full-model" if passed else "edge-probe-failed",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "library_sha256": library_hash,
        "variant": "nested_rcp_approx",
        "cases": results,
        "scope": (
            "Supported edge bytes compared exactly; rejected inputs remain unsupported, not compatibility evidence"
        ),
    }
    with (args.output / "complete.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
