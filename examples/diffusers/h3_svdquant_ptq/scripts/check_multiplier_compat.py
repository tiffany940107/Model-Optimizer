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

"""Isolated synthetic GPU probe of new ModelOpt multiplier arithmetic variants."""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import torch
from h3_modelopt_ptq import FrozenFactors, factorize_legacy, import_frozen
from modelopt_multiplier_compat import VARIANTS, pack_compatible_layer
from native_breakdown import byte_comparison
from recipe_comparison import pack_for_trt, pack_modelopt_for_trt, tensor_errors
from svdquant_h3.core import TRTNVFP4SVDQuantLinearMethod
from torch import nn

import modelopt.torch.opt as mto


def mismatch_examples(actual, expected, weight, normalized, limit=24):
    """Record original values and normalized positions near E2M1 boundaries."""

    def unpack(packed):
        return torch.stack((packed & 15, packed >> 4), dim=-1).reshape_as(weight)

    a, b = unpack(actual), unpack(expected)
    indices = (a != b).nonzero()[:limit]
    return [
        {
            "row": int(i),
            "column": int(j),
            "input_bf16": weight[i, j].item(),
            "normalized": normalized[i, j].item(),
            "candidate_code": int(a[i, j]),
            "gold_code": int(b[i, j]),
        }
        for i, j in indices.cpu().tolist()
    ]


@torch.no_grad()
def main():
    """Measure candidates without changing the already measured native baseline."""
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
        raise RuntimeError("Probe requires original Python3.12/torch2.11+cu130 SM100 runtime")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.ops.load_library(str(args.trt_library.resolve()))
    torch.manual_seed(1101)
    model = nn.Sequential(nn.Linear(256, 256, bias=False)).to("cuda", torch.bfloat16)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    factors = factorize_legacy(model[0].weight, x.float().abs().amax(0), seed=42)
    gold = pack_for_trt(factors)
    packed = {
        key: value
        for key, value in gold.items()
        if key in {"weight", "weight_scale", "weight_global_scale", "input_scale", "alpha"}
    }
    factors = FrozenFactors(**{**vars(factors), "packed": packed})
    model = import_frozen(model, {"0": factors}, require_full_h3=False)
    original = TRTNVFP4SVDQuantLinearMethod(gold)
    original.apply(None, x)
    expected = original.apply(None, x)
    native, _ = pack_modelopt_for_trt(model[0])
    report = {
        "status": "measured-not-accepted",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "library_sha256": hashlib.sha256(args.trt_library.read_bytes()).hexdigest(),
        "native_baseline": {
            "data": byte_comparison(native["weight"], gold["weight"]),
            "scales": byte_comparison(native["weight_scale"], gold["weight_scale"]),
        },
        "variants": {},
    }
    for variant in VARIANTS:
        candidate, normalized = pack_compatible_layer(model[0], variant=variant)
        method = TRTNVFP4SVDQuantLinearMethod(candidate)
        method.apply(None, x)
        output = method.apply(None, x)
        report["variants"][variant] = {
            "data": byte_comparison(candidate["weight"], gold["weight"]),
            "scales": byte_comparison(candidate["weight_scale"], gold["weight_scale"]),
            "same_trt_output": tensor_errors(output, expected),
            "mismatch_examples": mismatch_examples(
                candidate["weight"], gold["weight"], model[0].weight, normalized
            ),
        }
        print(json.dumps({"variant": variant, **report["variants"][variant]}), flush=True)
    buffer = io.BytesIO()
    mto.save(model, buffer)
    buffer.seek(0)
    restored = mto.restore(nn.Sequential(nn.Linear(256, 256, bias=False)).bfloat16(), buffer).cuda()
    buffer.close()
    replay, _ = pack_compatible_layer(restored[0], variant="nested_rcp_approx")
    reference, _ = pack_compatible_layer(model[0], variant="nested_rcp_approx")
    if any(not torch.equal(replay[key], reference[key]) for key in replay):
        raise RuntimeError("New multiplier buffer/pack path changed across ModelOpt save/restore")
    report["modelopt_state_replay_exact"] = True
    report["scope"] = (
        "synthetic one-layer differential arithmetic probe; not full H3 quality or universal compatibility"
    )
    with (args.output / "complete.json").open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
