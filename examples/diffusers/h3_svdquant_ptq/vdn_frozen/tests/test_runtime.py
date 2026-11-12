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

import pytest
import torch
from vdn_frozen.runtime import Dispatch, bf16_group, validate_payload


def payload():
    return {
        "weight": torch.zeros((128, 32), dtype=torch.uint8),
        "weight_scale": torch.zeros((128, 4), dtype=torch.uint8),
        "weight_global_scale": torch.tensor([2.0]),
        "input_scale": torch.tensor([4.0]),
        "alpha": torch.tensor([0.125]),
        "pre_quant_scale": torch.ones(64, dtype=torch.bfloat16),
        "svdquant_lora_a": torch.zeros((32, 64), dtype=torch.bfloat16),
        "svdquant_lora_b": torch.zeros((128, 32), dtype=torch.bfloat16),
    }


def test_payload_validation_does_not_change_values():
    original = payload()
    saved = {key: value.clone() for key, value in original.items()}
    validate_payload(original, (128, 64))
    assert all(torch.equal(value, original[key]) for key, value in saved.items())


@pytest.mark.parametrize("field", ["rank", "dtype", "alpha", "packing", "extra"])
def test_reject_incompatible_payload(field):
    data = payload()
    if field == "rank":
        data["svdquant_lora_a"] = torch.zeros((16, 64), dtype=torch.bfloat16)
    if field == "dtype":
        data["svdquant_lora_b"] = data["svdquant_lora_b"].float()
    if field == "alpha":
        data["alpha"] *= 2
    if field == "packing":
        data["weight"] = data["weight"].float()
    if field == "extra":
        data["new_branch"] = torch.tensor(1)
    with pytest.raises(ValueError):
        validate_payload(data, (128, 64))


def test_dispatch_union_and_reset():
    d = Dispatch(["qkv", "fc1"])
    for _ in range(2):
        d.start(bf16_steps=[0], bf16_groups=["qkv"])
        for step in range(8):
            d.before(None, None)
            assert d.route("qkv") == "bf16"
            assert d.route("fc1") == ("bf16" if step == 0 else "nvfp4")
            d.after(None, None, True)
        counts = d.finish()
        assert counts["qkv"] == {"bf16": 8}
        assert counts["fc1"] == {"bf16": 1, "nvfp4": 7}


def test_dispatch_fail_closed():
    d = Dispatch(["qkv"])
    with pytest.raises(ValueError):
        d.start(bf16_steps=[8])
    with pytest.raises(ValueError):
        d.start(bf16_groups=["q_only"])
    d.start()
    d.before(None, None)
    d.route("qkv")
    with pytest.raises(RuntimeError):
        d.route("qkv")
    with pytest.raises(RuntimeError):
        d.finish()


def test_bf16_qkv_keeps_original_three_gemms():
    layers = [torch.nn.Linear(16, 8, bias=False) for _ in range(3)]
    x = torch.randn(129, 16)
    parts = bf16_group(layers, x).chunk(3, -1)
    assert all(torch.equal(part, layer(x)) for part, layer in zip(parts, layers))
