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

"""CPU coverage of new ModelOpt arithmetic variants; PTX is not emulated."""

import pytest
import torch
from h3_modelopt_ptq import factorize_legacy, import_frozen
from modelopt_multiplier_compat import MultiplierNVFP4QTensor, pack_compatible_layer
from torch import nn

import modelopt.torch.opt as mto


@pytest.mark.parametrize("variant", ["multiplier_div", "multiplier_rcp_rn", "nested_rcp_rn"])
def test_multiplier_formula_matches_explicit_arithmetic(variant):
    x = torch.tensor([[0.75, 1.75, 3.5, 6] * 4], dtype=torch.bfloat16)
    q, scale, global_scale, normalized = MultiplierNVFP4QTensor.quantize(
        x, torch.tensor([448.0]), variant=variant
    )
    assert scale.item() == 448
    assert torch.equal(normalized, x.float())
    assert q._quantized_data[0, :2].tolist() == [66, 118]
    assert torch.equal(global_scale, torch.tensor([1 / 448]))


def test_zero_scale_and_negative_zero_are_explicit():
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    x[:, 0] = -0.0
    q, scale, _, _ = MultiplierNVFP4QTensor.quantize(
        x, torch.tensor([448.0]), variant="nested_rcp_rn"
    )
    assert scale.float().item() == 0
    assert q._quantized_data[0, 0].item() == 8
    assert q._quantized_data[0, 1:].sum().item() == 0


def test_cpu_does_not_silently_emulate_ptx():
    with pytest.raises(ValueError, match="no CPU fallback"):
        MultiplierNVFP4QTensor.quantize(torch.ones(1, 16).bfloat16(), torch.tensor([448.0]))


def test_nonzero_scale_underflow_fails_closed():
    with pytest.raises(ValueError, match="underflow"):
        MultiplierNVFP4QTensor.quantize(
            torch.full((1, 16), 1e-8).bfloat16(), torch.tensor([1.0]), variant="nested_rcp_rn"
        )


def test_real_modelopt_state_retains_multiplier_and_pack_after_restore(tmp_path):
    torch.manual_seed(15)
    model = nn.Sequential(nn.Linear(64, 48, bias=False)).bfloat16()
    factors = factorize_legacy(model[0].weight, torch.ones(64), seed=1101)
    model = import_frozen(model, {"0": factors}, require_full_h3=False)
    before, _ = pack_compatible_layer(model[0], variant="nested_rcp_rn")
    assert hasattr(model[0].weight_quantizer, "_h3_compat_multiplier")
    checkpoint = tmp_path / "modelopt.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(nn.Sequential(nn.Linear(64, 48, bias=False)).bfloat16(), checkpoint)
    after, _ = pack_compatible_layer(restored[0], variant="nested_rcp_rn")
    assert all(torch.equal(before[key], after[key]) for key in before)
