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

"""CPU tests of actual quantizer packing, layout conversion and layer RNG schedule."""

import pytest
import torch
from h3_modelopt_ptq import factorize_legacy, import_frozen
from legacy_oracle import load_test_oracle
from recipe_comparison import (
    compare_factorizations,
    layer_schedule,
    pack_modelopt_for_trt,
    reorder_qkv,
    swizzle_scales,
    tensor_errors,
)
from sglang_bridge import _environment
from torch import nn


def core():
    return load_test_oracle()


def test_exporter_schedule_preserves_insertion_position_when_sliced():
    names = ["blocks.9.mlp.fc1", "token_refiner.blocks.1.attn.qkv_proj", "blocks.0.mlp.fc2"]
    assert layer_schedule(names, start=1) == [(1, names[1], 1102), (2, names[2], 1103)]
    with pytest.raises(ValueError):
        layer_schedule(names, start=2, stop=1)


def test_real_same_device_regeneration_matches_bound_legacy():
    torch.manual_seed(42)
    weight = torch.randn(48, 64).bfloat16()
    factors, equality = compare_factorizations(weight, torch.ones(64), core(), seed=1107)
    assert all(equality.values())
    assert factors.lora_a.shape == (32, 64)


@pytest.mark.parametrize(("rows", "cols"), [(32, 32), (128, 64), (129, 80), (256, 128)])
def test_swizzle_inverts_actual_legacy_layout(rows, cols):
    scales = (
        torch.arange(rows * (cols // 16)).remainder(120).to(torch.uint8).view(torch.float8_e4m3fn)
    )
    scales = scales.reshape(rows, cols // 16)
    swizzled = swizzle_scales(scales, rows, cols)
    actual = core()._swizzled_e4m3_to_linear(swizzled, rows, cols)
    assert torch.equal(actual.view(torch.uint8), scales.view(torch.uint8))


def test_actual_modelopt_native_packer_export_decodes_through_legacy_layout():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 48, bias=False)).bfloat16()
    factors = factorize_legacy(model[0].weight, torch.ones(64), seed=1101)
    model = import_frozen(model, {"0": factors}, require_full_h3=False)
    packed, upstream_dequant = pack_modelopt_for_trt(model[0])
    legacy_dequant = core().dequantize_nvfp4(
        packed["weight"], packed["weight_scale"], packed["weight_global_scale"]
    )
    # Validate both exact arithmetic contracts independently. Multiplication by
    # ModelOpt's global scale and division by its reciprocal can straddle BF16
    # rounding boundaries; a layout conversion is not an equivalence proof.
    values = core()._unpack_e2m1(packed["weight"]).reshape(48, 4, 16)
    scales = core()._swizzled_e4m3_to_linear(packed["weight_scale"], 48, 64).float()
    native_global = model[0].weight_quantizer.amax.float() / 2688
    expected_native = (values * (scales * native_global).unsqueeze(-1)).reshape(48, 64).bfloat16()
    expected_trt = (
        (values * (scales / packed["weight_global_scale"]).unsqueeze(-1)).reshape(48, 64).bfloat16()
    )
    assert torch.equal(upstream_dequant, expected_native)
    assert torch.equal(legacy_dequant, expected_trt)
    assert torch.isfinite(upstream_dequant).all()
    assert tensor_errors(upstream_dequant, legacy_dequant)["relative_l2"] >= 0


def test_qkv_reordering_matches_runtime_contract():
    source = torch.arange(56 * 3 * 128).reshape(56 * 3 * 128, 1)
    actual = reorder_qkv(source).reshape(3, 56, 128)
    assert torch.equal(actual, source.reshape(56, 3, 128).permute(1, 0, 2))


def test_reciprocal_dequant_roundtrip_has_a_fixed_bf16_counterexample():
    # Observed in actual ModelOpt CPU pack export; exact FP32 bits make this
    # regression independent of SVD/random inputs or a tolerance choice.
    global_dequant = torch.tensor(954204160, dtype=torch.int32).view(torch.float32)
    value, block_scale = torch.tensor(-3.0), torch.tensor(240.0)
    modelopt_value = (value * (block_scale * global_dequant)).bfloat16()
    trt_value = (value * (block_scale / (1 / global_dequant))).bfloat16()
    assert modelopt_value.item() == -0.0771484375
    assert trt_value.item() == -0.07666015625
    assert modelopt_value != trt_value


def test_sglang_environment_is_child_only_and_clears_previous_quantization(tmp_path):
    original = {
        "KEEP": "unchanged",
        "SGLANG_H3_SVDQUANT_MODE": "collect",
        "SGLANG_H3_SVDQUANT_BF16_BLOCKS": "0-49",
        "SGLANG_H3_SVDQUANT_INCLUDE_ADALN": "1",
    }
    result = _environment(original, tmp_path / "overlay", tmp_path / "kernel.so")
    assert original["SGLANG_H3_SVDQUANT_MODE"] == "collect"
    assert result["KEEP"] == "unchanged"
    assert result["SGLANG_H3_SVDQUANT_MODE"] == "kernel"
    assert "SGLANG_H3_SVDQUANT_BF16_BLOCKS" not in result
    assert result["SGLANG_H3_SVDQUANT_INCLUDE_ADALN"] == "0"
