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

"""CPU representation/recipe tests; these are not CUDA GEMM accuracy evidence."""

import copy
from pathlib import Path

import pytest
import torch
from h3_modelopt_ptq import (
    FrozenFactors,
    export_frozen,
    factorize_legacy,
    h3_layer_names,
    import_frozen,
    modelopt_cpu_reference,
    pack_modelopt_cpu,
)
from legacy_oracle import load_test_oracle
from torch import nn

import modelopt.torch.opt as mto
from modelopt.torch.quantization.nn.modules.quant_linear import SVDQuantLinear

ROOT = Path(__file__).resolve().parents[1]


def legacy_module():
    return load_test_oracle()


def fixture():
    torch.manual_seed(551)
    model = nn.Sequential(nn.Linear(64, 48, bias=True), nn.Linear(48, 16)).bfloat16()
    stats = torch.rand(64).float() * 3
    factors = factorize_legacy(model[0].weight, stats, seed=42, rank=16)
    return model, factors


def assert_factors_equal(left, right):
    for name in ("pre_quant_scale", "residual", "lora_a", "lora_b", "activation_absmax"):
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype
        assert torch.equal(a, b), name
    assert (left.packed is None) == (right.packed is None)
    if left.packed is not None:
        for key in left.packed:
            assert torch.equal(left.packed[key], right.packed[key]), key


def test_exact_h3_selector():
    names = h3_layer_names()
    assert len(names) == len(set(names)) == 208
    assert "blocks.49.attn.qkv_proj" in names
    assert "token_refiner.blocks.1.mlp.fc2" in names
    assert not any("adaln" in name or "gate" in name or "encoder" in name for name in names)


def test_full_208_layer_scope_converts_and_leaves_adaln_disabled():
    def block():
        result = nn.Module()
        result.attn = nn.ModuleDict({"qkv_proj": nn.Linear(16, 16), "out_proj": nn.Linear(16, 16)})
        result.mlp = nn.ModuleDict({"fc1": nn.Linear(16, 16), "fc2": nn.Linear(16, 16)})
        result.adaln_proj = nn.ModuleDict({"linear": nn.Linear(16, 16)})
        return result

    model = nn.Module()
    model.blocks = nn.ModuleList([block() for _ in range(50)])
    model.token_refiner = nn.Module()
    model.token_refiner.blocks = nn.ModuleList([block() for _ in range(2)])
    model.bfloat16()
    stats = torch.ones(16)
    factors = factorize_legacy(model.blocks[0].attn.qkv_proj.weight, stats, seed=11, rank=4)
    imported = import_frozen(model, dict.fromkeys(h3_layer_names(), factors))
    enabled = [
        name
        for name, module in imported.named_modules()
        if isinstance(module, SVDQuantLinear) and module.weight_quantizer.is_enabled
    ]
    assert set(enabled) == set(h3_layer_names())
    assert not imported.blocks[0].adaln_proj.linear.weight_quantizer.is_enabled


def test_zero_blocks_cpu_packing_are_finite():
    tensor = torch.zeros(2, 32, dtype=torch.bfloat16)
    packed, scale, global_scale = pack_modelopt_cpu(tensor, torch.tensor(0.0))
    output = packed.dequantize(scale=scale, double_scale=global_scale, block_sizes={-1: 16})
    assert torch.isfinite(output).all()
    assert torch.equal(output, tensor)


def test_factor_recipe_matches_original_and_preserves_rng():
    model, factors = fixture()
    state = torch.random.get_rng_state()
    generated = factorize_legacy(model[0].weight, factors.activation_absmax, seed=42, rank=16)
    assert torch.equal(state, torch.random.get_rng_state())
    with torch.random.fork_rng():
        torch.manual_seed(42)
        original = legacy_module().factorize_svdquant(
            model[0].weight, factors.activation_absmax, rank=16, quantize_residual=False
        )
    for name in ("pre_quant_scale", "residual", "lora_a", "lora_b"):
        assert torch.equal(getattr(original, name), getattr(generated, name)), name


def test_real_modelopt_save_restore_and_cpu_packer(tmp_path):
    model, factors = fixture()
    untouched = model[1].weight.clone()
    converted = import_frozen(model, {"0": factors}, require_full_h3=False)
    assert isinstance(converted[0], SVDQuantLinear)
    assert converted[0].input_quantizer.is_enabled
    assert converted[0].weight_quantizer.is_enabled
    assert not converted[1].input_quantizer.is_enabled
    assert not converted[1].weight_quantizer.is_enabled
    assert torch.equal(untouched, converted[1].weight)
    assert not any(p.requires_grad for p in converted.parameters())
    x = torch.randn(5, 64).bfloat16()
    before = modelopt_cpu_reference(converted[0], x)
    assert torch.isfinite(before).all()
    checkpoint = tmp_path / "modelopt.pt"
    mto.save(converted, checkpoint)
    restored = mto.restore(
        nn.Sequential(nn.Linear(64, 48), nn.Linear(48, 16)).bfloat16(), checkpoint
    )
    assert isinstance(restored[0], SVDQuantLinear)
    assert_factors_equal(factors, export_frozen(restored[0]))
    assert torch.equal(before, modelopt_cpu_reference(restored[0], x))
    assert not any(p.requires_grad for p in restored.parameters())
    assert mto.modelopt_state(restored)["modelopt_state_dict"][0][0] == "h3_svdquant_legacy_v1"


def test_frozen_packed_bytes_survive_modelopt_restore(tmp_path):
    model, factors = fixture()
    # Synthetic opaque payload only: tests storage, not TRT numerical validity.
    payload = {
        "weight": torch.randint(0, 256, (48, 32), dtype=torch.uint8),
        "weight_scale": torch.randint(0, 256, (512,), dtype=torch.uint8),
        "weight_global_scale": torch.tensor([3.0]),
        "input_scale": torch.tensor([7.0]),
        "alpha": torch.tensor([1 / 21]),
    }
    factors = FrozenFactors(**{**vars(factors), "packed": payload})
    converted = import_frozen(model, {"0": factors}, require_full_h3=False)
    checkpoint = tmp_path / "packed.pt"
    mto.save(converted, checkpoint)
    restored = mto.restore(
        nn.Sequential(nn.Linear(64, 48), nn.Linear(48, 16)).bfloat16(), checkpoint
    )
    assert_factors_equal(factors, export_frozen(restored[0]))


def test_modelopt_tie_behavior_differs_from_legacy_fake():
    values = torch.tensor([[0.75, 1.75, 3.5, 6.0] * 4], dtype=torch.float32)
    q, scale, global_scale = pack_modelopt_cpu(values, torch.tensor(6.0))
    native = q.dequantize(scale=scale, double_scale=global_scale, block_sizes={-1: 16})
    old_fake = legacy_module().fake_nvfp4(values)
    assert native[0, :4].tolist() == [1.0, 2.0, 4.0, 6.0]
    assert old_fake[0, :4].tolist() == [0.5, 1.5, 3.0, 6.0]
    assert not torch.equal(native, old_fake)


def test_scope_gate_and_validation_before_mutation():
    model, factors = fixture()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="208"):
        import_frozen(model, {"0": factors})
    bad = FrozenFactors(**{**vars(factors), "pre_quant_scale": torch.zeros(64).bfloat16()})
    with pytest.raises(ValueError, match="positive"):
        import_frozen(model, {"0": bad}, require_full_h3=False)
    assert type(model[0]) is nn.Linear
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_nan_calibration_rejected():
    model, factors = fixture()
    factors.activation_absmax[0] = float("nan")
    with pytest.raises(ValueError, match="Invalid calibration"):
        factorize_legacy(model[0].weight, factors.activation_absmax, seed=42)
