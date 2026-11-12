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

"""CPU route/cache integration tests; no claims of NVFP4 kernel execution."""

import pytest
import torch
from vdn_frozen import policy
from vdn_frozen.environment import Method
from vdn_frozen.pipeline import validate_cache
from vdn_frozen.runtime import Dispatch, RoutedMethod, bf16_group


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16) for _ in range(3)]
        )
        self.output = torch.nn.Linear(24, 8, bias=False, dtype=torch.bfloat16)

    def forward(self, value):
        return self.output(torch.cat([layer(value) for layer in self.qkv], dim=-1))


class ConstantMethod:
    """Deliberate CPU stand-in to distinguish dispatch paths, not quantization."""

    def __init__(self, width):
        self.width, self.calls = width, 0

    def apply(self, _layer, value):
        self.calls += 1
        return torch.full((*value.shape[:-1], self.width), 2, dtype=value.dtype)


def test_real_adapter_runs_all_eight_steps_with_two_bf16_evaluations(monkeypatch):
    model = TinyTransformer()
    selected = {"qkv": list(model.qkv), "output": [model.output]}
    monkeypatch.setattr(policy, "select", lambda _: selected)
    value = torch.randn(1, 7, 8, dtype=torch.bfloat16)
    teacher = model(value)
    methods = {"qkv": ConstantMethod(24), "output": ConstantMethod(8)}
    dispatch = Dispatch(selected)
    routes = {
        name: RoutedMethod(name, layers, methods[name], dispatch)
        for name, layers in selected.items()
    }
    policy.install(model, routes)
    model.register_forward_pre_hook(dispatch.before)
    model.register_forward_hook(dispatch.after, always_call=True)
    for _ in range(2):
        dispatch.start(bf16_steps=(0, 5), bf16_groups=())
        for step in range(8):
            output = model(value)
            if step in (0, 5):
                assert torch.equal(output, teacher)
            else:
                assert torch.equal(output, torch.full_like(output, 2))
        assert dispatch.finish() == {name: {"bf16": 2, "nvfp4": 6} for name in selected}
    assert all(method.calls == 12 for method in methods.values())
    assert torch.equal(
        bf16_group(
            [model.output],
            torch.cat(
                [torch.nn.functional.linear(value, layer.weight) for layer in model.qkv], dim=-1
            ),
        ),
        teacher,
    )


def test_runtime_down_and_alpha_folding_are_unchanged_cpu_math():
    payload = {
        "weight": torch.zeros((128, 32), dtype=torch.uint8),
        "weight_scale": torch.zeros((128, 4), dtype=torch.uint8),
        "input_scale": torch.tensor([2.0]),
        "alpha": torch.tensor([0.125]),
        "pre_quant_scale": torch.randn(64).bfloat16(),
        "svdquant_lora_a": torch.randn(32, 64).bfloat16(),
        "svdquant_lora_b": torch.randn(128, 32).bfloat16(),
    }
    method = Method(payload)
    assert torch.equal(
        method.lora_down, (payload["pre_quant_scale"].unsqueeze(1) * payload["svdquant_lora_a"].T)
    )
    assert torch.equal(
        method.up, (payload["svdquant_lora_b"].float() / payload["alpha"]).bfloat16()
    )


def cache():
    return {
        "height": 1344,
        "width": 768,
        "reference_size": 768,
        "keyframe_anchors": ["ref"] * 3,
        "prompt_embeds": torch.zeros(2, 5120, dtype=torch.bfloat16),
        "text_token_tags": torch.tensor([0, 1]),
        "condition_latents": [torch.zeros(1, 24, 1, 48, 48) for _ in range(3)],
    }


@pytest.mark.parametrize("mutation", ["geometry", "references", "nan", "text"])
def test_cache_boundary_rejects_incompatible_conditioning(tmp_path, mutation):
    value = cache()
    path = tmp_path / "prompt.pt"
    torch.save(value, path)
    validate_cache(path)
    if mutation == "geometry":
        value["width"] = 1344
    elif mutation == "references":
        value["keyframe_anchors"] = ["first"]
    elif mutation == "nan":
        value["condition_latents"][0].fill_(float("nan"))
    else:
        value["prompt_embeds"] = value["prompt_embeds"].float()
    torch.save(value, path)
    with pytest.raises(ValueError):
        validate_cache(path)
