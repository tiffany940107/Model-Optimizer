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

"""CPU selection/joint-QKV contracts, not NVFP4 GPU validation."""

import pytest
import torch
from vdn_frozen.policy import JointQKV, groups, select


def test_scope():
    items = groups()
    assert len(items) == len({item.name for item in items}) == 208
    names = [name for item in items for name in item.modules]
    assert len(names) == len(set(names)) == 312
    assert not any("to_out_linear" in name or "gate" in name or "adaln" in name for name in names)
    assert sum("token_refiner" in item.name for item in items) == 8
    assert sum(len(item.modules) == 3 for item in items) == 52


def test_joint_call_exact_views_and_no_stale_reuse():
    weight = torch.randn(12, 16)
    calls = []

    def linear(value):
        calls.append(1)
        return torch.nn.functional.linear(value, weight)

    joint = JointQKV(linear)
    for _ in range(2):
        value = torch.randn(5, 16)
        parts = [joint(i, value) for i in range(3)]
        assert torch.equal(torch.cat(parts, -1), linear(value))
        assert joint.next == 0 and joint.value is None
    assert len(calls) == 4  # one joint compute + one reference per request


def test_joint_rejects_reordered_or_changed_inputs():
    value = torch.ones(1, 3)
    joint = JointQKV(lambda x: x.repeat(1, 3))
    with pytest.raises(RuntimeError, match="order"):
        joint(1, value)
    joint(0, value)
    with pytest.raises(RuntimeError, match="same unchanged"):
        joint(1, value.clone())
    value.add_(1)
    with pytest.raises(RuntimeError, match="same unchanged"):
        joint(1, value)


def test_exact_geometry_without_allocating_full_weights():
    model = torch.nn.Module()
    for group in groups():
        for name in group.modules:
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            n, k = group.shape
            parent.add_module(
                parts[-1],
                torch.nn.Linear(
                    k, n // len(group.modules), bias=False, dtype=torch.bfloat16, device="meta"
                ),
            )
    assert len(select(model)) == 208
    layer = model.get_submodule(groups()[0].modules[0])
    layer.bias = torch.nn.Parameter(torch.empty(layer.out_features, device="meta"))
    with pytest.raises(ValueError, match="Unexpected"):
        select(model)
