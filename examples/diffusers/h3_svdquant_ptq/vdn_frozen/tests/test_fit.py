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
from vdn_frozen.fit import (
    Statistics,
    improves,
    losses,
    replace_b,
    sample_indices,
    solve_delta,
    updated_b,
)


def test_ridge_recovers_known_correction():
    torch.manual_seed(1)
    d = torch.randn(257, 32)
    delta = torch.randn(13, 32) * 0.01
    e = d @ delta.T
    s = Statistics(13, "cpu")
    s.add(d, e, e + 1, {"one": (0, 129), "two": (129, 257)})
    result = solve_delta(s.h, s.g, 1e-5)
    assert torch.allclose(result, delta, rtol=1e-3, atol=1e-6)


def test_balanced_sampling_and_tail():
    mods = {
        name: torch.arange(i * 50, (i + 1) * 50)
        for i, name in enumerate(("video", "reference", "audio", "text"))
    }
    indices, parts = sample_indices(200, mods)
    assert indices.numel() == 129
    assert {k: b - a for k, (a, b) in parts.items()} == {
        "video": 33,
        "reference": 32,
        "audio": 32,
        "text": 32,
    }
    assert sample_indices(83, mods, refiner=True)[0].max() == 82
    with pytest.raises(ValueError):
        sample_indices(100, mods)


def test_frozen_schema_and_trust_region():
    original = torch.ones(17, 32, dtype=torch.bfloat16)
    result = updated_b(original, torch.full((17, 32), 100.0), 1)
    assert result.dtype == torch.bfloat16 and result.shape == original.shape
    assert (result.float() - original.float()).norm() / original.float().norm() <= 0.1
    payload = {"svdquant_lora_b": original, "weight": torch.ones(1)}
    changed = replace_b(payload, result)
    assert changed["weight"] is payload["weight"] and payload["svdquant_lora_b"] is original
    with pytest.raises(ValueError):
        replace_b(payload, result.float())
    with pytest.raises(ValueError):
        updated_b(original[:, :16], torch.zeros(17, 16), 1)


def test_candidate_gate_does_not_hide_modality_regression():
    assert improves({"video": 0.8, "audio": 0.9}, {"video": 1, "audio": 1})
    assert not improves({"video": 0.5, "audio": 1.02}, {"video": 1, "audio": 1})
    assert not improves({"video": 0.999, "audio": 0.999}, {"video": 1, "audio": 1})
    x = torch.ones(129, 4)
    assert losses(x, x, {"all": (0, 129)}) == {"all": 0}
