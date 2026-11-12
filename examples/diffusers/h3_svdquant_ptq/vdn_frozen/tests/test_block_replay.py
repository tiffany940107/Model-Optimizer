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
from vdn_frozen.block_replay import generated_video_rows, replay, temporary_methods


def test_explicit_vae_chunk_geometry():
    def native_signature(frames, frames_per_chunk, latents_per_chunk):
        assert (frames, frames_per_chunk, latents_per_chunk) == (124, 17, 5)
        return 37

    assert generated_video_rows(native_signature) == 37296
    with pytest.raises(ValueError):
        generated_video_rows(native_signature, height=1345)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, bias=False) for _ in range(6)]
        )

    def forward(self, x):
        q, k, v, out, fc1, fc2 = self.projections
        return x + out(q(x) + k(x) + v(x)) + fc2(torch.relu(fc1(x)))


class Method:
    def __init__(self, layers):
        self.layers = layers

    def apply(self, _layer, x):
        parts = [torch.nn.functional.linear(x, layer.weight) for layer in self.layers]
        return parts[0] if len(parts) == 1 else torch.cat(parts, -1)


def fixture():
    block = Block()
    layers = block.projections
    selected = {"qkv": list(layers[:3]), "out": [layers[3]], "fc1": [layers[4]], "fc2": [layers[5]]}
    methods = {name: Method(group) for name, group in selected.items()}
    return block, selected, methods


@torch.no_grad()
def test_full_block_replay_identity_and_restoration():
    block, selected, methods = fixture()
    x = torch.randn(1, 17, 8)
    teacher = block(x)
    losses = replay(
        block,
        (x,),
        {},
        selected,
        methods,
        teacher,
        {"video": torch.arange(3, 17), "text": torch.arange(3)},
    )
    assert losses == {"video": 0, "text": 0}
    assert torch.equal(block(x), teacher)
    assert all("forward" not in layer.__dict__ for layer in block.projections)


def test_restore_on_exception_and_reject_existing_adapter():
    block, selected, methods = fixture()
    with pytest.raises(ValueError, match="injected"), temporary_methods(selected, methods):
        raise ValueError("injected")
    assert all("forward" not in layer.__dict__ for layer in block.projections)
    layer = block.projections[0]
    original = layer.forward
    layer.forward = original
    with pytest.raises(ValueError, match="existing adapter"), temporary_methods(selected, methods):
        pass
    assert layer.forward is original


def test_requires_offline_no_grad():
    block, selected, methods = fixture()
    x = torch.randn(1, 17, 8)
    with pytest.raises(RuntimeError, match="no_grad"):
        replay(block, (x,), {}, selected, methods, block(x), {"video": torch.arange(17)})
