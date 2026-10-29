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

"""Tiny CPU packed-state fixtures, not valid production H3 quality evidence."""

import json

import pytest
import torch
from artifact_binding import digest, local_file, validate_payload
from packed_replay import export_packed, import_packed, make_empty_layer
from safetensors.torch import save_file
from streaming_roundtrip import run_roundtrip, validate_overlay

import modelopt.torch.opt as mto


def payload():
    torch.manual_seed(101)
    n, k = 32, 32
    stats = torch.ones(k)
    tensors = {
        "weight": torch.randint(0, 256, (n, k // 2), dtype=torch.uint8),
        "weight_scale": torch.full((512,), 56, dtype=torch.uint8),
        "pre_quant_scale": torch.ones(k, dtype=torch.bfloat16),
        "svdquant_lora_a": torch.randn(32, k).bfloat16(),
        "svdquant_lora_b": torch.randn(n, 32).bfloat16(),
        "input_scale": torch.tensor([2688.0]),
        "weight_global_scale": torch.tensor([1344.0]),
        "alpha": 1 / (torch.tensor([2688.0]) * torch.tensor([1344.0])),
    }
    return tensors, stats


def test_packed_modelopt_uses_empty_dense_weight_and_roundtrips(tmp_path):
    tensors, stats = payload()
    model = import_packed(tensors, stats)
    assert model[0].weight.numel() == 0
    with pytest.raises(RuntimeError, match="Packed-only"):
        model(torch.zeros(1, 32).bfloat16())
    checkpoint = tmp_path / "state.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(make_empty_layer(32, 32), checkpoint)
    assert restored[0].weight.numel() == 0
    assert all(torch.equal(value, export_packed(restored)[key]) for key, value in tensors.items())
    with pytest.raises(RuntimeError, match="Packed-only"):
        restored(torch.zeros(1, 32).bfloat16())


def test_streaming_real_modelopt_state_and_tamper_detection(tmp_path, monkeypatch):
    monkeypatch.setattr("artifact_binding.expected_shape", lambda _name: (32, 32))
    source = tmp_path / "legacy"
    source.mkdir()
    tensors, stats = payload()
    layers = {}
    for name in ("blocks.0.mlp.fc1", "blocks.0.mlp.fc2"):
        path = source / (name.replace(".", "__") + ".safetensors")
        save_file(tensors, str(path))
        layers[name] = {
            "file": path.name,
            "sha256": digest(path),
            "shape": [32, 32],
            "bytes": path.stat().st_size,
        }
    binding = {
        "artifact": source,
        "manifest": {"seed": 1101, "layers": layers},
        "stats": dict.fromkeys(layers, stats),
        "signatures": {},
        "binding_sha256": {"synthetic_fixture": "a" * 64},
    }
    output = tmp_path / "roundtrip"
    receipt = run_roundtrip(binding, source, output)
    assert receipt["layers"] == 2
    assert validate_overlay(output, binding)["status"] == "all-payloads-exact"
    for record in receipt["records"].values():
        assert not record["dense_residual_stored"]
    with pytest.raises(FileExistsError):
        run_roundtrip(binding, source, output)
    checkpoint = (
        output / "modelopt_states" / next(iter(receipt["records"].values()))["modelopt_state_file"]
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="state changed"):
        validate_overlay(output, binding)


def test_calibration_scale_gate_and_unsafe_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("artifact_binding.expected_shape", lambda _name: (32, 32))
    tensors, stats = payload()
    validate_payload("fixture", tensors, stats)
    with pytest.raises(ValueError, match="calibration"):
        validate_payload("fixture", tensors, stats * 2)
    with pytest.raises(ValueError, match="Unsafe"):
        local_file(tmp_path, "../external.pt")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({}))
    (tmp_path / "link.json").symlink_to(path)
    with pytest.raises(ValueError, match="unsafe"):
        local_file(tmp_path, "link.json")
