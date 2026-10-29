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

"""Portable pipeline CPU contracts, not GPU/kernel or full video acceptance."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from h3_modelopt_ptq import export_frozen, factorize_legacy, h3_layer_names
from portable_calibration import ChannelMax, load_calibration, merge_ranks, save_rank
from portable_export import modelopt_roundtrip
from portable_io import local_file, read_json, validate_order
from portable_sglang import Controller
from torch import nn

SPEC = importlib.util.spec_from_file_location(
    "h3_requests", Path(__file__).resolve().parents[1] / "scripts/h3_requests.py"
)
assert SPEC is not None and SPEC.loader is not None
REQUESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REQUESTS)


def test_real_streamed_modelopt_roundtrip():
    factors = factorize_legacy(torch.randn(48, 64).bfloat16(), torch.ones(64), seed=42, rank=16)
    restored = export_frozen(modelopt_roundtrip(factors))
    for name in ("pre_quant_scale", "residual", "lora_a", "lora_b", "activation_absmax"):
        assert torch.equal(getattr(restored, name), getattr(factors, name))


def test_channel_max_chunked_matches_dense():
    collector = ChannelMax(16, "cpu")
    x = torch.randn(3, 1030, 16).bfloat16()
    collector.update(x)
    collector.update(torch.zeros(2, 16))
    assert torch.equal(collector.maximum, x.float().reshape(-1, 16).abs().amax(0))
    assert collector.rows == 3092 and collector.calls == 2
    with pytest.raises(ValueError, match="width"):
        collector.update(torch.ones(1, 17))


def make_rank(directory, rank, value):
    collectors = {}
    for name, (_, width) in validate_order(h3_layer_names()).items():
        item = ChannelMax(width, "cpu")
        item.update(torch.full((1, width), value))
        collectors[name] = item
    save_rank(
        directory,
        collectors,
        rank=rank,
        world_size=2,
        forwards=1,
        source={"test_only": "synthetic-statistics"},
        request_sha256="test-only",
        media_sha256=["synthetic-image"],
    )


def test_full208_rank_merge_and_tamper(tmp_path):
    make_rank(tmp_path, 0, 1.0)
    with pytest.raises(ValueError, match="Missing"):
        merge_ranks(tmp_path, tmp_path / "missing", world_size=2, expected_forwards=1)
    make_rank(tmp_path, 1, 2.0)
    with pytest.raises(ValueError, match="Rank/count"):
        merge_ranks(tmp_path, tmp_path / "wrong-count", world_size=2, expected_forwards=49)
    output = tmp_path / "merged"
    merge_ranks(tmp_path, output, world_size=2, expected_forwards=1)
    receipt, stats = load_calibration(output)
    assert len(stats) == 208 and receipt["world_size"] == 2
    assert receipt["media_sha256"] == ["synthetic-image"]
    assert all(torch.all(value == 2) for value in stats.values())
    with (output / "absmax.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        load_calibration(output)


def test_calibration_refuses_incomplete_or_nonfinite(tmp_path):
    collectors = {
        name: ChannelMax(shape[1], "cpu")
        for name, shape in validate_order(h3_layer_names()).items()
    }
    with pytest.raises(ValueError, match="Incomplete"):
        save_rank(
            tmp_path,
            collectors,
            rank=0,
            world_size=1,
            forwards=1,
            source={},
            request_sha256="",
            media_sha256=[],
        )
    for item in collectors.values():
        item.update(torch.ones(1, item.maximum.numel()))
    next(iter(collectors.values())).maximum[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        save_rank(
            tmp_path,
            collectors,
            rank=0,
            world_size=1,
            forwards=1,
            source={},
            request_sha256="",
            media_sha256=[],
        )


@pytest.mark.parametrize("name", ["../escape", "/absolute", "..", "", None])
def test_payload_path_rejection(tmp_path, name):
    with pytest.raises(ValueError):
        local_file(tmp_path, name)


def manifest(tmp_path, purpose, image, prompt):
    path = tmp_path / (purpose + ".json")
    path.write_text(
        json.dumps(
            {
                "format": "h3-ptq-requests-v1",
                "purpose": purpose,
                "requests": [
                    {
                        "id": purpose,
                        "prompt": prompt,
                        "seed": 42,
                        "reference_images": [image],
                        "seconds": 5,
                        "short_edge": 768,
                        "aspect_ratio": "16:9",
                        "num_inference_steps": 50,
                        "expected_dit_calls": 49,
                    }
                ],
            }
        )
    )
    return path


def test_request_disjoint_content_and_schema(tmp_path):
    (tmp_path / "cal.png").write_bytes(b"fixture-calibration-image")
    (tmp_path / "eval.png").write_bytes(b"fixture-heldout-image")
    cal = manifest(tmp_path, "calibration", "cal.png", "A cup")
    evaluation = manifest(tmp_path, "evaluation", "eval.png", "A boat")
    REQUESTS.check_disjoint(cal, evaluation)
    (tmp_path / "eval.png").write_bytes(b"fixture-calibration-image")
    with pytest.raises(ValueError, match="reuse"):
        REQUESTS.check_disjoint(cal, evaluation)
    row = read_json(evaluation)
    row["requests"][0]["reference_images"] = ["../outside.png"]
    evaluation.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="relative"):
        REQUESTS.load_requests(evaluation)


def test_reused_prompt_rejected(tmp_path):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.png").write_bytes(b"b")
    left = manifest(tmp_path, "calibration", "a.png", "A   Cup")
    right = manifest(tmp_path, "evaluation", "b.png", " a cup ")
    with pytest.raises(ValueError, match="reuse"):
        REQUESTS.check_disjoint(left, right)


class MetaH3(nn.Module):
    """Exact selection/shape metadata only; deliberately does not run H3 GEMMs."""

    def __init__(self):
        super().__init__()
        self.token_refiner = nn.Identity()
        for name, (n, k) in validate_order(h3_layer_names()).items():
            parent = self
            parts = name.split(".")
            for component in parts[:-1]:
                if not hasattr(parent, component):
                    parent.add_module(component, nn.Module())
                parent = getattr(parent, component)
            layer = nn.Linear(k, n, bias=False, device="meta", dtype=torch.bfloat16)
            layer.quant_method = object()
            parent.add_module(parts[-1], layer)

    def forward(self, value, fail=False, recurse=False):
        if fail:
            raise ValueError("intentional fixture failure")
        if recurse:
            return self(value)
        return self.token_refiner(value)


def test_root_and_request_static_refiner_hooks(tmp_path):
    model = MetaH3()
    controller = Controller(
        model, {"mode": "bf16", "world_size": 1, "files": {}, "output": str(tmp_path)}
    )
    value = torch.ones(1)
    assert torch.equal(model.token_refiner(value), value)
    assert controller.prepared and controller.count == 0
    model(value)
    assert controller.count == 1 and not controller.lock.locked()
    with pytest.raises(ValueError, match="intentional"):
        model(value, fail=True)
    assert controller.failed and not controller.lock.locked()
    with pytest.raises(RuntimeError, match="Previous"):
        model(value)


def test_reentrant_root_does_not_release_outer_lock(tmp_path):
    model = MetaH3()
    controller = Controller(
        model, {"mode": "bf16", "world_size": 1, "files": {}, "output": str(tmp_path)}
    )
    with pytest.raises(RuntimeError, match="reentrant"):
        model(torch.ones(1), recurse=True)
    assert controller.failed and not controller.lock.locked() and controller.count == 0
