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

"""CPU receipt tests on explicit tiny 208-layer fixtures, not GPU evidence."""

import json
import os
from pathlib import Path

import binding_cache as cache
import pytest
import torch
from artifact_binding import verify_binding
from test_artifact_binding import bound_fixture as bound_fixture  # Synthetic pytest fixture.


@pytest.fixture
def prepared(bound_fixture, tmp_path):
    root, artifact, calibration, source_receipt = bound_fixture
    kwargs = {
        "root": root,
        "artifact": artifact,
        "calibration": calibration,
        "source_receipt": source_receipt,
    }
    binding = verify_binding(root, artifact, calibration, source_receipt)
    receipt = tmp_path / "cpu-binding.json"
    cache.write_binding_cache(binding, receipt, **kwargs)
    return kwargs, binding, receipt


def test_cached_binding_matches_full_metadata_and_stats_without_tensor_rehash(
    prepared, monkeypatch
):
    kwargs, golden, receipt = prepared
    digest = cache.digest
    hashed = []

    def tracked(path):
        hashed.append(str(path))
        assert Path(path).suffix != ".safetensors", "GPU launch must not rehash tensor shards"
        return digest(path)

    monkeypatch.setattr(cache, "digest", tracked)
    actual = cache.load_binding_cache(receipt, **kwargs)
    assert actual["manifest"] == golden["manifest"]
    assert actual["provenance"] == golden["provenance"]
    assert actual["binding_sha256"] == golden["binding_sha256"]
    assert list(actual["stats"]) == list(golden["stats"])
    assert all(torch.equal(actual["stats"][key], value) for key, value in golden["stats"].items())
    assert str(kwargs["artifact"] / "merged-calibration.pt") in hashed
    assert (
        actual["binding_verification"]["mode"] == "cpu-full-hash-then-gpu-stat-and-small-hash-reuse"
    )


@pytest.mark.parametrize("key", ["root", "artifact", "calibration", "source_receipt"])
def test_all_four_launch_paths_are_bound(prepared, key):
    kwargs, _, receipt = prepared
    kwargs = {**kwargs, key: Path(str(kwargs[key]) + "-different")}
    with pytest.raises(ValueError, match="paths"):
        cache.load_binding_cache(receipt, **kwargs)


@pytest.mark.parametrize(
    "mutation", ["timestamp", "metadata", "coverage", "cached-manifest", "code"]
)
def test_cache_rejects_mutation_or_incomplete_metadata(prepared, monkeypatch, mutation):
    kwargs, _, path = prepared
    receipt = json.loads(path.read_text())
    if mutation == "timestamp":
        source = next(Path(name) for name in receipt["signatures"] if name.endswith(".safetensors"))
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    elif mutation == "metadata":
        (kwargs["artifact"] / "manifest.json").write_text("{}")
    elif mutation == "coverage":
        del receipt["small_file_sha256"][str(kwargs["source_receipt"])]
    elif mutation == "cached-manifest":
        receipt["metadata"]["manifest"]["seed"] = 9000
    else:
        monkeypatch.setattr(cache, "implementation_hashes", lambda: {"changed": "code"})
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        cache.load_binding_cache(path, **kwargs)


def test_cache_never_overwrites_existing_receipt(prepared):
    kwargs, binding, receipt = prepared
    with pytest.raises(FileExistsError):
        cache.write_binding_cache(binding, receipt, **kwargs)
