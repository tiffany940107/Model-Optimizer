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

"""Full 208-name provenance-chain tests with tiny, explicitly synthetic file fixtures."""

import json

import pytest
import torch
from artifact_binding import REPOSITORY, REVISION, digest, verify_binding
from h3_modelopt_ptq import h3_layer_names
from safetensors.torch import save_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return digest(path)


@pytest.fixture
def bound_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("artifact_binding.expected_shape", lambda _name: (32, 32))
    root, artifact, calibration = (tmp_path / name for name in ("root", "artifact", "calibration"))
    artifact.mkdir()
    calibration.mkdir()
    names = list(h3_layer_names())
    model = root / "models/MiniMax-H3/Ref2VA"
    index = {name + ".weight": "tiny-source.safetensors" for name in names}
    source_hashes = {
        "model_index.json": write_json(
            model / "model_index.json", {"_minimax_h3": {"partition": "ref2va"}}
        ),
        "transformer/config.json": write_json(
            model / "transformer/config.json", {"synthetic_fixture": True}
        ),
        "transformer/model.safetensors.index.json": write_json(
            model / "transformer/model.safetensors.index.json", {"weight_map": index}
        ),
    }
    shard = model / "transformer/tiny-source.safetensors"
    save_file({"fixture": torch.ones(1)}, str(shard))
    source_hashes["transformer/tiny-source.safetensors"] = digest(shard)
    download_sha = write_json(
        root / "results/download_complete.json",
        {
            "repository": REPOSITORY,
            "revision": REVISION,
            "status": "downloaded-size-checked",
            "files": [
                {
                    "name": "Ref2VA/" + name,
                    "bytes": (model / name).stat().st_size,
                    "local_sha256": sha,
                }
                for name, sha in source_hashes.items()
            ],
        },
    )
    source_receipt = root / "source-receipt.json"
    preflight_sha = write_json(
        source_receipt,
        {
            "repository": REPOSITORY,
            "revision": REVISION,
            "status": "source-content-verified",
            "source_file_sha256": source_hashes,
            "download_receipt_sha256": download_sha,
        },
    )
    calibration_manifest_sha = write_json(
        root / "calibration_manifest.json", {"independent_fixture": True}
    )
    contract_sha = write_json(calibration / "contract.json", {"fixture": True})
    complete_sha = write_json(calibration / "complete.json", {"fixture": True})
    ranks = []
    for rank in range(2):
        stats_path = calibration / f"calibration.rank{rank}.pt"
        stats = {name: torch.ones(32) * (rank + 1) for name in names}
        torch.save({"activation_absmax": stats, "num_forwards": 49}, stats_path)
        samples = stats_path.with_suffix("")
        samples.mkdir()
        entries = {}
        for name in names:
            sample = samples / (name.replace(".", "__") + ".pt")
            sample.write_bytes(b"synthetic hash-only sample fixture")
            entries[name] = {"file": sample.name, "sha256": digest(sample)}
        sample_manifest_sha = write_json(samples / "manifest.json", entries)
        ranks.append(
            {
                "rank": rank,
                "forwards": 49,
                "stats_sha256": digest(stats_path),
                "samples_manifest_sha256": sample_manifest_sha,
                "samples": entries,
            }
        )
    calibration_sha = write_json(
        calibration / "calibration_validated.json",
        {
            "status": "passed",
            "source_partition": "ref2va",
            "ranks": ranks,
            "calibration_manifest_sha256": calibration_manifest_sha,
            "generation_contract_sha256": contract_sha,
            "generation_complete_sha256": complete_sha,
        },
    )
    torch.save(
        {"activation_absmax": {name: torch.ones(32) * 2 for name in names}, "num_forwards": 98},
        artifact / "merged-calibration.pt",
    )
    layers = {}
    for name in names:
        path = artifact / (name.replace(".", "__") + ".safetensors")
        save_file({"fixture": torch.ones(1)}, str(path))
        layers[name] = {
            "file": path.name,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "shape": [32, 32],
            "source_key": name + ".weight",
            "source_shard": index[name + ".weight"],
            "factorizer": "global",
            "weight_quantizer": "rtn",
            "gptq": None,
            "block_aware": None,
        }
    manifest_sha = write_json(
        artifact / "manifest.json",
        {
            "format": "minimax_h3_nvfp4_svdquant_v1",
            "rank": 32,
            "alpha": 0.5,
            "seed": 1101,
            "factorizer": "global",
            "weight_quantizer": "rtn",
            "weight_scale_rule": "static_6",
            "block_size": 16,
            "scale_dtype": "float8_e4m3fn",
            "residual_dtype": "float4_e2m1fn_x2",
            "layers": layers,
        },
    )
    rows = [
        {"layer": name, "rows": size, "relative_l2": 0.001, "repeat_exact": True}
        for name in names
        for size in (1, 128, 129)
    ]
    (artifact / "kernel-validation.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    codes = {}
    for name in (
        "scripts/export_ref2va.py",
        "scripts/validate_calibration.py",
        "vendor/svdquant_scripts/export_h3_svdquant_nvfp4.py",
        "vendor/svdquant_scripts/merge_h3_trajectory_calibration.py",
        "svdquant_h3/core.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic binding fixture, never executed.\n")
        codes[name] = digest(path)
    provenance = {
        "status": "validated",
        "source_partition": "ref2va",
        "repository": REPOSITORY,
        "revision": REVISION,
        "layers": 208,
        "gradient_updates": 0,
        "kernel_comparisons": 624,
        "validation_rows": [1, 128, 129],
        "kernel_relative_l2_limit": 0.02,
        "manifest_sha256": manifest_sha,
        "merged_calibration_sha256": digest(artifact / "merged-calibration.pt"),
        "kernel_validation_sha256": digest(artifact / "kernel-validation.jsonl"),
        "calibration_receipt_sha256": calibration_sha,
        "download_receipt_sha256": download_sha,
        "source_preflight_receipt_sha256": preflight_sha,
        "source_model_index_sha256": source_hashes["model_index.json"],
        "source_index_sha256": source_hashes["transformer/model.safetensors.index.json"],
        "source_file_sha256": source_hashes,
        "source_code_sha256": codes,
    }
    write_json(artifact / "ref2va_provenance.json", provenance)
    return root, artifact, calibration, source_receipt


def test_all_208_bindings_pass_and_source_mutation_is_rejected(bound_fixture):
    root, artifact, calibration, source_receipt = bound_fixture
    binding = verify_binding(root, artifact, calibration, source_receipt)
    assert len(binding["stats"]) == 208
    assert list(binding["stats"]) == list(h3_layer_names())
    shard = root / "models/MiniMax-H3/Ref2VA/transformer/tiny-source.safetensors"
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source size"):
        verify_binding(root, artifact, calibration, source_receipt)


def test_rank_statistics_mutation_is_rejected(bound_fixture):
    root, artifact, calibration, source_receipt = bound_fixture
    stats = calibration / "calibration.rank1.pt"
    stats.write_bytes(stats.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Content binding"):
        verify_binding(root, artifact, calibration, source_receipt)


def test_incomplete_kernel_gate_cannot_be_hidden_by_updating_receipt_hash(bound_fixture):
    root, artifact, calibration, source_receipt = bound_fixture
    records = artifact / "kernel-validation.jsonl"
    records.write_text(records.read_text().splitlines()[0] + "\n")
    receipt = artifact / "ref2va_provenance.json"
    provenance = json.loads(receipt.read_text())
    provenance["kernel_validation_sha256"] = digest(records)
    write_json(receipt, provenance)
    with pytest.raises(ValueError, match="coverage/results"):
        verify_binding(root, artifact, calibration, source_receipt)
