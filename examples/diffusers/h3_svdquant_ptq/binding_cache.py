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

"""Trusted local CPU full-hash receipt, then GPU stat/metadata-bound reuse.

Not a signature or protection against forged local receipts. Heavy source and
packed tensors are NOT content-rehashed on GPU. Original verify_binding remains
the authority and default path; this cache never substitutes mocked validation.
"""

import json
from pathlib import Path

from artifact_binding import digest, file_signature


def path_binding(root, artifact, calibration, source_receipt):
    return dict(
        zip(
            ("root", "artifact", "calibration", "source_receipt"),
            (str(Path(path).resolve()) for path in (root, artifact, calibration, source_receipt)),
            strict=True,
        )
    )


def implementation_hashes():
    directory = Path(__file__).resolve().parent
    return {
        name: digest(directory / name)
        for name in (
            "binding_cache.py",
            "artifact_binding.py",
            "h3_modelopt_ptq.py",
            "scripts/preflight_regeneration.py",
            "scripts/compare_regenerated.py",
        )
    }


def check_signatures(signatures):
    if not signatures:
        raise ValueError("Missing CPU source signatures")
    for path, expected in signatures.items():
        if file_signature(path) != expected:
            raise ValueError(f"CPU-preflight input changed: {path}")


def write_binding_cache(binding, output, *, root, artifact, calibration, source_receipt):
    """Only call with the result of a successful real verify_binding invocation."""
    paths = path_binding(root, artifact, calibration, source_receipt)
    check_signatures(binding["signatures"])
    metadata = {
        key: binding[key] for key in ("manifest", "provenance", "source_index", "binding_sha256")
    }
    metadata.update(model=str(binding["model"]), artifact=str(binding["artifact"]))
    small = {
        path: digest(path)
        for path in binding["signatures"]
        if Path(path).suffix in (".json", ".jsonl", ".py", ".txt")
    }
    merged = Path(paths["artifact"]) / "merged-calibration.pt"
    # Small (~208 vectors) and authoritative at launch, unlike multi-GB weights.
    small[str(merged)] = digest(merged)
    if small[str(merged)] != binding["provenance"]["merged_calibration_sha256"]:
        raise ValueError("Merged calibration changed after full verification")
    check_signatures(binding["signatures"])
    receipt = {
        "schema": "h3-modelopt-binding-cache-v1",
        "status": "cpu-full-binding-verified",
        "paths": paths,
        "implementation_sha256": implementation_hashes(),
        "metadata": metadata,
        "signatures": binding["signatures"],
        "small_file_sha256": small,
        "scope": (
            "CPU full source/calibration/artifact SHA validation; GPU reuse only "
            "unchanged signatures and small hashes, not all tensor bytes again"
        ),
    }
    with Path(output).open("x") as stream:
        json.dump(receipt, stream, indent=2)
    return {
        "path": str(Path(output).resolve()),
        "sha256": digest(output),
        "scope": receipt["scope"],
    }


def load_binding_cache(receipt_path, *, root, artifact, calibration, source_receipt):
    """Rehydrate verified small metadata/stats without rehashing model shards."""
    import torch

    receipt = json.loads(Path(receipt_path).read_text())
    paths = path_binding(root, artifact, calibration, source_receipt)
    if (
        receipt.get("schema") != "h3-modelopt-binding-cache-v1"
        or receipt.get("status") != "cpu-full-binding-verified"
        or receipt.get("paths") != paths
        or receipt.get("implementation_sha256") != implementation_hashes()
    ):
        raise ValueError("CPU binding cache schema/paths/validator implementation mismatch")
    signatures = receipt["signatures"]
    check_signatures(signatures)
    metadata = receipt["metadata"]
    model = Path(paths["root"]) / "models/MiniMax-H3/Ref2VA"
    artifact = Path(paths["artifact"])
    if (
        Path(metadata["model"]).resolve() != model.resolve()
        or Path(metadata["artifact"]).resolve() != artifact
    ):
        raise ValueError("CPU binding cache model/artifact path mismatch")
    json_paths = {
        "manifest": artifact / "manifest.json",
        "provenance": artifact / "ref2va_provenance.json",
        "source_index": model / "transformer/model.safetensors.index.json",
    }
    merged = artifact / "merged-calibration.pt"
    required_small = {str(path) for path in json_paths.values()} | {
        str(merged),
        paths["source_receipt"],
        str(Path(paths["calibration"]) / "calibration_validated.json"),
        str(Path(paths["root"]) / "results/download_complete.json"),
        str(artifact / "kernel-validation.jsonl"),
    }
    expected_small = {
        path for path in signatures if Path(path).suffix in (".json", ".jsonl", ".py", ".txt")
    } | {str(merged)}
    small = receipt.get("small_file_sha256", {})
    if (
        set(small) != expected_small
        or not required_small <= set(small)
        or not set(small) <= set(signatures)
    ):
        raise ValueError("CPU cache lacks complete small-receipt/source hash coverage")
    for path, expected in small.items():
        if digest(path) != expected:
            raise ValueError(f"CPU binding metadata changed: {path}")
    for key, path in json_paths.items():
        actual = json.loads(path.read_text())
        if key == "source_index":
            actual = actual["weight_map"]
        if actual != metadata[key]:
            raise ValueError(f"Cached metadata differs from original receipt: {key}")
    heavy_required = {str(model / name) for name in metadata["provenance"]["source_file_sha256"]}
    heavy_required |= {
        str(artifact / entry["file"]) for entry in metadata["manifest"]["layers"].values()
    }
    if not heavy_required <= set(signatures):
        raise ValueError("Missing source/packed-tensor signature coverage")
    binding_hashes = {
        "legacy_provenance": small[str(json_paths["provenance"])],
        "legacy_manifest": small[str(json_paths["manifest"])],
        "merged_calibration": small[str(merged)],
        "calibration_receipt": small[
            str(Path(paths["calibration"]) / "calibration_validated.json")
        ],
        "source_receipt": small[paths["source_receipt"]],
    }
    if metadata["binding_sha256"] != binding_hashes:
        raise ValueError("Cached source binding differs from original small receipts")
    if small[str(merged)] != metadata["provenance"]["merged_calibration_sha256"]:
        raise ValueError("Cached calibration digest differs from original provenance")
    stats = torch.load(merged, map_location="cpu", weights_only=True)["activation_absmax"]
    if list(stats) != list(metadata["manifest"]["layers"]) or len(stats) != 208:
        raise ValueError("Cached calibration layer coverage/order changed")
    check_signatures(signatures)
    return {
        **metadata,
        "model": model,
        "artifact": artifact,
        "stats": stats,
        "signatures": signatures,
        "binding_verification": {
            "mode": "cpu-full-hash-then-gpu-stat-and-small-hash-reuse",
            "receipt_sha256": digest(receipt_path),
            "scope": receipt["scope"],
        },
    }
