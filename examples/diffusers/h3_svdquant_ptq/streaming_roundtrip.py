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

"""One-layer-at-a-time ModelOpt checkpoint/export of bound legacy artifacts."""

import copy
import gc
import json
from pathlib import Path

import torch
from artifact_binding import digest, file_signature, local_file, validate_payload
from packed_replay import export_packed, import_packed, make_empty_layer
from safetensors.torch import load_file, save_file

import modelopt.torch.opt as mto

__all__ = ["check_unchanged", "run_roundtrip", "validate_overlay"]


def check_unchanged(signatures):
    """Fail if any previously content-hashed source changed during this run."""
    for path, signature in signatures.items():
        if file_signature(path) != signature:
            raise ValueError(f"Bound source changed during operation: {path}")


def _write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run_roundtrip(binding, artifact, output):
    """Stream every validated layer through a real ModelOpt save/restore cycle.

    Stores packed-only ModelOpt checkpoints: no full model, original weight or
    fake BF16 residual is materialized. The peak is several packed-layer copies.
    A complete marker is published only after every layer and binding passes.
    """
    artifact, output = Path(artifact).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_dir = output / "modelopt_states"
    state_dir.mkdir()
    manifest = copy.deepcopy(binding["manifest"])
    records = {}
    for ordinal, (name, entry) in enumerate(manifest["layers"].items()):
        source = local_file(artifact, entry["file"])
        if digest(source) != entry["sha256"]:
            raise ValueError("Layer changed after initial binding")
        tensors = load_file(str(source), device="cpu")
        validate_payload(name, tensors, binding["stats"][name])
        model = import_packed(tensors, binding["stats"][name])
        if model[0].weight.numel() != 0:
            raise RuntimeError("Packed-only state unexpectedly allocated dense weight")
        checkpoint = state_dir / (Path(entry["file"]).stem + ".pt")
        mto.save(model, checkpoint)
        del model
        n, k = entry["shape"]
        restored = mto.restore(make_empty_layer(n, k), checkpoint)
        exported = export_packed(restored)
        if set(exported) != set(tensors) or any(
            exported[key].dtype != value.dtype or not torch.equal(exported[key], value)
            for key, value in tensors.items()
        ):
            raise ValueError(f"ModelOpt checkpoint/export changed tensor bytes: {name}")
        if not torch.equal(
            restored[0].input_quantizer._h3_activation_absmax, binding["stats"][name]
        ):
            raise ValueError("ModelOpt checkpoint changed calibration")
        target = output / entry["file"]
        save_file(exported, str(target))
        reread = load_file(str(target), device="cpu")
        if any(not torch.equal(reread[key], value) for key, value in tensors.items()):
            raise ValueError("Exported safetensors changed payload")
        records[name] = {
            "ordinal": ordinal,
            "layer_seed": manifest["seed"] + ordinal,
            "legacy_file_sha256": entry["sha256"],
            "export_file_sha256": digest(target),
            "modelopt_state_file": checkpoint.name,
            "modelopt_state_sha256": digest(checkpoint),
            "tensor_payload_exact": True,
            "dense_residual_stored": False,
        }
        entry["sha256"], entry["bytes"] = records[name]["export_file_sha256"], target.stat().st_size
        print(
            json.dumps(
                {
                    "phase": "modelopt-packed-roundtrip",
                    "layer": name,
                    "ordinal": ordinal,
                    "total": len(manifest["layers"]),
                }
            ),
            flush=True,
        )
        del tensors, exported, restored, reread
        gc.collect()
    check_unchanged(binding["signatures"])
    _write_json(output / "manifest.json", manifest)
    receipt = {
        "schema": "h3-modelopt-packed-roundtrip-v1",
        "status": "all-payloads-exact",
        "source_partition": "ref2va",
        "layers": len(records),
        "source_binding_sha256": binding["binding_sha256"],
        "manifest_sha256": digest(output / "manifest.json"),
        "records": records,
        "torch": torch.__version__,
        "modelopt_mode": "h3_svdquant_packed_replay_v1",
        "scope": (
            "Original packed tensor payloads preserved through real ModelOpt state; "
            "no new calibration or native quantization/GPU/quality validation"
        ),
        "source_verification": (
            "full source/calibration SHA256 at start; "
            "size/mtime/ctime/inode/resolved path checked again at end"
        ),
    }
    _write_json(output / "modelopt_compat_provenance.json", receipt)
    return receipt


def validate_overlay(output, binding):
    """Revalidate a completed bridge against the caller's freshly bound golden artifact."""
    output = Path(output).resolve()
    receipt = json.loads(local_file(output, "modelopt_compat_provenance.json").read_text())
    if (receipt.get("schema"), receipt.get("status"), receipt.get("modelopt_mode")) != (
        "h3-modelopt-packed-roundtrip-v1",
        "all-payloads-exact",
        "h3_svdquant_packed_replay_v1",
    ) or receipt.get("source_binding_sha256") != binding["binding_sha256"]:
        raise ValueError("Overlay is not bound to this validated legacy artifact")
    manifest_path = local_file(output, "manifest.json")
    if digest(manifest_path) != receipt.get("manifest_sha256"):
        raise ValueError("Overlay manifest changed")
    manifest = json.loads(manifest_path.read_text())
    original = binding["manifest"]
    if {key: value for key, value in manifest.items() if key != "layers"} != {
        key: value for key, value in original.items() if key != "layers"
    }:
        raise ValueError("Overlay recipe metadata changed")
    if list(manifest["layers"]) != list(original["layers"]) or set(receipt["records"]) != set(
        original["layers"]
    ):
        raise ValueError("Overlay layer coverage/order changed")
    if receipt.get("layers") != len(original["layers"]):
        raise ValueError("Overlay layer count mismatch")
    for ordinal, (name, entry) in enumerate(manifest["layers"].items()):
        record = receipt["records"][name]
        if {key: value for key, value in entry.items() if key not in {"sha256", "bytes"}} != {
            key: value
            for key, value in original["layers"][name].items()
            if key not in {"sha256", "bytes"}
        }:
            raise ValueError("Overlay source/shape metadata changed")
        if (
            record["ordinal"] != ordinal
            or record["layer_seed"] != original["seed"] + ordinal
            or not record["tensor_payload_exact"]
            or record["dense_residual_stored"]
            or record["legacy_file_sha256"] != original["layers"][name]["sha256"]
        ):
            raise ValueError("Overlay replay metadata changed")
        layer = local_file(output, entry["file"])
        state = local_file(output / "modelopt_states", record["modelopt_state_file"])
        if (
            digest(layer) != record["export_file_sha256"]
            or entry["sha256"] != record["export_file_sha256"]
            or layer.stat().st_size != entry["bytes"]
        ):
            raise ValueError("Overlay layer file changed")
        if digest(state) != record["modelopt_state_sha256"]:
            raise ValueError("ModelOpt state changed")
        gold_file = local_file(binding["artifact"], original["layers"][name]["file"])
        if digest(gold_file) != original["layers"][name]["sha256"]:
            raise ValueError("Original golden layer changed")
        candidate, golden = load_file(str(layer)), load_file(str(gold_file))
        if set(candidate) != set(golden) or any(
            candidate[key].dtype != value.dtype or not torch.equal(candidate[key], value)
            for key, value in golden.items()
        ):
            raise ValueError("Overlay no longer has original golden tensor payloads")
        del candidate, golden
    return receipt
