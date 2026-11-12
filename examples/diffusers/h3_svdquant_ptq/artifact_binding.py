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

"""Fail-closed bindings for the reviewed original Ref2VA static-6 PTQ artifact."""

import hashlib
import json
from pathlib import Path

import torch
from h3_modelopt_ptq import h3_layer_names

__all__ = [
    "digest",
    "expected_shape",
    "file_signature",
    "local_file",
    "validate_payload",
    "verify_binding",
]

REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
REPOSITORY = "MiniMaxAI/MiniMax-H3"


def digest(path):
    """Compute streaming SHA256 without reading a whole checkpoint into RAM."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def local_file(directory, name):
    """Allow one existing regular basename, rejecting traversal and symlinks."""
    directory = Path(directory).resolve()
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError("Unsafe artifact basename")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
        raise ValueError(f"Missing/unsafe artifact: {name}")
    return path


def file_signature(path):
    """Record mutation-detection metadata, not a replacement for content hashing."""
    stat = Path(path).stat()
    return [
        str(Path(path).resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_ino,
    ]


def expected_shape(name):
    """Return the original released H3 projection shape, including doubled FC1."""
    if name not in h3_layer_names():
        raise ValueError("Unexpected H3 projection")
    return next(
        shape
        for suffix, shape in {
            "attn.qkv_proj": (21504, 5376),
            "attn.out_proj": (5376, 7168),
            "mlp.fc1": (28672, 5376),
            "mlp.fc2": (5376, 14336),
        }.items()
        if name.endswith(suffix)
    )


def _json(path):
    return json.loads(Path(path).read_text())


def _match(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError(f"Content binding mismatch: {path}")


def validate_payload(name, tensors, stats):
    """Check TRT layouts, numerical scales, and frozen calibration association."""
    n, k = expected_shape(name)
    shapes = {
        "weight": (n, k // 2),
        "pre_quant_scale": (k,),
        "svdquant_lora_a": (32, k),
        "svdquant_lora_b": (n, 32),
        "input_scale": (1,),
        "weight_global_scale": (1,),
        "alpha": (1,),
    }
    if set(tensors) != set(shapes) | {"weight_scale"}:
        raise ValueError("Unexpected packed tensor set")
    for key, value in tensors.items():
        dtype = (
            torch.uint8
            if key in {"weight", "weight_scale"}
            else torch.bfloat16
            if key in {"pre_quant_scale", "svdquant_lora_a", "svdquant_lora_b"}
            else torch.float32
        )
        if value.dtype != dtype or (key in shapes and tuple(value.shape) != shapes[key]):
            raise ValueError(f"Bad packed shape/dtype: {name}/{key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Nonfinite packed metadata")
    scales = tensors["weight_scale"]
    if scales.numel() != ((n + 127) // 128) * ((k + 63) // 64) * 512:
        raise ValueError("Invalid TRT swizzled scale storage")
    fp8 = scales.view(torch.float8_e4m3fn).float()
    if not torch.isfinite(fp8).all() or (fp8 < 0).any():
        raise ValueError("Invalid E4M3 scales")
    if (
        stats.shape != (k,)
        or stats.dtype != torch.float32
        or not torch.isfinite(stats).all()
        or (stats < 0).any()
    ):
        raise ValueError("Bad calibration maxima")
    for key in ("pre_quant_scale", "input_scale", "weight_global_scale", "alpha"):
        if not (tensors[key] > 0).all():
            raise ValueError("Nonpositive quantization multiplier")
    expected = (
        2688 / (stats * tensors["pre_quant_scale"].float()).amax().clamp_min(1e-12)
    ).reshape(1)
    if not torch.equal(expected, tensors["input_scale"]):
        raise ValueError("Activation scale does not match frozen calibration")
    if not torch.equal(
        tensors["alpha"], 1 / (tensors["input_scale"] * tensors["weight_global_scale"])
    ):
        raise ValueError("Alpha does not match packed multipliers")


def verify_binding(root, artifact, calibration, source_receipt):
    """Fully hash source/calibration bindings before a streaming roundtrip.

    Existing trusted local receipts are the provenance root, not externally
    signed attestations. SHA checks detect changed files, not forged receipts.
    Returns only small metadata and ~208 channel-max vectors, never full weights.
    """
    root, artifact, calibration = (Path(p).resolve() for p in (root, artifact, calibration))
    source_receipt = Path(source_receipt).resolve()
    provenance_path = local_file(artifact, "ref2va_provenance.json")
    provenance = _json(provenance_path)
    if (
        provenance.get("status"),
        provenance.get("source_partition"),
        provenance.get("repository"),
        provenance.get("revision"),
    ) != ("validated", "ref2va", REPOSITORY, REVISION):
        raise ValueError("Expected a validated original Ref2VA artifact")
    if provenance.get("layers") != 208 or provenance.get("gradient_updates") != 0:
        raise ValueError("Expected original 208-layer PTQ artifact")
    if (
        provenance.get("kernel_comparisons") != 624
        or provenance.get("validation_rows") != [1, 128, 129]
        or provenance.get("kernel_relative_l2_limit") != 0.02
    ):
        raise ValueError("Missing reviewed 624-comparison original kernel gate")
    checks = {
        "manifest_sha256": local_file(artifact, "manifest.json"),
        "merged_calibration_sha256": local_file(artifact, "merged-calibration.pt"),
        "kernel_validation_sha256": local_file(artifact, "kernel-validation.jsonl"),
        "calibration_receipt_sha256": local_file(calibration, "calibration_validated.json"),
        "download_receipt_sha256": root / "results/download_complete.json",
        "source_preflight_receipt_sha256": source_receipt,
    }
    for key, path in checks.items():
        _match(path, provenance.get(key))
    manifest, download, preflight = (
        _json(checks[key])
        for key in ("manifest_sha256", "download_receipt_sha256", "source_preflight_receipt_sha256")
    )
    if (download.get("repository"), download.get("revision"), download.get("status")) != (
        REPOSITORY,
        REVISION,
        "downloaded-size-checked",
    ):
        raise ValueError("Unreviewed source download receipt")
    if (preflight.get("repository"), preflight.get("revision"), preflight.get("status")) != (
        REPOSITORY,
        REVISION,
        "source-content-verified",
    ):
        raise ValueError("Unreviewed source preflight receipt")
    _match(checks["download_receipt_sha256"], preflight.get("download_receipt_sha256"))
    required = {
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
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported legacy recipe: v1 regeneration only global SVD/static-6 RTN")
    names = h3_layer_names()
    if set(manifest["layers"]) != set(names):
        raise ValueError("Legacy manifest must cover all 208 layers")
    kernel_rows = [
        json.loads(line) for line in checks["kernel_validation_sha256"].read_text().splitlines()
    ]
    if (
        len(kernel_rows) != 624
        or {(row["layer"], row["rows"]) for row in kernel_rows}
        != {(name, size) for name in names for size in (1, 128, 129)}
        or any(
            not 0 <= row["relative_l2"] < 0.02 or row["repeat_exact"] is not True
            for row in kernel_rows
        )
    ):
        raise ValueError("Original kernel validation coverage/results mismatch")
    model = root / "models/MiniMax-H3/Ref2VA"
    if _json(model / "model_index.json")["_minimax_h3"]["partition"] != "ref2va":
        raise ValueError("Wrong source model partition")
    index_path = model / "transformer/model.safetensors.index.json"
    source_index = _json(index_path)["weight_map"]
    _match(index_path, provenance.get("source_index_sha256"))
    _match(model / "model_index.json", provenance.get("source_model_index_sha256"))
    selected = {
        "model_index.json",
        "transformer/config.json",
        "transformer/model.safetensors.index.json",
    }
    selected |= {"transformer/" + source_index[name + ".weight"] for name in names}
    if (
        set(provenance["source_file_sha256"]) != selected
        or preflight["source_file_sha256"] != provenance["source_file_sha256"]
    ):
        raise ValueError("Incomplete/mismatched source hash coverage")
    pinned = {entry["name"]: entry for entry in download["files"]}
    signatures = {}
    for relative in sorted(selected):
        if relative != "model_index.json" and Path(relative).parent != Path("transformer"):
            raise ValueError("Unsafe source shard path")
        path = model / relative
        expected = pinned["Ref2VA/" + relative]
        sha = expected.get("hub_lfs_sha256", expected.get("local_sha256"))
        if (
            path.stat().st_size != expected["bytes"]
            or sha != provenance["source_file_sha256"][relative]
        ):
            raise ValueError("Pinned source size/digest mismatch")
        _match(path, sha)
        signatures[str(path)] = file_signature(path)
        print(json.dumps({"phase": "source-content-hash", "file": relative}), flush=True)
    for name, entry in manifest["layers"].items():
        if (
            entry.get("shape") != list(expected_shape(name))
            or entry.get("source_key") != name + ".weight"
            or entry.get("source_shard") != source_index[name + ".weight"]
            or entry.get("factorizer") != "global"
            or entry.get("weight_quantizer") != "rtn"
            or entry.get("gptq") is not None
            or entry.get("block_aware") is not None
            or entry.get("file") != name.replace(".", "__") + ".safetensors"
        ):
            raise ValueError("Layer source/recipe mismatch")
        path = local_file(artifact, entry["file"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("Layer size mismatch")
        _match(path, entry["sha256"])
        signatures[str(path)] = file_signature(path)
    receipt = _json(checks["calibration_receipt_sha256"])
    if (
        receipt.get("status") != "passed"
        or receipt.get("source_partition") != "ref2va"
        or [r["rank"] for r in receipt["ranks"]] != [0, 1]
    ):
        raise ValueError("Expected passed SP2 independent calibration")
    for path, key in (
        (root / "calibration_manifest.json", "calibration_manifest_sha256"),
        (calibration / "contract.json", "generation_contract_sha256"),
        (calibration / "complete.json", "generation_complete_sha256"),
    ):
        _match(path, receipt.get(key))
        signatures[str(path)] = file_signature(path)
    rank_stats = []
    for row in receipt["ranks"]:
        path = local_file(calibration, f"calibration.rank{row['rank']}.pt")
        _match(path, row["stats_sha256"])
        signatures[str(path)] = file_signature(path)
        rank_stats.append(torch.load(path, map_location="cpu", weights_only=True))
        sample_dir = path.with_suffix("")
        sample_manifest = local_file(sample_dir, "manifest.json")
        _match(sample_manifest, row["samples_manifest_sha256"])
        signatures[str(sample_manifest)] = file_signature(sample_manifest)
        if set(row["samples"]) != set(names):
            raise ValueError("Missing calibration sample coverage")
        for entry in row["samples"].values():
            sample = local_file(sample_dir, entry["file"])
            _match(sample, entry["sha256"])
            signatures[str(sample)] = file_signature(sample)
    merged = torch.load(checks["merged_calibration_sha256"], map_location="cpu", weights_only=True)
    stats = merged["activation_absmax"]
    if (
        set(stats) != set(names)
        or list(stats) != list(rank_stats[0]["activation_absmax"])
        or list(manifest["layers"]) != list(stats)
    ):
        raise ValueError("Merged statistics/manifest/rank0 layer order mismatch")
    if merged["num_forwards"] != sum(row["forwards"] for row in receipt["ranks"]):
        raise ValueError("Rank-forward accounting mismatch")
    for name, values in stats.items():
        if not torch.equal(
            values, torch.maximum(*(rank["activation_absmax"][name] for rank in rank_stats))
        ):
            raise ValueError("Merged calibration differs from per-rank maxima")
    # Hold source/receipt/library versions fixed for executable regeneration.
    if not {
        "scripts/export_ref2va.py",
        "scripts/validate_calibration.py",
        "vendor/svdquant_scripts/export_h3_svdquant_nvfp4.py",
        "vendor/svdquant_scripts/merge_h3_trajectory_calibration.py",
        "svdquant_h3/core.py",
    } <= set(provenance["source_code_sha256"]):
        raise ValueError("Missing bound legacy algorithm/source-validator code")
    for relative, sha in provenance["source_code_sha256"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe bound source-code path")
        _match(path, sha)
        signatures[str(path)] = file_signature(path)
    for path in [provenance_path, *checks.values()]:
        signatures[str(path)] = file_signature(path)
    return {
        "manifest": manifest,
        "provenance": provenance,
        "stats": stats,
        "source_index": source_index,
        "model": model,
        "artifact": artifact,
        "signatures": signatures,
        "binding_sha256": {
            "legacy_provenance": digest(provenance_path),
            "legacy_manifest": digest(checks["manifest_sha256"]),
            "merged_calibration": digest(checks["merged_calibration_sha256"]),
            "calibration_receipt": digest(checks["calibration_receipt_sha256"]),
            "source_receipt": digest(source_receipt),
        },
    }
