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

"""Streaming absmax calibration and rank merge for the RTN-only recipe."""

from pathlib import Path

import torch
from portable_io import digest, local_file, read_json, validate_order, write_json
from safetensors.torch import load_file, save_file

__all__ = ["ChannelMax", "load_calibration", "merge_ranks", "save_rank"]


class ChannelMax:
    """Observe unmodified linear inputs without retaining activation samples."""

    def __init__(self, width, device):
        self.maximum = torch.zeros(width, dtype=torch.float32, device=device)
        self.rows = 0
        self.calls = 0

    @torch.no_grad()
    def update(self, value):
        if value.shape[-1] != self.maximum.numel():
            raise ValueError("Calibration channel width mismatch")
        flat = value.detach().reshape(-1, value.shape[-1])
        if flat.shape[0]:
            # Bound temporary FP32 memory, especially for full video FC2 input.
            for chunk in flat.split(1024):
                torch.maximum(self.maximum, chunk.float().abs().amax(0), out=self.maximum)
            self.rows += flat.shape[0]
        self.calls += 1


def save_rank(
    directory, collectors, *, rank, world_size, forwards, source, request_sha256, media_sha256
):
    """Write immutable per-rank safetensors followed by its completion receipt."""
    directory = Path(directory)
    names = list(collectors)
    shapes = validate_order(names)
    if not 0 <= rank < world_size or forwards < 1:
        raise ValueError("Invalid rank or forward count")
    values = {}
    for name, collector in collectors.items():
        tensor = collector.maximum.cpu()
        if (
            tensor.shape != (shapes[name][1],)
            or collector.rows < 1
            or collector.calls < 1
            or collector.calls > forwards
            or (name.startswith("blocks.") and collector.calls != forwards)
            or not torch.isfinite(tensor).all()
            or (tensor < 0).any()
        ):
            raise ValueError(f"Incomplete/nonfinite calibration: {name}")
        values[name] = tensor
    path = directory / f"rank-{rank}.safetensors"
    if path.exists() or (directory / f"rank-{rank}.json").exists():
        raise FileExistsError("Refuse calibration overwrite")
    save_file(values, str(path))
    write_json(
        directory / f"rank-{rank}.json",
        {
            "format": "h3-ptq-calibration-rank-v1",
            "rank": rank,
            "world_size": world_size,
            "forwards": forwards,
            "layer_order": names,
            "source": source,
            "requests_sha256": request_sha256,
            "media_sha256": media_sha256,
            "file": path.name,
            "sha256": digest(path),
            "observations": {
                name: {"rows": c.rows, "calls": c.calls} for name, c in collectors.items()
            },
            "scope": "BF16 input absmax only; no GPTQ Hessian or block-aware samples",
        },
    )


def merge_ranks(directory, output, *, world_size, expected_forwards):
    """Require every rank and identical source/request/order; take channel maxima."""
    directory, output = Path(directory), Path(output)
    if world_size < 1 or expected_forwards < 1 or output.exists():
        raise ValueError("Positive counts and a fresh merge output are required")
    merged, common, receipts = {}, None, []
    for rank in range(world_size):
        receipt_path = local_file(directory, f"rank-{rank}.json")
        row = read_json(receipt_path)
        if (
            row.get("format") != "h3-ptq-calibration-rank-v1"
            or row["rank"] != rank
            or row["world_size"] != world_size
            or row["forwards"] != expected_forwards
            or row["file"] != f"rank-{rank}.safetensors"
        ):
            raise ValueError("Rank/count/format mismatch")
        signature = {
            key: row[key] for key in ("source", "requests_sha256", "media_sha256", "layer_order")
        }
        if common is not None and signature != common:
            raise ValueError("Calibration ranks differ in source, requests or order")
        common = signature
        shapes = validate_order(row["layer_order"])
        path = local_file(directory, row["file"])
        if digest(path) != row["sha256"]:
            raise ValueError("Calibration payload hash changed")
        tensors = load_file(str(path))
        if set(tensors) != set(shapes) or set(row["observations"]) != set(shapes):
            raise ValueError("Incomplete calibration layer coverage")
        for name in row["layer_order"]:
            value = tensors[name]
            seen = row["observations"][name]
            if (
                value.shape != (shapes[name][1],)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
                or (value < 0).any()
                or seen["rows"] < 1
                or not 1 <= seen["calls"] <= expected_forwards
                or (name.startswith("blocks.") and seen["calls"] != expected_forwards)
            ):
                raise ValueError("Invalid calibration tensor or observation count")
            merged[name] = value if rank == 0 else torch.maximum(merged[name], value)
        receipts.append(
            {"rank": rank, "receipt_sha256": digest(receipt_path), "payload_sha256": row["sha256"]}
        )
    if common is None:
        raise ValueError("No rank receipts")
    output.mkdir(parents=True, exist_ok=False)
    save_file(merged, str(output / "absmax.safetensors"))
    write_json(
        output / "complete.json",
        {
            "format": "h3-ptq-calibration-merged-v1",
            **common,
            "world_size": world_size,
            "forwards_per_rank": expected_forwards,
            "ranks": receipts,
            "file": "absmax.safetensors",
            "sha256": digest(output / "absmax.safetensors"),
        },
    )


def load_calibration(directory):
    """Validate a completed merge before factorization or runtime installation."""
    directory = Path(directory)
    receipt = read_json(local_file(directory, "complete.json"))
    if (
        receipt.get("format") != "h3-ptq-calibration-merged-v1"
        or receipt["file"] != "absmax.safetensors"
    ):
        raise ValueError("Expected completed portable calibration")
    shapes = validate_order(receipt["layer_order"])
    path = local_file(directory, receipt["file"])
    if digest(path) != receipt["sha256"]:
        raise ValueError("Merged statistics changed")
    values = load_file(str(path))
    if set(values) != set(shapes):
        raise ValueError("Merged calibration layer coverage mismatch")
    for name, value in values.items():
        if (
            value.shape != (shapes[name][1],)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError("Invalid merged activation range")
    return receipt, values
