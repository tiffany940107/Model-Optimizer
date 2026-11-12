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

"""Portable PTQ contracts: no customer-project receipts or implicit success."""

import hashlib
import json
from pathlib import Path

from artifact_binding import expected_shape
from h3_modelopt_ptq import h3_layer_names

__all__ = ["digest", "local_file", "read_json", "source_identity", "validate_order", "write_json"]


def digest(path):
    """Hash an actual file without loading model shards into host memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    """Reject ambiguous duplicate keys and nonfinite JSON numbers."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)


def write_json(path, value):
    """Exclusive-create metadata; incomplete directories are never reused."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def local_file(root, name):
    """Resolve a single regular file, refusing traversal and symlink payloads."""
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Expected one safe basename")
    path = Path(root) / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlink file: {path}")
    return path


def validate_order(names):
    """Require exact H3 scope while preserving observed layer order for RNG."""
    if len(names) != 208 or set(names) != set(h3_layer_names()):
        raise ValueError("Require all 208 unique original H3 projections")
    return {name: expected_shape(name) for name in names}


def source_identity(model):
    """Bind Ref2VA model index plus complete selected transformer shards.

    This is a local content identity, not an assertion of HF authenticity or a
    hash of every text/VAE/audio component. Inference still needs all components.
    """
    model = Path(model).resolve(strict=True)
    model_index = local_file(model, "model_index.json")
    if read_json(model_index).get("_minimax_h3", {}).get("partition") != "ref2va":
        raise ValueError("This recipe requires the original Ref2VA model partition")
    transformer = model / "transformer"
    index_path = local_file(transformer, "model.safetensors.index.json")
    weights = read_json(index_path)["weight_map"]
    names = h3_layer_names()
    selected = {weights[name + ".weight"] for name in names}
    files = {"model_index.json": digest(model_index)}
    for name in sorted(selected | {"config.json", "model.safetensors.index.json"}):
        files["transformer/" + name] = digest(local_file(transformer, name))
    return {"format": "h3-ref2va-local-source-v1", "files": files}
