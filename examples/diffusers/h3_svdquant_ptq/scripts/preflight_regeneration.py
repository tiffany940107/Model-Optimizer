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

"""CPU preflight of four representative projections and the full bound H3 export."""

import argparse
import json
from pathlib import Path

from artifact_binding import verify_binding
from h3_modelopt_ptq import h3_layer_names
from regeneration_planning import layer_storage, representative_schedule


def main():
    """Report shape-only estimates or verify actual source/calibration/artifact bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-project", type=Path)
    parser.add_argument("--legacy-artifact", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binding-receipt", type=Path, help="NEW trusted-local CPU binding cache for GPU reuse"
    )
    args = parser.parse_args()
    paths = (args.legacy_project, args.legacy_artifact, args.calibration, args.source_receipt)
    if any(paths) and not all(paths):
        parser.error("Provide all four binding paths, or none for unvalidated shape-only planning")
    if args.binding_receipt and not all(paths):
        parser.error("A binding receipt requires real full source/calibration/artifact validation")
    cache = None
    if all(paths):
        binding = verify_binding(*paths)
        names = list(binding["stats"])
        seed = binding["manifest"]["seed"]
        status = "source-calibration-artifact-bound-cpu-only"
        source_binding = binding["binding_sha256"]
        if args.binding_receipt:
            from binding_cache import write_binding_cache

            cache = write_binding_cache(
                binding,
                args.binding_receipt,
                root=args.legacy_project,
                artifact=args.legacy_artifact,
                calibration=args.calibration,
                source_receipt=args.source_receipt,
            )
    else:
        names = list(h3_layer_names())
        seed = 1101
        status = "shape-only-not-an-artifact-preflight"
        source_binding = None
    selected = representative_schedule(names, seed)
    storage = [layer_storage(name) for name in names]
    report = {
        "status": status,
        "source_binding_sha256": source_binding,
        "binding_cache": cache,
        "layer_count": len(names),
        "representatives": [
            {"ordinal": ordinal, "seed": layer_seed, **layer_storage(name)}
            for ordinal, name, layer_seed in selected
        ],
        "largest_bf16_tensor_bytes": max(row["dense_bf16_bytes"] for row in storage),
        "largest_fp32_tensor_bytes": max(row["one_fp32_matrix_bytes"] for row in storage),
        "one_candidate_all_layers_approx_bytes": sum(
            row["packed_payload_approx_bytes"] for row in storage
        ),
        "two_candidates_all_layers_approx_bytes": 2
        * sum(row["packed_payload_approx_bytes"] for row in storage),
        "schedule_note": (
            "Shape-only ordinals are not exporter ordinals; bound mode inherits actual merged-stats insertion order"
        ),
        "memory_note": (
            "Tensor inventory, not a hard peak bound. No full-model dense load. "
            "CUDA library/allocator/RSS peaks must be measured."
        ),
        "recommended_initial_reservation": {"gpus": 1, "gpu_memory_gib": 24, "host_memory_gib": 16},
        "first_gpu_stage": "compare_regenerated.py --representative-four --compat-pack nested_rcp_approx",
        "full_gpu_stage": (
            "compare_regenerated.py --compat-pack nested_rcp_approx "
            "(new output directory, omit representatives/slice)"
        ),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
