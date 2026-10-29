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

"""Report validation with synthetic metadata, independent of weights and GPU access."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "regeneration_summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_regeneration.py",
)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reports(tmp_path):
    exact = {"exact": True, "relative_l2": 0.0, "max_abs": 0.0}
    same = {"exact": True, "different_bytes": 0, "total_bytes": 100}
    mismatch = {"exact": False, "different_bytes": 1, "total_bytes": 100}
    rows = []
    for prefix, count in (("token_refiner.blocks", 2), ("blocks", 50)):
        for i in range(count):
            for suffix in summary.SUFFIXES:
                ordinal = len(rows)
                rows.append(
                    {
                        "layer": f"{prefix}.{i}.{suffix}",
                        "ordinal": ordinal,
                        "seed": 1101 + ordinal,
                        "factor_exact": dict.fromkeys(summary.FACTORS, True),
                        "packed_exact": dict.fromkeys(summary.PACKED, True),
                        "modelopt_native_quantization": {
                            "packed_tensors_exact": {k: k != "weight" for k in summary.PACKED},
                            "forward_metrics": [
                                {
                                    "rows": m,
                                    "breakdown": {
                                        "weight_data": mismatch,
                                        "weight_scales_swizzled": same,
                                        "lowrank": {"example": exact},
                                        "full_output": {"example": exact},
                                        "residual_gemm": exact,
                                        "fused_final_addition": exact,
                                        "activation": {
                                            "data": same,
                                            "reference": exact,
                                        },
                                    },
                                }
                                for m in (1, 128, 129)
                            ],
                        },
                        "modelopt_compat_quantization": {
                            "variant": "nested_rcp_approx",
                            "packed_tensors_exact": dict.fromkeys(summary.PACKED, True),
                            "weight_data": same,
                            "weight_scale": same,
                            "forward_metrics": [
                                {
                                    "rows": m,
                                    "same_trt_vs_gold": exact,
                                    "repeat_exact": True,
                                }
                                for m in (1, 128, 129)
                            ],
                        },
                        "memory": dict.fromkeys(
                            (
                                "cuda_peak_allocated_bytes",
                                "cuda_peak_reserved_bytes",
                                "host_process_maxrss_kib_lifetime",
                            ),
                            100,
                        ),
                    }
                )
    for kind in ("native", "compat"):
        write(
            tmp_path / f"{kind}_modelopt_candidate/manifest.json",
            {
                "seed": 1101,
                "layers": {r["layer"]: {"bytes": 100} for r in rows},
            },
        )
    write(
        tmp_path / "complete.json",
        {
            "schema": "h3-modelopt-recipe-regeneration-v1",
            "status": "all-208-exact",
            "layers": 208,
            "start": 0,
            "stop": None,
            "representative_four": False,
            "native_candidate_manifest_sha256": summary.sha(
                tmp_path / "native_modelopt_candidate/manifest.json"
            ),
            "compat_candidate": {
                "manifest_sha256": summary.sha(tmp_path / "compat_modelopt_candidate/manifest.json")
            },
            "source_binding_sha256": {"synthetic_test_only": "no-weight-evidence"},
            "torch": "synthetic",
            "cuda": "none",
            "gpu": "none",
        },
    )
    update_rows(tmp_path, rows)
    return tmp_path, rows


def update_rows(path, rows):
    (path / "layers.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    complete = summary.read(path / "complete.json")
    complete["layer_records_sha256"] = summary.sha(path / "layers.jsonl")
    write(path / "complete.json", complete)


def test_weight_counts_once_not_three_times(reports):
    root, _ = reports
    report = summary.summarize(root)
    weights = report["native_weight_bytes_counted_once_per_layer"]["weight_data"]
    assert weights["total_bytes"] == 20800 and weights["different_bytes"] == 208
    assert report["activation_bytes_all_624_cases"]["data"]["total_bytes"] == 62400
    assert report["factor_exact_checks"] == 832
    assert report["legacy_packed_exact_checks"] == 1664
    assert report["compat_repeat_exact_cases"] == 624


@pytest.mark.parametrize(
    "problem", ["duplicate", "partial", "ordinal", "seed", "rows", "factor", "variant"]
)
def test_missing_or_mislabeled_coverage_rejected(reports, problem):
    root, rows = reports
    rows = copy.deepcopy(rows)
    if problem == "duplicate":
        rows[-1] = rows[0]
    elif problem == "partial":
        rows.pop()
    elif problem in ("ordinal", "seed"):
        rows[0][problem] += 1
    elif problem == "rows":
        rows[0]["modelopt_native_quantization"]["forward_metrics"].pop()
    elif problem == "factor":
        rows[0]["factor_exact"]["residual"] = False
    else:
        rows[0]["modelopt_compat_quantization"]["variant"] = "different"
    update_rows(root, rows)
    with pytest.raises(ValueError):
        summary.summarize(root)


def test_content_and_manifest_mutation_rejected(reports):
    root, _ = reports
    path = root / "native_modelopt_candidate/manifest.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest hash"):
        summary.summarize(root)


def test_errors_are_not_pooled_or_hidden():
    result = summary.metric_summary(
        [
            {
                "layer": "first",
                "rows": 1,
                "metric": {"exact": True, "max_abs": 0, "relative_l2": 0},
            },
            {
                "layer": "second",
                "rows": 129,
                "metric": {"exact": False, "max_abs": 4, "relative_l2": 0.1},
            },
        ]
    )
    assert result["relative_l2_median"] == 0.05
    assert result["worst_relative_l2"]["layer"] == "second"
    with pytest.raises(ValueError, match="Exact"):
        summary.metric_summary([{"metric": {"exact": True, "max_abs": 4, "relative_l2": 0}}])
    with pytest.raises(ValueError, match="byte-count"):
        summary.byte_summary([{"exact": True, "different_bytes": 1, "total_bytes": 100}])
