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

"""CPU-only aggregate of full-208 numerical reports; no weight loading or E2E claim."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
PACKED = {
    "weight",
    "weight_scale",
    "weight_global_scale",
    "input_scale",
    "alpha",
    "pre_quant_scale",
    "svdquant_lora_a",
    "svdquant_lora_b",
}
FACTORS = {"pre_quant_scale", "residual", "lora_a", "lora_b"}


def sha(path):
    """Hash report bytes, not model payloads."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    """Reject nonfinite JSON values."""
    value = json.loads(Path(path).read_text())
    json.dumps(value, allow_nan=False)
    return value


def byte_summary(records):
    """Aggregate raw bytes with explicit denominators (not FP4 element counts)."""
    for row in records:
        if (
            type(row.get("different_bytes")) is not int
            or type(row.get("total_bytes")) is not int
            or not 0 <= row["different_bytes"] <= row["total_bytes"]
            or row["total_bytes"] <= 0
            or row.get("exact") is not (row["different_bytes"] == 0)
        ):
            raise ValueError("Invalid byte-count/exactness evidence")
    different = sum(r["different_bytes"] for r in records)
    total = sum(r["total_bytes"] for r in records)
    return {
        "comparisons": len(records),
        "exact_comparisons": sum(r["exact"] for r in records),
        "different_bytes": different,
        "total_bytes": total,
        "different_byte_fraction": different / total,
    }


def metric_summary(records):
    """Report distribution across comparisons, never pretend it is pooled output L2."""
    for record in records:
        row = record["metric"]
        if type(row.get("exact")) is not bool or any(
            not isinstance(row.get(key), (int, float))
            or not math.isfinite(row[key])
            or row[key] < 0
            for key in ("relative_l2", "max_abs")
        ):
            raise ValueError("Invalid finite error metric")
        if row["exact"] and (row["relative_l2"] != 0 or row["max_abs"] != 0):
            raise ValueError("Exact metric cannot have nonzero error")
    worst = max(records, key=lambda r: r["metric"]["relative_l2"])
    values = [r["metric"]["relative_l2"] for r in records]
    return {
        "comparisons": len(records),
        "exact_comparisons": sum(r["metric"]["exact"] for r in records),
        "relative_l2_min": min(values),
        "relative_l2_median": statistics.median(values),
        "relative_l2_max": max(values),
        "max_abs_over_comparisons": max(r["metric"]["max_abs"] for r in records),
        "worst_relative_l2": worst,
        "aggregation": "unweighted per-comparison distribution, not pooled L2 or E2E quality",
    }


def summarize(directory):
    """Fail closed on partial/duplicate reports and stale candidate manifests."""
    directory = Path(directory)
    complete = read(directory / "complete.json")
    if (
        complete.get("schema") != "h3-modelopt-recipe-regeneration-v1"
        or complete.get("status") != "all-208-exact"
        or complete.get("layers") != 208
        or complete.get("start") != 0
        or complete.get("stop") is not None
        or complete.get("representative_four") is not False
        or sha(directory / "layers.jsonl") != complete.get("layer_records_sha256")
    ):
        raise ValueError("Not a content-bound full-208 receipt")
    expected = {
        f"{prefix}.{i}.{suffix}"
        for prefix, count in (("blocks", 50), ("token_refiner.blocks", 2))
        for i in range(count)
        for suffix in SUFFIXES
    }
    layers = [json.loads(line) for line in (directory / "layers.jsonl").read_text().splitlines()]
    json.dumps(layers, allow_nan=False)
    if len(layers) != 208 or {r["layer"] for r in layers} != expected:
        raise ValueError("Missing or duplicate original H3 layer")
    manifests = {}
    for kind, key in (("native", "native_candidate_manifest_sha256"), ("compat", None)):
        path = directory / f"{kind}_modelopt_candidate/manifest.json"
        expected_hash = complete[key] if key else complete["compat_candidate"]["manifest_sha256"]
        if sha(path) != expected_hash:
            raise ValueError("Candidate manifest hash mismatch")
        manifests[kind] = read(path)
        if set(manifests[kind]["layers"]) != expected:
            raise ValueError("Candidate manifest is not full-208")
    seed = manifests["compat"]["seed"]
    if list(manifests["compat"]["layers"]) != [r["layer"] for r in layers]:
        raise ValueError("Layer order differs from candidate manifest")
    for ordinal, record in enumerate(layers):
        if record["ordinal"] != ordinal or record["seed"] != seed + ordinal:
            raise ValueError("Layer ordinal/seed mismatch")
        for name, keys in (("factor_exact", FACTORS), ("packed_exact", PACKED)):
            if set(record[name]) != keys or any(v is not True for v in record[name].values()):
                raise ValueError("Legacy regeneration gate did not pass")
        for kind in ("native", "compat"):
            item = record[f"modelopt_{kind}_quantization"]
            if set(item["packed_tensors_exact"]) != PACKED or any(
                type(v) is not bool for v in item["packed_tensors_exact"].values()
            ):
                raise ValueError("Incomplete packed-tensor comparisons")
            if [r["rows"] for r in item["forward_metrics"]] != [1, 128, 129]:
                raise ValueError("Missing/duplicate rows1/128/129 coverage")
        if record["modelopt_compat_quantization"]["variant"] != "nested_rcp_approx":
            raise ValueError("Report uses a different compatibility variant")
    native = [r["modelopt_native_quantization"] for r in layers]
    compat = [r["modelopt_compat_quantization"] for r in layers]
    # Weight evidence is repeated for three M cases: count it only once/layer.
    weight = {}
    for key in ("weight_data", "weight_scales_swizzled"):
        for item in native:
            if any(
                r["breakdown"][key] != item["forward_metrics"][0]["breakdown"][key]
                for r in item["forward_metrics"]
            ):
                raise ValueError("Weight evidence changed across M cases")
        weight[key] = byte_summary([r["forward_metrics"][0]["breakdown"][key] for r in native])
    metrics, activations = {}, {}
    for record in layers:
        for case in record["modelopt_native_quantization"]["forward_metrics"]:
            breakdown = case["breakdown"]
            for group in ("lowrank", "full_output"):
                for name, item in breakdown[group].items():
                    metrics.setdefault(group + "/" + name, []).append(
                        {"layer": record["layer"], "rows": case["rows"], "metric": item}
                    )
            for name in ("residual_gemm", "fused_final_addition"):
                metrics.setdefault(name, []).append(
                    {
                        "layer": record["layer"],
                        "rows": case["rows"],
                        "metric": breakdown[name],
                    }
                )
            for name, item in breakdown["activation"].items():
                if isinstance(item, dict):
                    if "total_bytes" in item:
                        activations.setdefault(name, []).append(item)
                    else:
                        metrics.setdefault("activation/" + name, []).append(
                            {
                                "layer": record["layer"],
                                "rows": case["rows"],
                                "metric": item,
                            }
                        )
        for case in record["modelopt_compat_quantization"]["forward_metrics"]:
            if type(case["repeat_exact"]) is not bool:
                raise ValueError("Missing compat repeat comparison")
            metrics.setdefault("compat_same_trt_vs_gold", []).append(
                {
                    "layer": record["layer"],
                    "rows": case["rows"],
                    "metric": case["same_trt_vs_gold"],
                }
            )
    return {
        "schema": "h3-modelopt-regeneration-summary-v1",
        "evidence": {
            "complete_sha256": sha(directory / "complete.json"),
            "layers_sha256": complete["layer_records_sha256"],
            "binding": complete["source_binding_sha256"],
            "runtime": {k: complete[k] for k in ("torch", "cuda", "gpu")},
        },
        "layers": 208,
        "forward_cases_per_path": 624,
        "factor_exact_checks": 208 * 4,
        "legacy_packed_exact_checks": 208 * 8,
        "native_packed_exact_layers": {
            key: sum(r["packed_tensors_exact"][key] for r in native) for key in sorted(PACKED)
        },
        "compat_packed_exact_layers": {
            key: sum(r["packed_tensors_exact"][key] for r in compat) for key in sorted(PACKED)
        },
        "native_weight_bytes_counted_once_per_layer": weight,
        "compat_weight_bytes_counted_once_per_layer": {
            key: byte_summary([r[key] for r in compat]) for key in ("weight_data", "weight_scale")
        },
        "activation_bytes_all_624_cases": {k: byte_summary(v) for k, v in activations.items()},
        "metrics": {k: metric_summary(v) for k, v in metrics.items()},
        "compat_repeat_exact_cases": sum(
            case["repeat_exact"] for r in compat for case in r["forward_metrics"]
        ),
        "memory_maxima": {
            key: max(r["memory"][key] for r in layers)
            for key in (
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "host_process_maxrss_kib_lifetime",
            )
        },
        "candidate_manifest_total_file_bytes": {
            kind: sum(r["bytes"] for r in manifest["layers"].values())
            for kind, manifest in manifests.items()
        },
        "scope": (
            "Reconciled report/manifest metadata only; model payloads not locally rehashed. "
            "Random diagnostic inputs scaled by real calibration, not captured full-model activations. "
            "Native means default ModelOpt pack/forward on legacy factors, not ModelOpt default AWQ/SVD. "
            "Matched E2E and visual quality remain separate pending gates."
        ),
    }


def main():
    """Emit a new summary without overwriting experiment evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "layers": report["layers"],
                "cases": report["forward_cases_per_path"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
