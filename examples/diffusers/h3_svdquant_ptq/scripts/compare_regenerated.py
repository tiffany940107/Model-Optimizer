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

"""Regenerate one H3 projection at a time and compare to the bound legacy export.

This is PTQ recipe regeneration using frozen collected statistics, including
separate native and optional compatibility pack measurements, not new
calibration-data collection, QAD training or an E2E quality approval.
"""

import argparse
import gc
import io
import json
import resource
import sys
from pathlib import Path

import torch
from artifact_binding import digest, local_file, validate_payload, verify_binding
from h3_modelopt_ptq import export_frozen, import_frozen
from modelopt_multiplier_compat import VARIANTS, pack_compatible_layer
from native_breakdown import byte_comparison, layer_breakdown
from packed_replay import make_empty_layer
from recipe_comparison import (
    compare_factorizations,
    layer_schedule,
    load_legacy_core,
    pack_for_trt,
    pack_modelopt_for_trt,
    reorder_qkv,
    tensor_errors,
)
from regeneration_planning import representative_schedule
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from streaming_roundtrip import check_unchanged

import modelopt.torch.opt as mto


@torch.no_grad()
def main():
    """Run exact factor + true TRT packed-artifact comparison with no full-model GPU load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-project", type=Path, required=True)
    parser.add_argument("--legacy-artifact", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument(
        "--binding-receipt",
        type=Path,
        help="Reuse CPU full-hash preflight with all signatures/small hashes checked",
    )
    parser.add_argument("--trt-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--compat-pack", choices=VARIANTS)
    parser.add_argument("--representative-four", action="store_true")
    args = parser.parse_args()
    if args.binding_receipt:
        from binding_cache import load_binding_cache

        binding = load_binding_cache(
            args.binding_receipt,
            root=args.legacy_project,
            artifact=args.legacy_artifact,
            calibration=args.calibration,
            source_receipt=args.source_receipt,
        )
    else:
        binding = verify_binding(
            args.legacy_project, args.legacy_artifact, args.calibration, args.source_receipt
        )
    if (
        sys.version_info[:2] != (3, 12)
        or torch.__version__ != "2.11.0+cu130"
        or torch.version.cuda != "13.0"
        or torch.cuda.get_device_capability() != (10, 0)
    ):
        raise RuntimeError(
            "Exact recipe comparison requires the original Python3.12/torch2.11+cu130 SM100 environment"
        )
    provenance = binding["provenance"]
    if (
        provenance["torch"] != torch.__version__
        or digest(args.trt_library) != provenance["kernel_sha256"]
    ):
        raise ValueError("Golden artifact runtime/library differs from regeneration")
    # Matching Torch/GPU family is necessary, not sufficient for deterministic
    # SVD. Record runtime details; actual exact-factor gates remain authoritative.
    torch.ops.load_library(str(args.trt_library))
    legacy = load_legacy_core(
        args.legacy_project, provenance["source_code_sha256"]["svdquant_h3/core.py"]
    )
    schedule = layer_schedule(binding["stats"], binding["manifest"]["seed"], args.start, args.stop)
    if args.representative_four:
        if args.start != 0 or args.stop is not None:
            parser.error("--representative-four cannot be combined with --start/--stop")
        schedule = representative_schedule(binding["stats"], binding["manifest"]["seed"])
    args.output.mkdir(parents=True, exist_ok=False)
    native_dir = args.output / "native_modelopt_candidate"
    native_dir.mkdir()
    native_layers = {}
    compat_layers = {}
    compat_dir = args.output / "compat_modelopt_candidate"
    if args.compat_pack:
        compat_dir.mkdir()
    results = []
    for ordinal, name, seed in schedule:
        torch.cuda.reset_peak_memory_stats()
        key = name + ".weight"
        shard = binding["source_index"][key]
        with safe_open(
            binding["model"] / "transformer" / shard, framework="pt", device="cpu"
        ) as stream:
            weight = stream.get_tensor(key).to("cuda")
        if name.endswith(".attn.qkv_proj"):
            weight = reorder_qkv(weight)
        stats = binding["stats"][name].cuda()
        factors, equality = compare_factorizations(weight, stats, legacy, seed=seed)
        record = {"layer": name, "ordinal": ordinal, "seed": seed, "factor_exact": equality}
        if all(equality.values()):
            n, k = weight.shape
            del weight
            adapter = make_empty_layer(n, k).to_empty(device="cuda")
            adapter = import_frozen(adapter, {"0": factors}, require_full_h3=False)
            # Exercise real dense ModelOpt state, one layer at a time. BytesIO
            # avoids retaining 208 BF16 residual checkpoint copies on disk.
            checkpoint = io.BytesIO()
            mto.save(adapter, checkpoint)
            del adapter
            checkpoint.seek(0)
            adapter = mto.restore(make_empty_layer(n, k).to_empty(device="cpu"), checkpoint).cuda()
            checkpoint.close()
            roundtrip_factors = export_frozen(adapter[0])
            candidate = pack_for_trt(roundtrip_factors)
            entry = binding["manifest"]["layers"][name]
            gold_path = local_file(args.legacy_artifact, entry["file"])
            if digest(gold_path) != entry["sha256"]:
                raise ValueError("Golden layer mutated after binding")
            golden = load_file(str(gold_path), device="cuda")
            validate_payload(name, candidate, stats)
            record["packed_exact"] = {
                key: torch.equal(value, golden[key]) for key, value in candidate.items()
            }
            # This is the actual ModelOpt quantization path, kept distinct from
            # the frozen/legacy packer compatibility gate above.
            native, native_dequant = pack_modelopt_for_trt(adapter[0])
            native["weight_scale"] = native["weight_scale"].reshape_as(golden["weight_scale"])
            native_fake = adapter[0].weight_quantizer(adapter[0].weight)
            golden_dequant = legacy.dequantize_nvfp4(
                golden["weight"], golden["weight_scale"], golden["weight_global_scale"]
            )
            record["modelopt_native_quantization"] = {
                "packed_tensors_exact": {
                    key: torch.equal(value, golden[key]) for key, value in native.items()
                },
                "export_dequant_vs_legacy": tensor_errors(native_dequant, golden_dequant),
                "tensor_quantizer_vs_export": tensor_errors(native_fake, native_dequant),
            }
            del native_fake, native_dequant, golden_dequant
            method = legacy.TRTNVFP4SVDQuantLinearMethod(native)
            gold_method = legacy.TRTNVFP4SVDQuantLinearMethod(golden)
            if args.compat_pack:
                if n % 128 or k % 64:
                    raise ValueError("Full-byte H3 scale gate requires no unused scale padding")
                compat, normalized = pack_compatible_layer(adapter[0], variant=args.compat_pack)
                del normalized
                compat["weight_scale"] = compat["weight_scale"].reshape_as(golden["weight_scale"])
                validate_payload(name, compat, stats)
                compat_method = legacy.TRTNVFP4SVDQuantLinearMethod(compat)
                record["modelopt_compat_quantization"] = {
                    "variant": args.compat_pack,
                    "packed_tensors_exact": {
                        key: torch.equal(value, golden[key]) for key, value in compat.items()
                    },
                    "weight_data": byte_comparison(compat["weight"], golden["weight"]),
                    "weight_scale": byte_comparison(compat["weight_scale"], golden["weight_scale"]),
                    "scale_padding": "H3 N and K are multiples of 128 and 64: no unused scale padding",
                    "forward_metrics": [],
                }
            forward_metrics = []
            for rows in (1, 128, 129):
                span = stats.float().clamp_min(1e-8)
                x = (torch.randn(rows, k, device="cuda") * (span / 3)).clamp(-span, span).bfloat16()
                old_output = gold_method.apply(None, x)
                native_output = method.apply(None, x)
                modelopt_output = adapter(x)
                if args.compat_pack:
                    compat_output = compat_method.apply(None, x)
                    compat_repeat = compat_method.apply(None, x)
                    record["modelopt_compat_quantization"]["forward_metrics"].append(
                        {
                            "rows": rows,
                            "same_trt_vs_gold": tensor_errors(compat_output, old_output),
                            "repeat_exact": torch.equal(compat_output, compat_repeat),
                        }
                    )
                    del compat_output, compat_repeat
                forward_metrics.append(
                    {
                        "rows": rows,
                        "breakdown": layer_breakdown(
                            adapter[0],
                            legacy,
                            golden,
                            native,
                            x,
                            old_output=old_output,
                            native_output=native_output,
                            modelopt_output=modelopt_output,
                        ),
                    }
                )
            record["modelopt_native_quantization"]["forward_metrics"] = forward_metrics
            record["modelopt_native_quantization"]["acceptance"] = (
                "measured candidate, not quality-equivalence approval"
            )
            native_path = native_dir / entry["file"]
            save_file(
                {key: value.cpu().contiguous() for key, value in native.items()}, str(native_path)
            )
            native_layers[name] = {
                **entry,
                "sha256": digest(native_path),
                "bytes": native_path.stat().st_size,
            }
            if args.compat_pack:
                compat_path = compat_dir / entry["file"]
                save_file(
                    {key: value.cpu().contiguous() for key, value in compat.items()},
                    str(compat_path),
                )
                compat_layers[name] = {
                    **entry,
                    "sha256": digest(compat_path),
                    "bytes": compat_path.stat().st_size,
                }
                del compat, compat_method
            del native, method, gold_method, x, old_output, native_output, modelopt_output, span
            del candidate, golden, adapter, roundtrip_factors
        record["memory"] = {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "host_process_maxrss_kib_lifetime": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "single-process one-layer CUDA peaks; RSS is process-lifetime Linux high-water mark",
        }
        with (args.output / "layers.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if not all(equality.values()) or not all(record.get("packed_exact", {}).values()):
            raise ValueError(f"Exact recipe/packing mismatch at {name}; no tolerance relaxation")
        results.append(record)
        del factors, stats
        gc.collect()
        torch.cuda.empty_cache()
    check_unchanged(binding["signatures"])
    full = len(results) == 208
    native_manifest = {
        **binding["manifest"],
        "layers": native_layers,
        "format": "h3_modelopt_native_nvfp4_candidate_v1",
        "quantizer": "ModelOpt NVFP4QTensor with native global dequant scale",
        "scope": "candidate for measured E2E, not legacy-exact or accepted quality",
    }
    with (native_dir / "manifest.json").open("x") as stream:
        json.dump(native_manifest, stream, indent=2)
    if args.compat_pack:
        compat_manifest = {
            **binding["manifest"],
            "layers": compat_layers,
            "format": "h3_modelopt_multiplier_nvfp4_candidate_v1",
            "quantizer": f"ModelOpt MultiplierNVFP4QTensor/{args.compat_pack}",
            "scope": "independent candidate; this manifest does not inherit old kernel-validation receipts",
        }
        with (compat_dir / "manifest.json").open("x") as stream:
            json.dump(compat_manifest, stream, indent=2)
    receipt = {
        "schema": "h3-modelopt-recipe-regeneration-v1",
        "status": "all-208-exact" if full else "slice-exact-not-full-model",
        "layers": len(results),
        "start": args.start,
        "stop": args.stop,
        "representative_four": args.representative_four,
        "layer_records_sha256": digest(args.output / "layers.jsonl"),
        "source_binding_sha256": binding["binding_sha256"],
        "binding_verification": binding.get(
            "binding_verification", {"mode": "full-content-hash-in-this-process"}
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "native_candidate_manifest_sha256": digest(native_dir / "manifest.json"),
        "native_candidate_status": "needs matched E2E quality gate",
        "scope": (
            "Frozen-statistics legacy recipe regenerated through ModelOpt factors and original TRT packer; "
            "no new calibration/data/E2E/native-quantizer quality claim"
        ),
    }
    if args.compat_pack:
        compat_exact = all(
            all(row["modelopt_compat_quantization"]["packed_tensors_exact"].values())
            and all(
                metric["same_trt_vs_gold"]["exact"] and metric["repeat_exact"]
                for metric in row["modelopt_compat_quantization"]["forward_metrics"]
            )
            for row in results
        )
        receipt["compat_candidate"] = {
            "variant": args.compat_pack,
            "manifest_sha256": digest(compat_dir / "manifest.json"),
            "observed_packed_and_output_exact": compat_exact,
            "status": "measured-not-e2e-approved",
            "full_208_layers": full,
            "scope": "Actual ModelOpt QTensor packing, same legacy TRT GEMM; not native forward equivalence",
        }
    with (args.output / "complete.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)


if __name__ == "__main__":
    main()
