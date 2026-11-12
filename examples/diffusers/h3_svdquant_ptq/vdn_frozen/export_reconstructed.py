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

"""Export B-only reconstruction through ModelOpt; never silently refactor W.

The updated B compensates quantization error. Recomputing residual=W-B@A would
cancel that correction, so retain the original residual and verify its bytes.
"""

import copy
import io
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from .environment import Method, build, identity, read, sha, write
from .fit import replace_b
from .policy import groups, select
from .runtime import validate_payload

__all__ = ["run"]


def run(root, output):
    # Optional ModelOpt/GPU dependencies are only needed for export.
    from h3_modelopt_ptq import factorize_legacy, import_frozen
    from modelopt_multiplier_compat import pack_compatible_layer

    import modelopt.torch.opt as mto

    args = SimpleNamespace(fit=root / "reconstruction", variant="group", output=output)
    config = read(root / "config.json")
    bound = identity(root)
    baseline = root / "baseline"
    manifest = read(baseline / "manifest.json")
    receipt = read(args.fit / "complete.json")
    b_path = args.fit / "group_B.safetensors"
    library = Path(config["library"])
    if (
        manifest["binding"] != bound
        or receipt["binding"] != bound
        or receipt["status"] != "B-fit-screening-completed-export-not-yet-accepted"
        or receipt["baseline_manifest_sha256"] != sha(baseline / "manifest.json")
        or receipt["group_B_sha256"] != sha(b_path)
        or receipt["kernel_sha256"] != sha(library)
        or manifest["library_sha256"] != sha(library)
        or manifest["rank"] != 32
        or len(manifest["layers"]) != 208
    ):
        raise ValueError("Source/kernel/B-fit binding differs")
    # The reviewed recipe's group and full-block gates accepted the same updates.
    # Do not silently deploy groups vetoed by a block on a new calibration set.
    if receipt["block_accepted"] != receipt["group_accepted"] or receipt["block_B_sha256"] != sha(
        args.fit / "block_B.safetensors"
    ):
        raise ValueError("Full-block gates rejected updates; this is not the Frozen recipe")
    group_values = load_file(str(b_path))
    block_values = load_file(str(args.fit / "block_B.safetensors"))
    if set(group_values) != set(block_values) or any(
        not torch.equal(group_values[key], block_values[key]) for key in group_values
    ):
        raise ValueError("Group and block gates selected different corrections")
    updates = load_file(str(b_path))
    if not updates or not set(updates) <= {g.name for g in groups()}:
        raise ValueError("Empty or unknown B updates")
    if len(updates) != receipt[f"{args.variant}_accepted"]:
        raise ValueError("Accepted group count mismatch")
    stats_path = root / "calibration/activation_absmax.safetensors"
    if sha(stats_path) != manifest["calibration_sha256"]:
        raise ValueError("Original calibration mutated")
    args.output.mkdir(parents=True, exist_ok=False)
    model = build(config)
    stats = load_file(str(stats_path), device="cuda")
    selected = select(model.transformer)
    records = []

    def pack(layer, original):
        packed, normalized = pack_compatible_layer(layer, variant="nested_rcp_approx")
        del normalized
        packed["weight_scale"] = packed["weight_scale"].reshape_as(original["weight_scale"])
        return packed

    def assert_equal(actual, expected, stage):
        if set(actual) != set(expected):
            raise ValueError(f"{stage}: payload schema differs")
        bad = [key for key in expected if not torch.equal(actual[key], expected[key])]
        if bad:
            raise ValueError(f"{stage}: changed fields {bad}")

    for group, entry in zip(groups(), manifest["layers"], strict=True):
        if (entry["name"], entry["shape"], entry["modules"]) != (
            group.name,
            list(group.shape),
            list(group.modules),
        ):
            raise ValueError("Group metadata differs")
        source = baseline / entry["file"]
        if source.parent.resolve() != baseline.resolve() or sha(source) != entry["sha256"]:
            raise ValueError("Baseline artifact changed")
        target = args.output / entry["file"]
        record = copy.deepcopy(entry)
        record.update(baseline_sha256=entry["sha256"], reconstructed_B=group.name in updates)
        print(f"PHASE ModelOpt B export {len(records) + 1}/208 {group.name}", flush=True)
        if group.name not in updates:
            shutil.copyfile(source, target)
            if sha(target) != entry["sha256"]:
                raise ValueError("Unchanged copy differs")
            records.append(record)
            continue
        original = load_file(str(source), device="cuda")
        validate_payload(original, group.shape)
        expected = replace_b(original, updates[group.name].cuda())
        relative_change = float(
            (expected["svdquant_lora_b"].float() - original["svdquant_lora_b"].float()).norm()
            / original["svdquant_lora_b"].float().norm().clamp_min(1e-30)
        )
        if relative_change > 0.100001:
            raise ValueError("B update exceeds fitted trust bound")
        weight = torch.cat([layer.weight for layer in selected[group.name]], dim=0)
        frozen = factorize_legacy(weight, stats[group.name], seed=entry["seed"], rank=32)
        n, k = group.shape

        def empty(device):
            return torch.nn.Sequential(
                torch.nn.Linear(k, n, bias=False, device="meta", dtype=torch.bfloat16)
            ).to_empty(device=device)

        baseline_adapter = import_frozen(empty("cuda"), {"0": frozen}, require_full_h3=False)
        assert_equal(pack(baseline_adapter[0], original), original, "original factorization replay")
        del baseline_adapter
        # Do NOT update residual, A, smoothing, scales, rank or checkpoint LoRAs.
        corrected = replace(frozen, lora_b=expected["svdquant_lora_b"], packed=None)
        adapter = import_frozen(empty("cuda"), {"0": corrected}, require_full_h3=False)
        assert_equal(pack(adapter[0], original), expected, "ModelOpt B import")
        state = io.BytesIO()
        mto.save(adapter, state)
        del adapter
        state.seek(0)
        restored = mto.restore(empty("cpu"), state).cuda()
        state.close()
        payload = pack(restored[0], original)
        assert_equal(payload, expected, "ModelOpt B save/restore")
        validate_payload(payload, group.shape)
        gold_data, gold_scales = torch.ops.trtllm.fp4_quantize(
            frozen.residual.contiguous(),
            payload["weight_global_scale"].reshape(1).contiguous(),
            16,
            False,
            True,
        )
        if not torch.equal(gold_data.view(torch.uint8), payload["weight"]) or not torch.equal(
            gold_scales.view(torch.uint8).reshape(-1), payload["weight_scale"].reshape(-1)
        ):
            raise ValueError("Original residual TRT pack differs")
        method, expected_method = Method(payload), Method(expected)
        generator = torch.Generator(device="cuda").manual_seed(entry["seed"])
        x = (
            torch.randn((129, k), device="cuda", generator=generator)
            * stats[group.name].clamp_min(1e-8)
            / 3
        ).bfloat16()
        actual = method.apply(None, x)
        if (
            not torch.isfinite(actual).all()
            or not torch.equal(actual, method.apply(None, x))
            or not torch.equal(actual, expected_method.apply(None, x))
        ):
            raise ValueError("Actual-kernel B replay failed")
        save_file({key: value.cpu().contiguous() for key, value in payload.items()}, str(target))
        record.update(
            sha256=sha(target),
            modelopt_restore_exact=True,
            trt_pack_exact=True,
            gemm_repeat_exact=True,
            only_canonical_B_changed=True,
            relative_B_change=relative_change,
        )
        records.append(record)
        del (
            weight,
            frozen,
            corrected,
            original,
            expected,
            restored,
            payload,
            method,
            expected_method,
            actual,
            x,
            gold_data,
            gold_scales,
        )
    result = copy.deepcopy(manifest)
    result["layers"] = records
    result["optimization"] = {
        "kind": "fixed-rank32-canonical-B-reconstruction",
        "variant": args.variant,
        "baseline_manifest_sha256": sha(baseline / "manifest.json"),
        "fit_receipt_sha256": sha(args.fit / "complete.json"),
        "B_sha256": sha(b_path),
        "exporter_sha256": sha(__file__),
        "changed_groups": len(updates),
        "only_changed_field": "svdquant_lora_b",
        "status": "Export validated; independent full-trajectory and quality acceptance remain separate",
    }
    write(args.output / "manifest.json", result)
    write(
        args.output / "complete.json",
        {"manifest_sha256": sha(args.output / "manifest.json"), **result["optimization"]},
    )
