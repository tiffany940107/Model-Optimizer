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

"""From new calibration to ModelOpt export, without any legacy project directory."""

import io
from pathlib import Path

import torch
from artifact_binding import validate_payload
from h3_modelopt_ptq import export_frozen, factorize_legacy, import_frozen
from modelopt_multiplier_compat import pack_compatible_layer
from portable_calibration import load_calibration
from portable_io import digest, local_file, read_json, source_identity, validate_order, write_json
from portable_runtime import PackedLinear, load_runtime
from recipe_comparison import pack_for_trt, reorder_qkv
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

import modelopt.torch.opt as mto

__all__ = ["export_model", "modelopt_roundtrip", "verify_export"]

RECIPE = {
    "rank": 32,
    "alpha": 0.5,
    "seed": 1101,
    "pack": "nested_rcp_approx",
    "weight_quantizer": "rtn",
    "factorizer": "global",
}


def modelopt_roundtrip(factors):
    """Use actual ModelOpt save/restore on one layer without storing all dense residuals."""
    n, k = factors.residual.shape
    model = nn.Sequential(
        nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device=factors.residual.device)
    )
    import_frozen(model, {"0": factors}, require_full_h3=False)
    state = io.BytesIO()
    mto.save(model, state)
    state.seek(0)
    restored = nn.Sequential(
        nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device=factors.residual.device)
    )
    mto.restore(restored, state)
    # Restored dynamic quantizer buffers may initially be on CPU.
    restored.to(factors.residual.device)
    actual = export_frozen(restored[0])
    for name in ("pre_quant_scale", "residual", "lora_a", "lora_b", "activation_absmax"):
        if not torch.equal(getattr(actual, name), getattr(factors, name)):
            raise ValueError("ModelOpt checkpoint roundtrip changed factor " + name)
    return restored[0]


@torch.no_grad()
def export_model(model, calibration, output, *, library, library_sha256):
    """Export all208 with per-layer real TRT packing/GEMM/repeat gates on B200."""
    model, calibration, output = Path(model), Path(calibration), Path(output)
    if output.exists():
        raise FileExistsError(output)
    collected, stats = load_calibration(calibration)
    calibration_hash = digest(calibration / "complete.json")
    identity = source_identity(model)
    if identity != collected["source"]:
        raise ValueError("Source weights changed since calibration")
    load_runtime(library, library_sha256)
    shapes = validate_order(collected["layer_order"])
    index = read_json(model / "transformer/model.safetensors.index.json")["weight_map"]
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for ordinal, name in enumerate(collected["layer_order"]):
        n, k = shapes[name]
        shard = local_file(model / "transformer", index[name + ".weight"])
        with safe_open(shard, framework="pt", device="cpu") as stream:
            weight = stream.get_tensor(name + ".weight")
        if weight.dtype != torch.bfloat16 or weight.shape != (n, k):
            raise ValueError("Source weight shape/dtype differs: " + name)
        weight = weight.cuda()
        if name.endswith(".attn.qkv_proj"):
            weight = reorder_qkv(weight)
        factors = factorize_legacy(weight, stats[name].cuda(), seed=1101 + ordinal, rank=32)
        layer = modelopt_roundtrip(factors)
        candidate, _ = pack_compatible_layer(layer, variant="nested_rcp_approx")
        golden = pack_for_trt(export_frozen(layer))
        validate_payload(name, candidate, stats[name].cuda())
        if set(candidate) != set(golden) or not all(
            torch.equal(candidate[key], golden[key]) for key in golden
        ):
            raise ValueError("New ModelOpt pack differs from actual TRT pack: " + name)
        actual, reference = PackedLinear(candidate), PackedLinear(golden)
        checks = []
        generator = torch.Generator(device="cuda").manual_seed(1101 + ordinal)
        for rows in (1, 128, 129):
            x = torch.randn(rows, k, generator=generator, device="cuda", dtype=torch.bfloat16)
            y, y_ref = actual.apply(None, x), reference.apply(None, x)
            repeat = actual.apply(None, x)
            if (
                not torch.isfinite(y).all()
                or not torch.equal(y, y_ref)
                or not torch.equal(y, repeat)
            ):
                raise ValueError("Same-kernel/repeat comparison failed: " + name)
            checks.append({"rows": rows, "same_kernel_exact": True, "repeat_exact": True})
        filename = f"{ordinal:04d}.safetensors"
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in candidate.items()},
            str(output / filename),
        )
        records[name] = {
            "file": filename,
            "sha256": digest(output / filename),
            "shape": [n, k],
            "seed": 1101 + ordinal,
            "checks": checks,
        }
        print(f"exported {ordinal + 1}/208 {name}", flush=True)
        del weight, factors, layer, candidate, golden, actual, reference, x, y, y_ref, repeat
    if (
        source_identity(model) != identity
        or digest(calibration / "complete.json") != calibration_hash
        or digest(calibration / "absmax.safetensors") != collected["sha256"]
    ):
        raise ValueError("Source/calibration changed during export")
    save_file(stats, str(output / "absmax.safetensors"))
    write_json(
        output / "manifest.json",
        {
            "format": "h3-modelopt-portable-ptq-v1",
            "status": "all208-pack-and-same-kernel-exact-not-e2e",
            "source": identity,
            "calibration_sha256": calibration_hash,
            "calibration_requests_sha256": collected["requests_sha256"],
            "calibration_media_sha256": collected["media_sha256"],
            "stats_sha256": digest(output / "absmax.safetensors"),
            "layer_order": collected["layer_order"],
            "layers": records,
            "recipe": RECIPE,
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "library_sha256": library_sha256,
            },
            "scope": "New source-bound recipe; no inherited historical video or quality acceptance",
        },
    )


def verify_export(model, artifact):
    """Validate full content, scope, source and packed formats before inference."""
    artifact = Path(artifact)
    manifest = read_json(local_file(artifact, "manifest.json"))
    if (
        manifest.get("format") != "h3-modelopt-portable-ptq-v1"
        or manifest.get("status") != "all208-pack-and-same-kernel-exact-not-e2e"
        or manifest["source"] != source_identity(model)
    ):
        raise ValueError("Missing/foreign/incomplete portable export")
    shapes = validate_order(manifest["layer_order"])
    if (
        manifest["recipe"] != RECIPE
        or manifest["runtime"]["torch"] != "2.11.0+cu130"
        or manifest["runtime"]["cuda"] != "13.0"
    ):
        raise ValueError("Export recipe or ABI differs")
    if set(manifest["layers"]) != set(shapes):
        raise ValueError("Wrong exported layer coverage")
    stats_path = local_file(artifact, "absmax.safetensors")
    if digest(stats_path) != manifest["stats_sha256"]:
        raise ValueError("Export calibration changed")
    stats = load_file(str(stats_path))
    if set(stats) != set(shapes):
        raise ValueError("Export calibration coverage differs")
    for ordinal, name in enumerate(manifest["layer_order"]):
        record = manifest["layers"][name]
        if (
            record["file"] != f"{ordinal:04d}.safetensors"
            or record["seed"] != 1101 + ordinal
            or record["shape"] != list(shapes[name])
            or record["checks"]
            != [
                {"rows": rows, "same_kernel_exact": True, "repeat_exact": True}
                for rows in (1, 128, 129)
            ]
        ):
            raise ValueError("Export coverage/order/shape or measured gates differ")
        path = local_file(artifact, record["file"])
        if digest(path) != record["sha256"]:
            raise ValueError("Packed layer changed: " + name)
        validate_payload(name, load_file(str(path)), stats[name])
    return manifest
