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

"""Fresh VDN calibration and actual ModelOpt baseline export."""

import io

import torch
from portable_io import local_file
from safetensors.torch import load_file, save_file

from .environment import Method, read, sha, write
from .policy import groups, select
from .runtime import validate_payload

__all__ = ["Calibration", "export", "load_methods"]


class Calibration:
    def __init__(self, model):
        self.stats, self.counts, self.handles = {}, {}, []
        for name, layers in select(model).items():
            self.stats[name] = torch.zeros(layers[0].in_features, device=layers[0].weight.device)
            self.counts[name] = 0

            def observe(_layer, args, name=name):
                value = args[0].detach().reshape(-1, args[0].shape[-1])
                for chunk in value.split(1024):
                    span = chunk.float().abs().amax(0)
                    if not torch.isfinite(span).all():
                        raise ValueError("Nonfinite calibration: " + name)
                    self.stats[name] = torch.maximum(self.stats[name], span)
                self.counts[name] += 1

            self.handles.append(layers[0].register_forward_pre_hook(observe))

    def finish(self, output, calls):
        for handle in self.handles:
            handle.remove()
        if len(self.stats) != 208 or any(count != calls for count in self.counts.values()):
            raise ValueError("Calibration did not observe all 208 groups at every NFE")
        save_file(
            {name: value.cpu().contiguous() for name, value in self.stats.items()}, str(output)
        )
        return self.counts


def export(model, statistics, output, binding, library):
    from h3_modelopt_ptq import factorize_legacy, import_frozen
    from modelopt_multiplier_compat import pack_compatible_layer

    import modelopt.torch.opt as mto

    selected = select(model)
    stats = load_file(str(statistics), device="cuda")
    if set(stats) != set(selected):
        raise ValueError("Statistics do not match VDN inherited scope")
    torch.ops.load_library(str(library))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for ordinal, group in enumerate(groups()):
        print(f"PHASE factorize/ModelOpt pack {ordinal + 1}/208 {group.name}", flush=True)
        weight = torch.cat([layer.weight for layer in selected[group.name]], dim=0)
        n, k = weight.shape
        seed = 1101 + ordinal
        factors = factorize_legacy(weight, stats[group.name], seed=seed, rank=32)
        adapter = torch.nn.Sequential(
            torch.nn.Linear(k, n, bias=False, device="meta", dtype=torch.bfloat16)
        ).to_empty(device="cuda")
        adapter = import_frozen(adapter, {"0": factors}, require_full_h3=False)
        packed, normalized = pack_compatible_layer(adapter[0], variant="nested_rcp_approx")
        del normalized
        # Compare actual ModelOpt-produced bytes with the deployed TRT packer.
        gold_data, gold_scales = torch.ops.trtllm.fp4_quantize(
            factors.residual.contiguous(),
            packed["weight_global_scale"].reshape(1).contiguous(),
            16,
            False,
            True,
        )
        packed["weight_scale"] = packed["weight_scale"].reshape_as(gold_scales)
        if not torch.equal(gold_data.view(torch.uint8), packed["weight"]) or not torch.equal(
            gold_scales.view(torch.uint8), packed["weight_scale"]
        ):
            raise ValueError("ModelOpt/TRT packed bytes differ: " + group.name)
        # Preserve and restore the actual ModelOpt state, not only packed bytes.
        state = io.BytesIO()
        mto.save(adapter, state)
        del adapter
        state.seek(0)
        restored = mto.restore(
            torch.nn.Sequential(
                torch.nn.Linear(k, n, bias=False, device="meta", dtype=torch.bfloat16)
            ).to_empty(device="cpu"),
            state,
        ).cuda()
        state.close()
        replay, normalized = pack_compatible_layer(restored[0], variant="nested_rcp_approx")
        del normalized
        replay["weight_scale"] = replay["weight_scale"].reshape_as(packed["weight_scale"])
        if set(replay) != set(packed) or any(
            not torch.equal(value, replay[key]) for key, value in packed.items()
        ):
            raise ValueError("ModelOpt save/restore changed packed state: " + group.name)
        method = Method(replay)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        span = stats[group.name].clamp_min(1e-8)
        x = (
            (torch.randn((129, k), device="cuda", generator=generator) * (span / 3))
            .clamp(-span, span)
            .bfloat16()
        )
        actual = method.apply(None, x)
        repeated = method.apply(None, x)
        if not torch.isfinite(actual).all() or not torch.equal(actual, repeated):
            raise ValueError("Actual GEMM finite/repeat gate failed: " + group.name)
        path = output / f"layer-{ordinal:03d}.safetensors"
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in replay.items()}, str(path)
        )
        records.append(
            {
                "name": group.name,
                "modules": list(group.modules),
                "shape": list(group.shape),
                "file": path.name,
                "sha256": sha(path),
                "seed": seed,
                "modelopt_restore_exact": True,
                "trt_pack_exact": True,
                "gemm_repeat_exact": True,
            }
        )
        del (
            weight,
            factors,
            restored,
            packed,
            replay,
            method,
            actual,
            repeated,
            x,
            gold_data,
            gold_scales,
        )
    manifest = {
        "format": "vdn-modelopt-svdquant-nvfp4-v1",
        "binding": binding,
        "calibration_sha256": sha(statistics),
        "library_sha256": sha(library),
        "layers": records,
        "scope": (
            "208 joint logical groups, 312 inherited linears; VDN-added projections/gates excluded; "
            "shared quantized QKV feeds both softmax and linear-attention branches"
        ),
        "rank": 32,
        "alpha": 0.5,
        "pack_variant": "nested_rcp_approx",
    }
    write(output / "manifest.json", manifest)
    return manifest


def load_methods(artifact, binding, library):
    manifest = read(artifact / "manifest.json")
    if (
        manifest["format"] != "vdn-modelopt-svdquant-nvfp4-v1"
        or manifest["binding"] != binding
        or manifest["library_sha256"] != sha(library)
    ):
        raise ValueError("Packed artifact source/runtime binding differs")
    expected = groups()
    if len(manifest["layers"]) != len(expected):
        raise ValueError("Packed artifact coverage differs")
    torch.ops.load_library(str(library))
    methods = {}
    for entry, group in zip(manifest["layers"], expected, strict=True):
        if (entry["name"], entry["modules"], entry["shape"]) != (
            group.name,
            list(group.modules),
            list(group.shape),
        ):
            raise ValueError("Packed layer order/geometry differs")
        path = local_file(artifact, entry["file"])
        if path.parent.resolve() != artifact.resolve() or sha(path) != entry["sha256"]:
            raise ValueError("Packed layer bytes changed")
        tensors = load_file(str(path), device="cuda")
        validate_payload(tensors, group.shape)
        methods[group.name] = Method(tensors)
    return methods
