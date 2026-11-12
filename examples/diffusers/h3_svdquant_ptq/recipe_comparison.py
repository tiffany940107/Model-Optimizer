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

"""Compare per-layer regeneration without retaining the complete H3 model."""

import importlib.util
import sys
from pathlib import Path

import torch
from artifact_binding import digest
from h3_modelopt_ptq import factorize_legacy

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

__all__ = [
    "compare_factorizations",
    "layer_schedule",
    "load_legacy_core",
    "pack_for_trt",
    "pack_modelopt_for_trt",
    "reorder_qkv",
    "swizzle_scales",
    "tensor_errors",
]


def load_legacy_core(root, expected_sha256):
    """Load only the reviewed local golden implementation after hash validation."""
    path = Path(root).resolve() / "svdquant_h3/core.py"
    if digest(path) != expected_sha256:
        raise ValueError("Legacy factorizer source changed")
    name = "h3_ptq_bound_legacy_core"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the verified legacy source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def layer_schedule(names, seed=1101, start=0, stop=None):
    """Match exporter seed+position in merged-stats insertion order, even for slices."""
    names = list(names)
    end = len(names) if stop is None else stop
    if not 0 <= start < end <= len(names) or len(names) != len(set(names)):
        raise ValueError("Invalid layer slice or duplicate names")
    return [(index, names[index], seed + index) for index in range(start, end)]


def reorder_qkv(weight):
    """Match released H3 checkpoint [head,qkv,dim] -> runtime [qkv,head,dim]."""
    grouped = weight.reshape(56, 3, 128, weight.shape[1])
    return torch.cat([grouped[:, index].reshape(-1, weight.shape[1]) for index in range(3)])


@torch.no_grad()
def compare_factorizations(weight, stats, legacy, *, seed, rank=32):
    """Exact same-device A/B of genuine legacy and new recipe implementations."""
    devices = [weight.device.index] if weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if weight.is_cuda:
            torch.cuda.default_generators[weight.device.index].manual_seed(seed)
        golden = legacy.factorize_svdquant(
            weight, stats, rank=rank, alpha=0.5, quantize_residual=False
        )
    candidate = factorize_legacy(weight, stats, seed=seed, rank=rank)
    equality = {
        name: torch.equal(getattr(golden, name), getattr(candidate, name))
        for name in ("pre_quant_scale", "residual", "lora_a", "lora_b")
    }
    return candidate, equality


@torch.no_grad()
def pack_for_trt(factors):
    """Use the actual legacy TRT static-6 packer, not a substitute CPU quantizer."""
    residual = factors.residual.bfloat16().contiguous()
    ws = (2688 / residual.float().abs().amax().clamp_min(1e-12)).reshape(1).contiguous()
    xs = (
        (
            2688
            / (factors.activation_absmax.float() * factors.pre_quant_scale.float())
            .amax()
            .clamp_min(1e-12)
        )
        .reshape(1)
        .contiguous()
    )
    packed, scales = torch.ops.trtllm.fp4_quantize(residual, ws, 16, False, True)
    return {
        "weight": packed.view(torch.uint8),
        "weight_scale": scales.view(torch.uint8),
        "input_scale": xs,
        "weight_global_scale": ws,
        "alpha": 1 / (xs * ws),
        "pre_quant_scale": factors.pre_quant_scale,
        "svdquant_lora_a": factors.lora_a,
        "svdquant_lora_b": factors.lora_b,
    }


def swizzle_scales(linear_scales, rows, cols):
    """Transform upstream ModelOpt linear E4M3 scales into TRT's 128x4-block layout."""
    row_tiles, col_tiles = (rows + 127) // 128, (cols + 63) // 64
    padded = torch.zeros(
        (row_tiles * 128, col_tiles * 4), dtype=torch.uint8, device=linear_scales.device
    )
    padded[:rows, : cols // 16] = linear_scales.view(torch.uint8).reshape(rows, cols // 16)
    return (
        padded.reshape(1, row_tiles, 4, 32, col_tiles, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .flatten()
    )


@torch.no_grad()
def pack_modelopt_for_trt(layer):
    """Actually quantize the residual with ModelOpt and adapt layout to old TRT GEMM.

    Native export uses ModelOpt dequantization scale = amax/2688, then converts
    to the runtime multiplier. Reciprocal-roundtrip and tie/clamp differences
    are measured, never silently replaced with the legacy packer.
    """
    q, scales, global_scale = NVFP4QTensor.quantize(
        layer.weight, 16, weights_scaling_factor_2=layer.weight_quantizer.amax.float() / 2688
    )
    n, k = layer.weight.shape
    ws = (1 / global_scale).reshape(1).contiguous()
    xs = (2688 / layer.input_quantizer.amax.float()).reshape(1).contiguous()
    tensors = {
        "weight": q._quantized_data.view(torch.uint8),
        "weight_scale": swizzle_scales(scales, n, k),
        "weight_global_scale": ws,
        "input_scale": xs,
        "alpha": 1 / (xs * ws),
        "pre_quant_scale": layer.input_quantizer.pre_quant_scale,
        "svdquant_lora_a": layer.weight_quantizer.svdquant_lora_a,
        "svdquant_lora_b": layer.weight_quantizer.svdquant_lora_b,
    }
    dequantized = q.dequantize(scale=scales, double_scale=global_scale, block_sizes={-1: 16})
    return tensors, dequantized


def tensor_errors(actual, reference):
    """Record finite-only numerical error; this does not define an acceptance threshold."""
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Nonfinite quantization comparison")
    diff = actual.float() - reference.float()
    return {
        "exact": torch.equal(actual, reference),
        "max_abs": diff.abs().max().item(),
        "relative_l2": (diff.norm() / reference.float().norm().clamp_min(1e-12)).item(),
    }
