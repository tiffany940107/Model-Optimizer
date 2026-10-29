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

"""Separate quantization, smoothing, low-rank and fused-GEMM numerical effects."""

import torch
from recipe_comparison import swizzle_scales, tensor_errors
from torch.nn import functional as F

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

__all__ = ["byte_comparison", "layer_breakdown"]


def byte_comparison(actual, reference):
    """Compare raw packed bytes independently of numerical output norms."""
    a = actual.contiguous().view(torch.uint8).flatten()
    b = reference.contiguous().view(torch.uint8).flatten()
    if a.shape != b.shape:
        return {"exact": False, "shape_mismatch": [list(actual.shape), list(reference.shape)]}
    differing = (a != b).sum().item()
    return {"exact": differing == 0, "different_bytes": differing, "total_bytes": a.numel()}


@torch.no_grad()
def layer_breakdown(
    layer, legacy, golden, native, x, *, old_output, native_output, modelopt_output
):
    """Execute real quantizers and branch-isolated TRT kernels for diagnosis.

    Zero only residual packed data or only low-rank B to isolate actual fused
    branches. Their separately rounded sum need not equal one fused epilogue.
    """
    s = layer.input_quantizer.pre_quant_scale
    xs = x * s
    amax = layer.input_quantizer.amax.float()
    q, scales, global_scale = NVFP4QTensor.quantize(xs, 16, weights_scaling_factor_2=amax / 2688)
    mo_scales = swizzle_scales(scales, x.shape[0], x.shape[1])
    trt_x, trt_scales = torch.ops.trtllm.nvfp4_quantize_smooth(x, s, golden["input_scale"])
    bf16_x, bf16_scales = torch.ops.trtllm.fp4_quantize(
        xs.contiguous(), golden["input_scale"], 16, False, True
    )
    with layer.input_quantizer.disable_pre_quant_scale():
        native_activation = layer.input_quantizer(xs)
    packed_dequant = q.dequantize(scale=scales, double_scale=global_scale, block_sizes={-1: 16})
    trt_dequant = legacy.dequantize_nvfp4(
        trt_x.view(torch.uint8), trt_scales.view(torch.uint8), golden["input_scale"]
    )
    activation = {
        "modelopt_pack_vs_trt_fused_smooth_data": byte_comparison(q._quantized_data, trt_x),
        "modelopt_pack_vs_trt_fused_smooth_scales": byte_comparison(mo_scales, trt_scales),
        "modelopt_pack_vs_trt_bf16_smoothed_data": byte_comparison(q._quantized_data, bf16_x),
        "modelopt_pack_vs_trt_bf16_smoothed_scales": byte_comparison(mo_scales, bf16_scales),
        "bf16_smoothed_trt_vs_fused_smooth_data": byte_comparison(bf16_x, trt_x),
        "bf16_smoothed_trt_vs_fused_smooth_scales": byte_comparison(bf16_scales, trt_scales),
        "native_tensor_quantizer_vs_native_pack": tensor_errors(native_activation, packed_dequant),
        "native_pack_dequant_vs_fused_trt": tensor_errors(packed_dequant, trt_dequant),
        "modelopt_vs_trt_logical_scales": byte_comparison(
            legacy._swizzled_e4m3_to_linear(mo_scales, x.shape[0], x.shape[1]),
            legacy._swizzled_e4m3_to_linear(trt_scales, x.shape[0], x.shape[1]),
        ),
        "scale_padding_note": "Raw swizzled byte counts include unused padded slots; logical scales exclude padding",
    }
    a, b = golden["svdquant_lora_a"], golden["svdquant_lora_b"]
    explicit_lowrank = F.linear(F.linear(xs, a), b)
    down = torch.mm(x, (s.unsqueeze(1) * a.t()).contiguous())
    premultiplied_lowrank = F.linear(down, b)
    lowrank_only = legacy.TRTNVFP4SVDQuantLinearMethod(
        {**golden, "weight": torch.zeros_like(golden["weight"])}
    )
    fused_lowrank = lowrank_only.apply(None, x)
    residual_only = legacy.TRTNVFP4SVDQuantLinearMethod(
        {**golden, "svdquant_lora_b": torch.zeros_like(b)}
    )
    fused_residual = residual_only.apply(None, x)
    gold_weight_dequant = legacy.dequantize_nvfp4(
        golden["weight"], golden["weight_scale"], golden["weight_global_scale"]
    )
    reference_residual = F.linear(trt_dequant, gold_weight_dequant)
    return {
        "weight_data": byte_comparison(native["weight"], golden["weight"]),
        "weight_scales_swizzled": byte_comparison(native["weight_scale"], golden["weight_scale"]),
        "activation": activation,
        "lowrank": {
            "bf16_smoothing_reassociation": tensor_errors(premultiplied_lowrank, explicit_lowrank),
            "fused_alpha_folded_vs_explicit": tensor_errors(fused_lowrank, explicit_lowrank),
            "fused_alpha_folded_vs_premultiplied": tensor_errors(
                fused_lowrank, premultiplied_lowrank
            ),
        },
        "residual_gemm": tensor_errors(fused_residual, reference_residual),
        "fused_final_addition": tensor_errors(old_output, fused_residual + fused_lowrank),
        "full_output": {
            "native_export_same_trt_vs_old": tensor_errors(native_output, old_output),
            "native_modelopt_vs_old_trt": tensor_errors(modelopt_output, old_output),
            "native_modelopt_vs_native_export_trt": tensor_errors(modelopt_output, native_output),
        },
        "precision_flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "scope": "actual quantizers and isolated TRT branches; diagnostics, not a quality acceptance threshold",
    }
