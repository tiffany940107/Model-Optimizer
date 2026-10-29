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

"""Opt-in ModelOpt NVFP4 pack arithmetic experiments, with a frozen multiplier.

This implementation never calls the TRT packer. It uses ModelOpt's QTensor and
E2M1 coding with explicit PyTorch arithmetic and an optional PTX reciprocal.
It is not accepted as compatible until exact GPU data/scale gates pass.
"""

import torch
from recipe_comparison import swizzle_scales

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

__all__ = ["VARIANTS", "MultiplierNVFP4QTensor", "pack_compatible_layer"]

VARIANTS = ("multiplier_div", "multiplier_rcp_rn", "nested_rcp_rn", "nested_rcp_approx")


def _reciprocal(value, approximate):
    if approximate:
        if not value.is_cuda:
            raise ValueError("Approximate PTX reciprocal has no CPU fallback")
        # Triton is optional for CPU reference tests; only the CUDA experiment
        # imports it. A CPU RN reciprocal is never mislabeled PTX approximate.
        from compat_rcp_triton import reciprocal_approximate

        return reciprocal_approximate(value)
    return torch.reciprocal(value)


class MultiplierNVFP4QTensor(NVFP4QTensor):
    """ModelOpt packing extension retaining the original quantization multiplier."""

    @classmethod
    @torch.no_grad()
    def quantize(cls, tensor, global_multiplier, *, variant="nested_rcp_approx"):
        """Pack BF16 block-16 values with an explicit, auditable arithmetic variant."""
        if (
            variant not in VARIANTS
            or tensor.dtype != torch.bfloat16
            or tensor.ndim != 2
            or tensor.shape[-1] % 16
        ):
            raise ValueError("Expected a supported variant and BF16 2D block-16 input")
        multiplier = global_multiplier.to(device=tensor.device, dtype=torch.float32).reshape(())
        if not torch.isfinite(tensor).all() or not torch.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Expected finite input and positive global multiplier")
        blocks = tensor.float().reshape(tensor.shape[0], -1, 16)
        maxima = blocks.abs().amax(-1)
        approximate = variant == "nested_rcp_approx"
        inv_six = _reciprocal(
            torch.tensor(6.0, device=tensor.device, dtype=torch.float32), approximate
        )
        scale_f32 = multiplier * (maxima * inv_six)
        scales = scale_f32.clamp(max=448).to(torch.float8_e4m3fn)
        scale_values = scales.float()
        # Zero blocks are legal and keep a zero scale. Unsupported nonzero
        # underflow is rejected explicitly rather than silently clamping it.
        if ((maxima > 0) & (scale_values == 0)).any():
            raise ValueError(
                "Nonzero scale underflow requires a separately validated saturation path"
            )
        safe_scales = torch.where(maxima > 0, scale_values, torch.ones_like(scale_values))
        if variant == "multiplier_div":
            normalized = (blocks * multiplier) / safe_scales.unsqueeze(-1)
        elif variant == "multiplier_rcp_rn":
            normalized = blocks * (multiplier * torch.reciprocal(safe_scales)).unsqueeze(-1)
        else:
            inverse_multiplier = _reciprocal(multiplier, approximate)
            encode_scale = _reciprocal(safe_scales * inverse_multiplier, approximate)
            normalized = blocks * encode_scale.unsqueeze(-1)
        normalized = torch.where(
            (maxima > 0).unsqueeze(-1), normalized, torch.zeros_like(normalized).copysign(blocks)
        )
        normalized = normalized.reshape_as(tensor)
        negative = torch.signbit(normalized).to(torch.uint8)
        # Upstream coding uses ties-to-even. Preserve IEEE negative zero too,
        # since CUDA cvt.e2m1 and ModelOpt's '<0' sign extraction can differ there.
        codes = cls._cast_fp4(normalized.clone())
        codes = (codes & 7) | (negative << 3)
        packed = (codes[:, 1::2] << 4) | codes[:, 0::2]
        result = cls(tensor.shape, tensor.dtype, packed)
        return result, scales, (1 / multiplier).reshape(1), normalized


@torch.no_grad()
def pack_compatible_layer(layer, *, variant="nested_rcp_approx"):
    """Run opt-in ModelOpt packing and preserve the multiplier in ModelOpt state.

    Existing frozen legacy multipliers take precedence. Newly regenerated
    factors derive it once from the actual BF16 residual, like the old exporter.
    This is a pack/export path, not an override of native TensorQuantizer.forward.
    """
    quantizer = layer.weight_quantizer
    if not hasattr(quantizer, "_h3_compat_multiplier"):
        multiplier = getattr(quantizer, "_h3_legacy_weight_global_scale", None)
        if multiplier is None:
            multiplier = (2688 / layer.weight.float().abs().amax().clamp_min(1e-12)).reshape(1)
        quantizer.register_buffer("_h3_compat_multiplier", multiplier.detach().clone())
    multiplier = quantizer._h3_compat_multiplier
    q, scales, _, normalized = MultiplierNVFP4QTensor.quantize(
        layer.weight, multiplier, variant=variant
    )
    n, k = layer.weight.shape
    input_scale = getattr(quantizer, "_h3_legacy_input_scale", None)
    if input_scale is None:
        input_scale = (2688 / layer.input_quantizer.amax).reshape(1)
    tensors = {
        "weight": q._quantized_data,
        "weight_scale": swizzle_scales(scales, n, k),
        "weight_global_scale": multiplier,
        "input_scale": input_scale,
        "alpha": 1 / (input_scale * multiplier),
        "pre_quant_scale": layer.input_quantizer.pre_quant_scale,
        "svdquant_lora_a": quantizer.svdquant_lora_a,
        "svdquant_lora_b": quantizer.svdquant_lora_b,
    }
    return tensors, normalized
