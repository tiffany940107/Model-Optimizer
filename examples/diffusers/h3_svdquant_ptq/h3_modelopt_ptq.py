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

"""ModelOpt representation of the legacy H3 SVDQuant PTQ recipe.

Import this plugin before calling ``modelopt.torch.opt.restore``. CPU helpers
exercise ModelOpt's real NVFP4 packing code, not its CUDA TensorQuantizer kernel.
No training or backward pass is used.
"""

import re
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import modelopt.torch.opt as mto
from modelopt.torch.opt.mode import ModeDescriptor
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.quantization.conversion import (
    convert_to_quantized_model,
    create_and_replace_svdquant_linear_on_the_fly,
    restore_quantizer_state,
    update_quantize_metadata,
)
from modelopt.torch.quantization.mode import QuantizeModeRegistry
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

__all__ = [
    "FrozenFactors",
    "export_frozen",
    "factorize_legacy",
    "h3_layer_names",
    "import_frozen",
    "modelopt_cpu_reference",
    "pack_modelopt_cpu",
]

_MODE = "h3_svdquant_legacy_v1"
_PACKED_KEYS = ("weight", "weight_scale", "weight_global_scale", "input_scale", "alpha")


def h3_layer_names() -> tuple[str, ...]:
    """Exact original H3 selector: 50 main + 2 refiner blocks, four projections."""
    suffixes = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    return tuple(
        f"{prefix}.{index}.{suffix}"
        for prefix, count in (("blocks", 50), ("token_refiner.blocks", 2))
        for index in range(count)
        for suffix in suffixes
    )


@dataclass(frozen=True)
class FrozenFactors:
    """Pre-pack residual and BF16 factors; packed payload, if supplied, is opaque.

    ``packed`` contains the legacy TRT exporter fields without transposition,
    swizzle conversion, recomputation, or repacking. It is an identity-replay
    artifact, not evidence that ModelOpt's own packer is equivalent.
    """

    pre_quant_scale: torch.Tensor
    residual: torch.Tensor
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    activation_absmax: torch.Tensor
    packed: dict[str, torch.Tensor] | None = None

    def to(self, device):
        """Move tensors without changing dtypes or packed layout."""
        return FrozenFactors(
            self.pre_quant_scale.to(device),
            self.residual.to(device),
            self.lora_a.to(device),
            self.lora_b.to(device),
            self.activation_absmax.to(device),
            {key: value.to(device) for key, value in self.packed.items()}
            if self.packed is not None
            else None,
        )

    def validate(self, shape):
        """Reject unsupported shapes, nonfinite data, and ambiguous scale types."""
        n, k = shape
        r = self.lora_a.shape[0]
        expected = ((k,), (n, k), (r, k), (n, r), (k,))
        tensors = (
            self.pre_quant_scale,
            self.residual,
            self.lora_a,
            self.lora_b,
            self.activation_absmax,
        )
        if k % 16 or not 0 < r <= min(n, k):
            raise ValueError("Requires block-16 input width and valid positive rank")
        for value, required in zip(tensors, expected, strict=True):
            if tuple(value.shape) != required or not torch.isfinite(value).all():
                raise ValueError("Invalid factor shape or nonfinite values")
        if any(t.dtype != torch.bfloat16 for t in tensors[:4]):
            raise ValueError("Legacy residual, smoothing and low-rank factors must be BF16")
        if self.activation_absmax.dtype != torch.float32:
            raise ValueError("Calibration maxima must be FP32")
        if (self.pre_quant_scale <= 0).any() or (self.activation_absmax < 0).any():
            raise ValueError("Smoothing must be positive; absmax must be nonnegative")
        if self.packed is not None:
            if set(self.packed) != set(_PACKED_KEYS):
                raise ValueError("Incomplete legacy static_6 packed payload")
            if self.packed["weight"].shape != (n, k // 2):
                raise ValueError("Invalid packed weight shape")
            if any(self.packed[key].dtype != torch.uint8 for key in ("weight", "weight_scale")):
                raise ValueError("Packed weight and swizzled scales must be raw uint8")
            for key in ("weight_global_scale", "input_scale", "alpha"):
                value = self.packed[key]
                if (
                    value.dtype != torch.float32
                    or value.numel() != 1
                    or not torch.isfinite(value).all()
                    or (value <= 0).any()
                ):
                    raise ValueError("Packed scalar scales must be positive finite FP32")


@torch.no_grad()
def factorize_legacy(weight, activation_absmax, *, seed, rank=32):
    """Generate the legacy FP32 randomized-SVD factors without modifying RNG state.

    Same-device / same-library replay is testable; a seed does not promise
    bitwise SVD identity across CPU and GPU or across PyTorch versions.
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Expected two-dimensional BF16 original weight")
    if activation_absmax.shape != (weight.shape[1],) or not 0 < rank <= min(weight.shape):
        raise ValueError("Invalid rank or calibration width")
    if (
        not torch.isfinite(weight).all()
        or not torch.isfinite(activation_absmax).all()
        or (activation_absmax < 0).any()
    ):
        raise ValueError("Invalid calibration or weight values")
    x_span = activation_absmax.float().clamp_min(1e-8)
    w_span = weight.float().abs().amax(dim=0).clamp_min(1e-8)
    scale = (w_span.pow(0.5) / x_span.pow(0.5)).clamp(1 / 16, 16)
    smoothed = weight.float() / scale.unsqueeze(0)
    devices = [weight.device.index] if weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if weight.is_cuda:
            torch.cuda.default_generators[weight.device.index].manual_seed(seed)
        u, singular, v = torch.svd_lowrank(smoothed, q=min(rank + 8, min(weight.shape)), niter=2)
    b, a = u[:, :rank] * singular[:rank].unsqueeze(0), v[:, :rank].t()
    return FrozenFactors(
        scale.bfloat16(),
        (smoothed - b @ a).bfloat16(),
        a.bfloat16(),
        b.bfloat16(),
        activation_absmax.float().clone(),
    )


def _convert(model, config):
    model, metadata = convert_to_quantized_model(model, config)
    create_and_replace_svdquant_linear_on_the_fly(model)
    return model, metadata


def _restore(model, config, metadata):
    model, _ = _convert(model, config)
    restore_quantizer_state(model, config, metadata)
    model.requires_grad_(False)
    return model


@QuantizeModeRegistry.register_mode
class H3LegacyMode(ModeDescriptor):
    """Versioned ModelOpt conversion/restoration mode; no AWQ or calibration rerun."""

    name = property(lambda self: _MODE)
    config_class = property(lambda self: QuantizeConfig)
    convert = property(lambda self: _convert)
    restore = property(lambda self: _restore)
    update_for_save = property(lambda self: update_quantize_metadata)
    update_for_new_mode = property(lambda self: update_quantize_metadata)


@torch.no_grad()
def import_frozen(model: nn.Module, factors: dict[str, FrozenFactors], *, require_full_h3=True):
    """Convert to ModelOpt and import exact tensors; unselected quantizers stay off.

    Full H3 is the default gate. Small fixtures must explicitly opt out. Input
    modules must be plain nn.Linear adapters, not opaque SGLang linear methods.
    """
    if require_full_h3 and set(factors) != set(h3_layer_names()):
        raise ValueError("Expected exactly 208 original H3 projection factors")
    if not factors:
        raise ValueError("No factors supplied")
    for name, factor in factors.items():
        if not re.fullmatch(r"[A-Za-z0-9_.]+", name):
            raise ValueError("Unsupported module name")
        layer = model.get_submodule(name)
        if type(layer) is not nn.Linear or layer.weight.dtype != torch.bfloat16:
            raise ValueError("Expected plain BF16 nn.Linear adapter")
        factor.validate(layer.weight.shape)
    attrs = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    cfg = [{"quantizer_name": "*", "enable": False}]
    for name in factors:
        cfg.extend(
            {"quantizer_name": f"{name}.{slot}", "cfg": attrs}
            for slot in ("input_quantizer", "weight_quantizer")
        )
    model = mto.apply_mode(model, [(_MODE, {"quant_cfg": cfg, "algorithm": None})])
    for name, factor in factors.items():
        layer = model.get_submodule(name)
        factor = factor.to(layer.weight.device)
        layer.weight.copy_(factor.residual)
        layer.input_quantizer.pre_quant_scale = factor.pre_quant_scale
        layer.weight_quantizer.svdquant_lora_a = factor.lora_a
        layer.weight_quantizer.svdquant_lora_b = factor.lora_b
        layer.input_quantizer.amax = (
            (factor.activation_absmax * factor.pre_quant_scale.float()).amax().clamp_min(1e-12)
        )
        layer.weight_quantizer.amax = factor.residual.float().abs().amax().clamp_min(1e-12)
        layer.input_quantizer.register_buffer(
            "_h3_activation_absmax", factor.activation_absmax.clone()
        )
        if factor.packed is not None:
            # Legacy tensors store quantization multipliers, whereas ModelOpt
            # stores amax. Keep original multipliers too: reciprocal round trips
            # are not a byte-preserving export operation.
            layer.input_quantizer.amax = (2688 / factor.packed["input_scale"]).reshape(())
            layer.weight_quantizer.amax = (2688 / factor.packed["weight_global_scale"]).reshape(())
            for key, value in factor.packed.items():
                layer.weight_quantizer.register_buffer(f"_h3_legacy_{key}", value.clone())
    model.requires_grad_(False)
    return model.eval()


def export_frozen(layer: nn.Module) -> FrozenFactors:
    """Export the imported tensor representation without repacking legacy bytes."""
    wq = layer.weight_quantizer
    packed = None
    if hasattr(wq, "_h3_legacy_weight"):
        packed = {key: getattr(wq, f"_h3_legacy_{key}").detach().clone() for key in _PACKED_KEYS}
    return FrozenFactors(
        layer.input_quantizer.pre_quant_scale.detach().clone(),
        layer.weight.detach().clone(),
        wq.svdquant_lora_a.detach().clone(),
        wq.svdquant_lora_b.detach().clone(),
        layer.input_quantizer._h3_activation_absmax.detach().clone(),
        packed,
    )


@torch.no_grad()
def pack_modelopt_cpu(tensor: torch.Tensor, global_amax: torch.Tensor):
    """Run upstream NVFP4QTensor packing with a frozen global scale on CPU.

    This uses linear (not TRT-swizzled) scales and ModelOpt's ties-to-even
    handling. Never pass its scale layout directly to the old TRT GEMM.
    """
    if tensor.device.type != "cpu":
        raise ValueError("CPU reference only; use native TensorQuantizer on GPU")
    return NVFP4QTensor.quantize(
        tensor, 16, weights_scaling_factor_2=global_amax.float().clamp_min(1e-12) / 2688
    )


@torch.no_grad()
def modelopt_cpu_reference(layer: nn.Module, x: torch.Tensor):
    """Evaluate ModelOpt packing + BF16 GEMMs; not a real NVFP4 GEMM benchmark."""

    def qdq(value, amax):
        packed, scale, global_scale = pack_modelopt_cpu(value, amax)
        return packed.dequantize(scale=scale, double_scale=global_scale, block_sizes={-1: 16})

    xs = x * layer.input_quantizer.pre_quant_scale
    residual = F.linear(
        qdq(xs, layer.input_quantizer.amax),
        qdq(layer.weight, layer.weight_quantizer.amax),
        layer.bias,
    )
    lowrank = F.linear(
        F.linear(xs, layer.weight_quantizer.svdquant_lora_a), layer.weight_quantizer.svdquant_lora_b
    )
    return residual + lowrank
