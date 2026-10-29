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

"""ModelOpt-owned packed-only checkpoints for streaming legacy H3 replay.

The original exporter does not retain the unquantized residual. This mode never
invents one, nor stores a full BF16 residual. Its only execution backend is the
legacy TRT packed kernel after export; native ModelOpt forward is disallowed.
"""

import torch
from h3_modelopt_ptq import H3LegacyMode, _convert
from torch import nn

import modelopt.torch.opt as mto
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.quantization.conversion import _replace_quant_module, restore_quantizer_state
from modelopt.torch.quantization.mode import QuantizeModeRegistry
from modelopt.torch.quantization.nn.modules.quant_linear import SVDQuantLinear

__all__ = ["export_packed", "import_packed", "make_empty_layer"]

_REGISTRY = _DMRegistryCls("H3PackedReplay")
_MODE = "h3_svdquant_packed_replay_v1"
_KEYS = ("weight", "weight_scale", "weight_global_scale", "input_scale", "alpha")


class PackedReplayLinear(SVDQuantLinear):
    """A packed-only ModelOpt module with explicit fail-closed native forward."""

    def _setup(self):
        super()._setup()
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, *_args, **_kwargs):
        raise RuntimeError(
            "Packed-only replay has no native ModelOpt forward; export to the original TRT GEMM"
        )


def _convert_packed(model, config):
    model, metadata = _convert(model, config)
    for module in model.modules():
        if isinstance(module, SVDQuantLinear) and type(module) not in _REGISTRY:
            _REGISTRY.register({type(module): module.__class__.__name__})(PackedReplayLinear)
    _replace_quant_module(model, registry=_REGISTRY)
    return model, metadata


def _restore_packed(model, config, metadata):
    model, _ = _convert_packed(model, config)
    restore_quantizer_state(model, config, metadata)
    return model.requires_grad_(False)


@QuantizeModeRegistry.register_mode
class H3PackedMode(H3LegacyMode):
    """Separate mode ensures a packed-only checkpoint cannot masquerade as PTQ regeneration."""

    name = property(lambda self: _MODE)
    convert = property(lambda self: _convert_packed)
    restore = property(lambda self: _restore_packed)


def make_empty_layer(n: int, k: int):
    """Create only meta shape information; no dense original weight allocation."""
    return nn.Sequential(nn.Linear(k, n, bias=False, device="meta", dtype=torch.bfloat16))


@torch.no_grad()
def import_packed(tensors, activation_absmax):
    """Import an already validated payload into ModelOpt, without dense decoding."""
    n, half_k = tensors["weight"].shape
    k = half_k * 2
    cfg = [{"quantizer_name": "*", "enable": False}]
    attrs = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
    cfg.extend(
        {"quantizer_name": f"0.{slot}", "cfg": attrs}
        for slot in ("input_quantizer", "weight_quantizer")
    )
    model = mto.apply_mode(make_empty_layer(n, k), [(_MODE, {"quant_cfg": cfg, "algorithm": None})])
    layer = model[0]
    layer.input_quantizer.pre_quant_scale = tensors["pre_quant_scale"].cpu()
    layer.input_quantizer.amax = (2688 / tensors["input_scale"].cpu()).reshape(())
    layer.weight_quantizer.amax = (2688 / tensors["weight_global_scale"].cpu()).reshape(())
    layer.weight_quantizer.svdquant_lora_a = tensors["svdquant_lora_a"].cpu()
    layer.weight_quantizer.svdquant_lora_b = tensors["svdquant_lora_b"].cpu()
    layer.input_quantizer.register_buffer("_h3_activation_absmax", activation_absmax.cpu().clone())
    for key in _KEYS:
        layer.weight_quantizer.register_buffer(f"_h3_legacy_{key}", tensors[key].cpu().clone())
    return model.requires_grad_(False).eval()


def export_packed(model):
    """Return exact original packed tensors from a restored ModelOpt checkpoint."""
    layer = model[0]
    if not isinstance(layer, PackedReplayLinear) or layer.weight.numel() != 0:
        raise ValueError("Expected packed-only H3 ModelOpt mode")
    q = layer.weight_quantizer
    return {
        **{key: getattr(q, f"_h3_legacy_{key}").detach().cpu().contiguous() for key in _KEYS},
        "pre_quant_scale": layer.input_quantizer.pre_quant_scale.detach().cpu().contiguous(),
        "svdquant_lora_a": q.svdquant_lora_a.detach().cpu().contiguous(),
        "svdquant_lora_b": q.svdquant_lora_b.detach().cpu().contiguous(),
    }
