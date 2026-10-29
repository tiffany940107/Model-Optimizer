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

"""SGLang linear-method protocol for the separately built TRT NVFP4 kernels."""

import sys

import torch
from portable_io import digest

__all__ = ["PackedLinear", "load_runtime"]


def load_runtime(library, expected_sha256):
    """Check the reviewed ABI/device and library identity before loading code."""
    if (
        sys.version_info[:2] != (3, 12)
        or torch.__version__ != "2.11.0+cu130"
        or torch.version.cuda != "13.0"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 0)
    ):
        raise RuntimeError("Requires Python3.12 / Torch2.11.0+cu130 / CUDA13.0 / B200 SM100")
    if digest(library) != expected_sha256:
        raise ValueError("TRT library SHA256 mismatch")
    torch.ops.load_library(str(library))
    for name in ("fp4_quantize", "nvfp4_quantize_smooth", "nvfp4_svdquant_gemm"):
        if not hasattr(torch.ops.trtllm, name):
            raise RuntimeError(f"Missing custom torch operator: {name}")


class PackedLinear:
    """Same smoothing/rank-down/fused-rank-up operation order as the PTQ recipe.

    No global mutable workspace is shared. Full TP1 weights only; callers must
    separately validate payload shapes and the exact source/checkpoint binding.
    """

    def __init__(self, tensors):
        self.weight = tensors["weight"].contiguous()
        self.scales = tensors["weight_scale"].contiguous()
        self.input_scale = tensors["input_scale"].float().reshape(1).contiguous()
        self.alpha = tensors["alpha"].float().reshape(1).contiguous()
        self.smoothing = tensors["pre_quant_scale"].bfloat16().contiguous()
        self.down = (self.smoothing.unsqueeze(1) * tensors["svdquant_lora_a"].t()).contiguous()
        self.up = (tensors["svdquant_lora_b"].float() / self.alpha).bfloat16().contiguous()

    def apply(self, layer, x, bias=None):
        """Implement the SGLang quant_method.apply interface, preserving bias."""
        del layer
        shape = x.shape
        value = x.reshape(-1, shape[-1]).bfloat16().contiguous()
        packed, scales = torch.ops.trtllm.nvfp4_quantize_smooth(
            value, self.smoothing, self.input_scale
        )
        lowrank = torch.mm(value, self.down)
        result = torch.ops.trtllm.nvfp4_svdquant_gemm(
            packed.view(torch.uint8),
            self.weight,
            scales.view(torch.uint8),
            self.scales,
            self.alpha,
            lowrank,
            self.up,
            torch.bfloat16,
            bias,
            -1,
        )
        return result.reshape(*shape[:-1], result.shape[-1])
