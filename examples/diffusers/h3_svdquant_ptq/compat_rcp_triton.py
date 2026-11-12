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

"""Small PTX reciprocal primitive; not a call to the legacy TRT packer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rcp_kernel(source, target, count: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    value = tl.load(source + index, index < count, other=1.0)
    inverse = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(target + index, inverse, index < count)


def reciprocal_approximate(tensor):
    """Execute exactly the NVIDIA PTX rcp.approx.ftz.f32 instruction on CUDA FP32."""
    if not tensor.is_cuda or tensor.dtype != torch.float32:
        raise ValueError("PTX reciprocal requires a CUDA FP32 tensor")
    source = tensor.contiguous()
    output = torch.empty_like(source)
    _rcp_kernel[(triton.cdiv(source.numel(), 256),)](source, output, source.numel(), block=256)
    return output
