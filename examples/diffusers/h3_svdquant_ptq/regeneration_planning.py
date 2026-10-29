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

"""CPU-only representative-layer selection and explicit memory estimates."""

from artifact_binding import expected_shape
from recipe_comparison import layer_schedule

SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def representative_schedule(names, seed=1101):
    """Pick one of each actual projection type without resetting its RNG ordinal."""
    schedule = layer_schedule(names, seed)
    selected = []
    remaining = set(SUFFIXES)
    for item in schedule:
        suffix = next((suffix for suffix in remaining if item[1].endswith("." + suffix)), None)
        if suffix is not None:
            selected.append(item)
            remaining.remove(suffix)
    if remaining:
        raise ValueError(f"Missing representative H3 projection types: {sorted(remaining)}")
    return selected


def layer_storage(name, rank=32):
    """Estimate materialized tensors, not allocator/workspace peak memory."""
    n, k = expected_shape(name)
    return {
        "name": name,
        "shape": [n, k],
        "dense_bf16_bytes": 2 * n * k,
        "one_fp32_matrix_bytes": 4 * n * k,
        "lowrank_bf16_bytes": 2 * rank * (n + k),
        "packed_payload_approx_bytes": n * k // 2 + n * k // 16 + 2 * rank * (n + k) + 2 * k + 12,
        "no_unused_scale_padding": n % 128 == 0 and k % 64 == 0,
    }
