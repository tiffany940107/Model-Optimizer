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

"""Explicit experimental dispatch; the baseline kernel and policy stay untouched."""

from collections import Counter

import torch
import torch.nn.functional as F

__all__ = ["Dispatch", "RoutedMethod", "bf16_group", "relative_l2", "validate_payload"]


def validate_payload(tensors, shape):
    n, k = shape
    expected = {
        "weight",
        "weight_scale",
        "weight_global_scale",
        "input_scale",
        "alpha",
        "pre_quant_scale",
        "svdquant_lora_a",
        "svdquant_lora_b",
    }
    if set(tensors) != expected:
        raise ValueError("Unexpected packed schema")
    if tensors["weight"].shape != (n, k // 2) or tensors["weight"].dtype != torch.uint8:
        raise ValueError("NVFP4 weight packing differs")
    # TRT scale storage pads N to 128 and K/16 to 4 before swizzling.
    scale_size = ((n + 127) // 128 * 128) * ((k // 16 + 3) // 4 * 4)
    if (
        tensors["weight_scale"].numel() != scale_size
        or tensors["weight_scale"].dtype != torch.uint8
    ):
        raise ValueError("NVFP4 scale storage differs")
    for name, target in (
        ("pre_quant_scale", (k,)),
        ("svdquant_lora_a", (32, k)),
        ("svdquant_lora_b", (n, 32)),
    ):
        if tensors[name].shape != target or tensors[name].dtype != torch.bfloat16:
            raise ValueError("BF16 rank32 geometry differs: " + name)
        if not torch.isfinite(tensors[name]).all():
            raise ValueError("Nonfinite factors")
    for name in ("input_scale", "weight_global_scale", "alpha"):
        value = tensors[name]
        if (
            value.numel() != 1
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
            or not (value > 0).all()
        ):
            raise ValueError("Invalid positive FP32 scalar: " + name)
    if not torch.allclose(
        tensors["alpha"],
        1 / (tensors["input_scale"] * tensors["weight_global_scale"]),
        rtol=1e-6,
        atol=0,
    ):
        raise ValueError("Scale/alpha coupling differs")


def bf16_group(layers, value):
    # Preserve the three separate BF16 Q/K/V GEMMs, not a new concatenated GEMM.
    parts = [F.linear(value, layer.weight, layer.bias) for layer in layers]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


class Dispatch:
    def __init__(self, names, nfe=8):
        self.names, self.nfe = frozenset(names), nfe
        self.active = False
        self.step = None
        self.completed = 0
        self.poisoned = False

    def start(self, *, bf16_steps=(), bf16_groups=()):
        steps, names = frozenset(bf16_steps), frozenset(bf16_groups)
        if self.active or self.poisoned:
            raise RuntimeError("Active or poisoned dispatch")
        if (
            any(type(step) is not int or not 0 <= step < self.nfe for step in steps)
            or not names <= self.names
        ):
            raise ValueError("Invalid recovery selection")
        self.steps, self.recovered = steps, names
        self.completed, self.step = 0, None
        self.counts = {name: Counter() for name in self.names}
        self.active = True

    def before(self, _model, _args):
        if not self.active or self.step is not None or self.completed >= self.nfe:
            self.poisoned = True
            raise RuntimeError("Invalid/reentrant root forward")
        self.step = self.completed
        self.seen = set()

    def route(self, name):
        if not self.active or self.step is None or name not in self.names or name in self.seen:
            self.poisoned = True
            raise RuntimeError("Unexpected group invocation")
        self.seen.add(name)
        route = "bf16" if self.step in self.steps or name in self.recovered else "nvfp4"
        self.counts[name][route] += 1
        return route

    def after(self, _model, _args, output):
        if output is None or self.step is None or self.seen != self.names:
            self.poisoned = True
            raise RuntimeError("Incomplete or failed transformer forward")
        self.completed += 1
        self.step = None

    def finish(self):
        if self.poisoned or not self.active or self.step is not None or self.completed != self.nfe:
            raise RuntimeError("Incomplete generation")
        for name, counts in self.counts.items():
            expected = self.nfe if name in self.recovered else len(self.steps)
            if counts["bf16"] != expected or counts["nvfp4"] != self.nfe - expected:
                raise RuntimeError("Wrong union recovery call counts")
        self.active = False
        return {name: dict(counts) for name, counts in self.counts.items()}


class RoutedMethod:
    def __init__(self, name, layers, method, dispatch):
        self.name, self.layers, self.method, self.dispatch = name, layers, method, dispatch

    def apply(self, _layer, value):
        if self.dispatch.route(self.name) == "bf16":
            return bf16_group(self.layers, value)
        return self.method.apply(None, value)


def relative_l2(value, reference):
    delta = (value.float() - reference.float()).double().square().sum()
    norm = reference.double().square().sum()
    return float(torch.sqrt(delta / norm.clamp_min(1e-30)))
