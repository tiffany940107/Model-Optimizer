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

"""VDN-only mapping of the original PTQ scope; no blanket Linear conversion."""

from dataclasses import dataclass
from threading import Lock
from types import MethodType

import torch

__all__ = ["Group", "JointQKV", "groups", "install", "select"]


@dataclass(frozen=True)
class Group:
    name: str
    modules: tuple[str, ...]
    shape: tuple[int, int]


def groups():
    result = []
    for family, count in (("blocks", 50), ("token_refiner.blocks", 2)):
        for index in range(count):
            name = f"{family}.{index}"
            prefix = (
                f"transformer_blocks.{index}"
                if family == "blocks"
                else f"token_refiner.refiner_blocks.{index}"
            )
            attention = prefix + (".attn.orig" if family == "blocks" else ".attn")
            result.extend(
                [
                    Group(
                        name + ".attn.qkv_proj",
                        tuple(attention + ".to_" + part for part in "qkv"),
                        (21504, 5376),
                    ),
                    Group(name + ".attn.out_proj", (attention + ".to_out.0",), (5376, 7168)),
                    Group(name + ".mlp.fc1", (prefix + ".ff.net.0.proj",), (28672, 5376)),
                    Group(name + ".mlp.fc2", (prefix + ".ff.net.2",), (5376, 14336)),
                ]
            )
    return result


def select(model):
    """Require original geometry and bias-free inherited projections after LoRA merge."""
    selected = {}
    for group in groups():
        layers = [model.get_submodule(name) for name in group.modules]
        n, k = group.shape
        for layer in layers:
            if (
                not isinstance(layer, torch.nn.Linear)
                or tuple(layer.weight.shape) != (n // len(layers), k)
                or layer.weight.dtype != torch.bfloat16
                or layer.bias is not None
            ):
                raise ValueError("Unexpected VDN inherited projection: " + group.name)
        selected[group.name] = layers
    if len(selected) != 208 or sum(map(len, selected.values())) != 312:
        raise ValueError("Unexpected coverage")
    return selected


class JointQKV:
    """One NVFP4 GEMM for sequential Q/K/V calls on exactly the same tensor.

    Single-request eager inference only. This preserves a joint rank32 residual
    decomposition instead of accidentally giving each projection its own rank32.
    A stale/reordered/reentrant call raises; no silent BF16 fallback is allowed.
    """

    def __init__(self, apply):
        self.apply = apply
        self.value: torch.Tensor | None = None
        self.parts: tuple[torch.Tensor, ...] | None = None
        self.next = 0
        self.version = None

    def __call__(self, index: int, value: torch.Tensor):
        if index != self.next:
            raise RuntimeError("QKV must be called once in Q,K,V order")
        if index == 0:
            result = self.apply(value)
            if result.shape[-1] % 3:
                raise ValueError("QKV output width must be divisible by 3")
            self.value = value
            self.version = value._version
            self.parts = result.chunk(3, dim=-1)
        elif value._version != self.version or value is not self.value:
            raise RuntimeError("QKV projections did not receive the same unchanged tensor")
        if self.parts is None:
            raise RuntimeError("Missing QKV result")
        result = self.parts[index]
        self.next += 1
        if self.next == 3:
            self.value, self.parts, self.version = None, None, None
            self.next = 0
        return result


def install(model, methods):
    """Replace only selected Linear forwards; keep source parameters for audit.

    Caller validates every packed payload and content binding before entering.
    Do not call model.to(), torch.compile, offload or training after installation.
    This integration is for accuracy tests, not compressed-residency claims.
    """
    selected = select(model)
    if set(methods) != set(selected):
        raise ValueError("Packed method coverage differs")
    pending, joints = [], []
    for name, layers in selected.items():
        method = methods[name]
        if len(layers) == 3:
            joint = JointQKV(lambda value, method=method: method.apply(None, value))
            joints.append(joint)
            for index, layer in enumerate(layers):

                def qkv_forward(_layer, value, index=index, joint=joint):
                    return joint(index, value)

                pending.append((layer, MethodType(qkv_forward, layer)))
        else:
            layer = layers[0]

            def single_forward(_layer, value, method=method):
                return method.apply(None, value)

            pending.append((layer, MethodType(single_forward, layer)))
    if any("forward" in layer.__dict__ for layer, _ in pending):
        raise ValueError("Refuse to overwrite another forward adapter")
    if "forward" in model.__dict__:
        raise ValueError("Refuse to overwrite another root forward adapter")
    lock = Lock()
    failed = False
    original_forward = model.forward

    def guarded_forward(_model, *args, **kwargs):
        nonlocal failed
        if not lock.acquire(blocking=False):
            raise RuntimeError("VDN PTQ adapter only supports serial eager inference")
        try:
            if failed or any(joint.next != 0 for joint in joints):
                raise RuntimeError(
                    "PTQ adapter is poisoned or has incomplete QKV state; reload model"
                )
            result = original_forward(*args, **kwargs)
            if any(joint.next != 0 for joint in joints):
                raise RuntimeError("Incomplete QKV sequence at end of model forward")
            return result
        except BaseException:
            failed = True
            raise
        finally:
            lock.release()

    for layer, forward in pending:
        layer.forward = forward
    model.forward = MethodType(guarded_forward, model)
    return joints
