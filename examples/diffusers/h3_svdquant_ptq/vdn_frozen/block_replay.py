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

"""Offline block-reconstruction primitives; not a new deployed kernel.

The GPU driver must supply a BF16 block's complete inputs, live hybrid layout,
and modality row indices. Slicing sequence rows before attention is forbidden:
it would change both softmax and linear-attention semantics.
"""

from contextlib import contextmanager
from types import MethodType

import torch

from .policy import JointQKV

__all__ = ["generated_video_rows", "replay", "temporary_methods", "tensor_versions"]


def generated_video_rows(latent_frame_function, frames=124, height=1344, width=768):
    """VDN/H3 VAE chunk geometry and (1,2,2) patchification, no implicit defaults."""
    if min(height, width) <= 0 or height % 32 or width % 32:
        raise ValueError("Canvas must align to spatial VAE16 times patch2")
    return latent_frame_function(frames, 17, 5) * (height // 32) * (width // 32)


@contextmanager
def temporary_methods(selected, methods):
    """Override only this block's logical groups, restoring even on exceptions.

    For a dedicated single-threaded offline calibration process only. This is
    not compatible with an already installed runtime forward adapter.
    """
    if set(selected) != set(methods) or len(selected) != 4:
        raise ValueError("A block requires exactly its four complete logical groups")
    layers = [layer for group in selected.values() for layer in group]
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Overlapping block groups")
    if any("forward" in layer.__dict__ for layer in layers):
        raise ValueError("Refuse to replace an existing adapter")
    patched, joints = [], []
    try:
        for name, group in selected.items():
            method = methods[name]
            if len(group) == 3:
                joint = JointQKV(lambda value, method=method: method.apply(None, value))
                joints.append(joint)
                for index, layer in enumerate(group):

                    def qkv_forward(_layer, value, joint=joint, index=index):
                        return joint(index, value)

                    layer.forward = MethodType(qkv_forward, layer)
                    patched.append(layer)
            elif len(group) == 1:
                layer = group[0]

                def single_forward(_layer, value, method=method):
                    return method.apply(None, value)

                layer.forward = MethodType(single_forward, layer)
                patched.append(layer)
            else:
                raise ValueError("Only complete QKV or single Linear groups are supported")
        yield
        if any(joint.next != 0 for joint in joints):
            raise RuntimeError("Incomplete block QKV replay")
    finally:
        for layer in patched:
            del layer.forward


def tensor_versions(value):
    if isinstance(value, torch.Tensor):
        return [(value, value._version)]
    if isinstance(value, dict):
        return [item for child in value.values() for item in tensor_versions(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in tensor_versions(child)]
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("Unsupported replay input object")
    return []


def replay(block, args, kwargs, selected, methods, teacher, modality_rows):
    """Replay the complete block at identical input; report each modality.

    Call block.forward directly to bypass the observer hook invoking this
    function. Input tensor versions and output geometry must remain unchanged.
    A GPU driver must first run an unmodified BF16 replay equality gate and
    validate that any non-tensor hybrid state also remains unchanged.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Offline real-kernel search requires no_grad, not training")
    versions = tensor_versions((args, kwargs))
    with temporary_methods(selected, methods):
        prediction = block.forward(*args, **kwargs)
    if any(value._version != version for value, version in versions):
        raise RuntimeError("Replay mutated teacher input")
    if prediction.shape != teacher.shape or not torch.isfinite(prediction).all():
        raise ValueError("Invalid replay output")
    if teacher.ndim != 3 or not torch.isfinite(teacher).all():
        raise ValueError("Expected finite batch/sequence/channel block output")
    losses = {}
    for name, indices in modality_rows.items():
        if indices.numel() == 0:
            continue
        # Chunk scoring to avoid an extra FP32 full-sequence activation copy.
        numerator = torch.zeros((), device=prediction.device, dtype=torch.float64)
        denominator = torch.zeros_like(numerator)
        for chunk in indices.split(512):
            target = teacher.index_select(1, chunk).float()
            candidate = prediction.index_select(1, chunk).float()
            numerator += (candidate - target).square().sum(dtype=torch.float64)
            denominator += target.square().sum(dtype=torch.float64)
        losses[name] = float(numerator / denominator.clamp_min(1e-30))
    if not losses:
        raise ValueError("No modality rows provided")
    return losses
