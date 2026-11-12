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

"""Offline rank32 correction, constrained to the existing canonical BF16 B."""

import math

import torch

__all__ = [
    "Statistics",
    "improves",
    "losses",
    "mean_losses",
    "replace_b",
    "sample_indices",
    "solve_delta",
    "updated_b",
]


def sample_indices(rows, modalities, *, refiner=False):
    if refiner:
        indices = torch.linspace(
            0, rows - 1, 129, device=next(iter(modalities.values())).device
        ).long()
        return indices, {"text": (0, 129)}
    indices, segments, start = [], {}, 0
    for name, count in (("video", 33), ("reference", 32), ("audio", 32), ("text", 32)):
        source = modalities[name]
        if not source.numel() or int(source.min()) < 0 or int(source.max()) >= rows:
            raise ValueError("Invalid/nonexistent modality rows")
        pick = torch.linspace(0, source.numel() - 1, count, device=source.device).long()
        indices.append(source.index_select(0, pick))
        segments[name] = (start, start + count)
        start += count
    return torch.cat(indices), segments


def losses(prediction, teacher, segments):
    if (
        prediction.shape != teacher.shape
        or not torch.isfinite(prediction).all()
        or not torch.isfinite(teacher).all()
    ):
        raise ValueError("Nonfinite or different prediction geometry")
    return {
        name: float(
            (prediction[a:b].float() - teacher[a:b].float()).square().sum()
            / teacher[a:b].float().square().sum().clamp_min(1e-30)
        )
        for name, (a, b) in segments.items()
    }


def mean_losses(rows):
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("Missing/inconsistent losses")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def improves(candidate, baseline, minimum=0.005, max_regression=0.01):
    if set(candidate) != set(baseline) or not baseline:
        raise ValueError("Inconsistent loss domains")
    values = list(candidate.values()) + list(baseline.values())
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid loss")
    return sum(candidate.values()) < sum(baseline.values()) * (1 - minimum) and all(
        candidate[key] <= baseline[key] * (1 + max_regression) for key in baseline
    )


class Statistics:
    def __init__(self, width, device):
        self.h = torch.zeros((32, 32), device=device, dtype=torch.float32)
        self.g = torch.zeros((32, width), device=device, dtype=torch.float32)
        self.calls = 0

    def add(self, down, error, teacher, segments):
        if down.shape != (teacher.shape[0], 32) or error.shape != teacher.shape:
            raise ValueError("Unexpected feature/target geometry")
        if not all(torch.isfinite(x).all() for x in (down, error, teacher)):
            raise ValueError("Nonfinite regression data")
        for start, end in segments.values():
            d, e = down[start:end].float(), error[start:end].float()
            weight = teacher[start:end].float().square().sum().clamp_min(1e-30).reciprocal()
            self.h.add_((d.T @ d) * weight)
            self.g.add_((d.T @ e) * weight)
        self.calls += 1


def solve_delta(h, g, ridge):
    if (
        h.shape != (32, 32)
        or g.ndim != 2
        or g.shape[0] != 32
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("Requires rank32 and positive ridge")
    if not torch.isfinite(h).all() or not torch.isfinite(g).all():
        raise ValueError("Nonfinite normal equations")
    scale = h.diag().clamp_min(1e-30).sqrt()
    normalized = h / (scale[:, None] * scale[None, :])
    normalized = (normalized + normalized.T) * 0.5
    rhs = g / scale[:, None]
    chol = torch.linalg.cholesky(normalized + torch.eye(32, device=h.device, dtype=h.dtype) * ridge)
    delta = (torch.cholesky_solve(rhs, chol) / scale[:, None]).T.contiguous()
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite correction")
    return delta


def updated_b(original, delta, shrink, max_relative_norm=0.1):
    if (
        original.dtype != torch.bfloat16
        or original.ndim != 2
        or original.shape[1] != 32
        or original.shape != delta.shape
    ):
        raise ValueError("Canonical BF16 rank32 B required")
    if not 0 < shrink <= 1 or not 0 < max_relative_norm <= 1:
        raise ValueError("Invalid trust-region bounds")
    change = delta.float() * shrink
    limit = original.float().norm() * max_relative_norm
    change = change * (limit / change.norm().clamp_min(1e-30)).clamp(max=1)
    candidate = (original.float() + change).bfloat16().contiguous()
    if not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite BF16 correction")
    for _ in range(8):
        distance = (candidate.float() - original.float()).norm()
        if distance <= limit:
            return candidate
        change *= 0.95 * limit / distance.clamp_min(1e-30)
        candidate = (original.float() + change).bfloat16().contiguous()
    # Do not violate a trust bound merely because BF16 rounding crossed it.
    return original.clone()


def replace_b(payload, candidate):
    old = payload["svdquant_lora_b"]
    if (
        candidate.shape != old.shape
        or candidate.dtype != torch.bfloat16
        or not torch.isfinite(candidate).all()
    ):
        raise ValueError("B schema differs")
    result = dict(payload)
    result["svdquant_lora_b"] = candidate.contiguous()
    return result
