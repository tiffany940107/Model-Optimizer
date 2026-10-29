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

"""Read-only launch adapter for the already hooked legacy SGLang H3 runtime.

No SGLang source mutation or monkey patching: the existing controller consumes
the byte-equivalent exported payload. This is ModelOpt state/export integration,
not a new SGLang native quantization backend.
"""

from pathlib import Path

from artifact_binding import digest, verify_binding
from streaming_roundtrip import check_unchanged, validate_overlay

__all__ = ["prepare_sglang_environment"]


def _environment(base, overlay, library):
    result = {
        key: value for key, value in base.items() if not key.startswith("SGLANG_H3_SVDQUANT_")
    }
    result.update(
        {
            "SGLANG_H3_SVDQUANT_MODE": "kernel",
            "SGLANG_H3_SVDQUANT_WEIGHTS": str(Path(overlay).resolve()),
            "SGLANG_H3_SVDQUANT_TRT_LIBRARY": str(Path(library).resolve()),
            "SGLANG_H3_SVDQUANT_RANK": "32",
            "SGLANG_H3_SVDQUANT_ALPHA": "0.5",
            "SGLANG_H3_SVDQUANT_SEED": "1101",
            "SGLANG_H3_SVDQUANT_INCLUDE_ADALN": "0",
        }
    )
    return result


def prepare_sglang_environment(
    *,
    legacy_project,
    legacy_artifact,
    calibration,
    source_receipt,
    overlay,
    library,
    base_environment,
    tensor_parallel_size=1,
):
    """Verify all bindings, then return (child-only environment, bound Ref2VA model path).

    Call this outside measured E2E timing. Only TP1 is supported: Ulysses/SP may
    vary, but a TP-sharded projection does not match these full packed weights.
    The caller must use the returned model path and existing legacy-hooked
    SGLang checkout, retain matched sampling inputs, and record its command.
    """
    if tensor_parallel_size != 1:
        raise ValueError("Packed overlay requires TP1; do not apply full weights to TP shards")
    binding = verify_binding(legacy_project, legacy_artifact, calibration, source_receipt)
    validate_overlay(overlay, binding)
    if digest(library) != binding["provenance"]["kernel_sha256"]:
        raise ValueError("SGLang library differs from validated original TRT kernel")
    check_unchanged(binding["signatures"])
    return _environment(base_environment, overlay, library), binding["model"]
