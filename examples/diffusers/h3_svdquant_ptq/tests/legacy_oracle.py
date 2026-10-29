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

"""Explicit, hash-bound dependency for legacy parity tests; no bundled runtime."""

import json
import os
from pathlib import Path

import pytest
from recipe_comparison import load_legacy_core


def load_test_oracle():
    location = os.environ.get("H3_LEGACY_ROOT")
    if not location:
        message = "Set H3_LEGACY_ROOT to run external legacy parity tests"
        if os.environ.get("H3_REQUIRE_LEGACY_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message)
    contract = json.loads(
        (Path(__file__).resolve().parents[1] / "compat_contract_v1.json").read_text()
    )
    return load_legacy_core(location, contract["legacy_core_sha256"])
