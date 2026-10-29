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

"""Check that unavailable or modified legacy source is never accepted as parity."""

import pytest
from legacy_oracle import load_test_oracle


def test_optional_external_oracle_is_explicitly_skipped(monkeypatch):
    monkeypatch.delenv("H3_LEGACY_ROOT", raising=False)
    monkeypatch.delenv("H3_REQUIRE_LEGACY_TESTS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="H3_LEGACY_ROOT"):
        load_test_oracle()


def test_required_external_oracle_fails_without_configuration(monkeypatch):
    monkeypatch.delenv("H3_LEGACY_ROOT", raising=False)
    monkeypatch.setenv("H3_REQUIRE_LEGACY_TESTS", "1")
    with pytest.raises(pytest.fail.Exception, match="H3_LEGACY_ROOT"):
        load_test_oracle()


def test_modified_external_oracle_is_rejected_before_import(monkeypatch, tmp_path):
    directory = tmp_path / "svdquant_h3"
    directory.mkdir()
    (directory / "core.py").write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setenv("H3_LEGACY_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="source changed"):
        load_test_oracle()
