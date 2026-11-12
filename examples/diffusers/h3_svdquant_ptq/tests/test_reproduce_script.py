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

"""CPU command-wiring checks, not GPU reproduction or media-quality evidence."""

import os
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
SCRIPT = EXAMPLE / "scripts/reproduce_frozen.sh"


def invoke(env, *args):
    # Execute the trusted checked-in wrapper; input paths stay separate argv/env.
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=EXAMPLE,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.fixture
def run_env(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("H3_")}
    for name in ("VDN", "BASE", "CHECKPOINT", "LIBRARY", "TEXT"):
        env[f"H3_{name}"] = str(tmp_path / name.lower())
    env.update(
        H3_LIBRARY_SHA256="0" * 64,
        H3_RUN=str(tmp_path / "new-run"),
        H3_INPUT=str(tmp_path / "inputs"),
        H3_PYTHON="/bin/echo",
    )
    inputs = Path(env["H3_INPUT"])
    inputs.mkdir()
    return env


@pytest.mark.parametrize("cached", [False, True])
def test_sequence_and_matched_render_seeds(run_env, cached):
    inputs = Path(run_env["H3_INPUT"])
    for split in ("fit", "dev", "eval"):
        if cached:
            cache = inputs / f"{split}-cache"
            cache.mkdir()
            (cache / "prompt.pt").touch()
        else:
            for name in (f"{split}.txt", *(f"{split}{i}.png" for i in (1, 2, 3))):
                (inputs / name).touch()
    result = invoke(run_env, *(["--from-cache"] if cached else []))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    offset = 0 if cached else 3
    assert len(lines) == offset + 7
    if not cached:
        assert all(line.startswith("-m vdn_frozen.encode ") for line in lines[:3])
    assert [line.split()[2] for line in lines[offset:]] == [
        "init",
        "calibrate",
        "export-baseline",
        "reconstruct",
        "export-frozen",
        "render",
        "render",
    ]
    assert "--fit-seed 1121" in lines[offset] and "--dev-seed 1122" in lines[offset]
    assert "--seed 42 --mode bf16" in lines[-2]
    assert "--seed 42 --mode frozen" in lines[-1]
    assert not Path(run_env["H3_RUN"]).exists()  # echo never performs GPU work


def test_existing_run_is_not_overwritten(run_env):
    Path(run_env["H3_RUN"]).mkdir()
    result = invoke(run_env, "--from-cache")
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert not result.stdout


def test_missing_input_fails_before_launch(run_env):
    result = invoke(run_env, "--from-cache")
    assert result.returncode == 2
    assert "input is missing" in result.stderr
    assert not result.stdout


def test_failure_stops_following_stages(run_env):
    for split in ("fit", "dev", "eval"):
        cache = Path(run_env["H3_INPUT"]) / f"{split}-cache"
        cache.mkdir()
        (cache / "prompt.pt").touch()
    run_env["H3_PYTHON"] = "/bin/false"
    result = invoke(run_env, "--from-cache")
    assert result.returncode == 1
    assert not result.stdout
    assert not Path(run_env["H3_RUN"]).exists()


def test_help_requires_no_environment():
    env = {key: value for key, value in os.environ.items() if not key.startswith("H3_")}
    result = invoke(env, "--help")
    assert result.returncode == 0
    assert "--from-cache" in result.stdout
