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

"""Portable calibration/export/serving entrypoint; no private project layout."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from h3_requests import check_disjoint, load_requests
from portable_calibration import merge_ranks
from portable_export import export_model, verify_export
from portable_io import digest, source_identity, write_json
from portable_sglang import file_signature

EXAMPLE = Path(__file__).resolve().parents[1]
SGLANG_REVISION = "0ba46c88e5b16c739be0b698ec7405a147a05cb4"
MODEL_FILE = "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
MODEL_ORIGINAL_SHA = "5f87319969c446685ee93d422fc34a7c040defb238eff2274d664f2f8310e997"


def prepare_sglang(path):
    """Apply only the reviewed opt-in patch to a matching, unmodified H3 file."""
    path = path.resolve(strict=True)
    if (
        subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        != SGLANG_REVISION
    ):
        raise ValueError("Use the documented pinned SGLang checkout")
    if digest(path / MODEL_FILE) != MODEL_ORIGINAL_SHA:
        raise ValueError("H3 source is already patched or modified; use a clean separate checkout")
    patch = EXAMPLE / "patches/sglang_h3_modelopt_ptq.patch"
    subprocess.run(["git", "-C", str(path), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(path), "apply", str(patch)], check=True)


def serve(args):
    """Validate CPU artifacts first, then foreground-exec the user-owned server."""
    if args.output.exists():
        raise FileExistsError(args.output)
    checkout = args.sglang.resolve(strict=True)
    if (
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        != SGLANG_REVISION
    ):
        raise ValueError("SGLang revision differs")
    # Require exactly the shipped patch, not an arbitrary matching marker string.
    patch = EXAMPLE / "patches/sglang_h3_modelopt_ptq.patch"
    subprocess.run(
        ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)], check=True
    )
    original = subprocess.check_output(["git", "-C", str(checkout), "show", f"HEAD:{MODEL_FILE}"])
    insertion = b"""        # Optional example integration; ordinary inference has no ModelOpt dependency.
        import os
        if os.environ.get("H3_MODELOPT_PTQ_CONFIG"):
            from portable_sglang import install
            install(self)

"""
    expected = original.replace(
        b"    @staticmethod\n    def _pos_ids",
        insertion + b"    @staticmethod\n    def _pos_ids",
        1,
    )
    if digest(checkout / MODEL_FILE) != hashlib.sha256(expected).hexdigest():
        raise ValueError("Unexpected additional H3 source changes")
    if not 1 <= args.port <= 65534:
        raise ValueError("Invalid HTTP/master port")
    source = source_identity(args.model)
    requests, media = load_requests(args.requests)
    other_media = check_disjoint(args.requests, args.other_requests)
    if args.mode == "collect" and requests["purpose"] != "calibration":
        raise ValueError("Collection requires an independent calibration manifest")
    if args.mode != "collect" and requests["purpose"] != "evaluation":
        raise ValueError("Inference requires a held-out evaluation manifest")
    manifest = None
    if args.mode == "kernel":
        if args.artifact is None or args.library is None:
            raise ValueError("Kernel mode requires artifact and compatible TRT library")
        manifest = verify_export(args.model, args.artifact)
        if digest(args.other_requests) != manifest["calibration_requests_sha256"]:
            raise ValueError("Held-out check must use the artifact's actual calibration manifest")
        if sorted({digest(p) for p in other_media}) != manifest["calibration_media_sha256"]:
            raise ValueError("Calibration reference contents changed since collection")
        if digest(args.library) != manifest["runtime"]["library_sha256"]:
            raise ValueError("Inference must use the validated export library")
    elif args.artifact or args.library:
        raise ValueError("BF16/collect must not receive packed artifact or kernel library")
    files = [args.model / relative for relative in source["files"]]
    files += [
        args.requests,
        args.other_requests,
        checkout / MODEL_FILE,
        patch,
        *media,
        *other_media,
    ]
    files += list(EXAMPLE.glob("*.py"))
    files += list((EXAMPLE / "scripts").glob("h3_*.py"))
    if manifest:
        files += [
            args.library,
            args.artifact / "manifest.json",
            args.artifact / "absmax.safetensors",
        ]
        files += [args.artifact / item["file"] for item in manifest["layers"].values()]
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "format": "h3-modelopt-portable-launch-v1",
        "mode": args.mode,
        "source": source,
        "model": str(args.model.resolve()),
        "world_size": args.gpus,
        "output": str(args.output.resolve()),
        "port": args.port,
        "sglang_revision": SGLANG_REVISION,
        "requests_sha256": digest(args.requests),
        "requests_media_sha256": sorted({digest(p) for p in media}),
        "expected_forwards": sum(row["expected_dit_calls"] for row in requests["requests"]),
        "files": {str(p.resolve()): file_signature(p) for p in files},
        "artifact_manifest": manifest,
        "artifact": str(args.artifact.resolve()) if args.artifact else None,
        "library": str(args.library.resolve()) if args.library else None,
    }
    config_path = args.output / "launch.json"
    write_json(config_path, config)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SGLANG_H3_SVDQUANT_", "H3_MODELOPT_PTQ_"))
    }
    env["H3_MODELOPT_PTQ_CONFIG"] = str(config_path.resolve())
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EXAMPLE), str(EXAMPLE.parents[2]), str(checkout / "python"), env.get("PYTHONPATH", "")]
    )
    command = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.entrypoints.cli.main",
        "serve",
        "--model-path",
        str(args.model.resolve()),
        "--num-gpus",
        str(args.gpus),
        "--ulysses-degree",
        str(args.gpus),
        "--performance-mode",
        "speed",
        "--warmup-mode",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--master-port",
        str(args.port + 1),
    ]
    write_json(
        args.output / "command.json", {"argv": command, "scope": "TP1; no offload or concurrency"}
    )
    os.execve(sys.executable, command, env)


def main():
    """Run one explicit pipeline stage; outputs never overwrite prior evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    patch = sub.add_parser("prepare-sglang")
    patch.add_argument("--sglang", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--calibration", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--world-size", type=int, required=True)
    merge.add_argument("--expected-forwards", type=int, required=True)
    export = sub.add_parser("export")
    for name in ("model", "calibration", "output", "library"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--library-sha256", required=True)
    verify = sub.add_parser("verify")
    for name in ("model", "artifact"):
        verify.add_argument("--" + name, type=Path, required=True)
    launch = sub.add_parser("serve")
    for name in ("model", "sglang", "requests", "other-requests", "output"):
        launch.add_argument("--" + name, type=Path, required=True)
    launch.add_argument("--mode", choices=("bf16", "collect", "kernel"), required=True)
    launch.add_argument("--artifact", type=Path)
    launch.add_argument("--library", type=Path)
    launch.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=2)
    launch.add_argument("--port", type=int, default=30010)
    args = parser.parse_args()
    if args.stage == "prepare-sglang":
        prepare_sglang(args.sglang)
    elif args.stage == "merge":
        merge_ranks(
            args.calibration,
            args.output,
            world_size=args.world_size,
            expected_forwards=args.expected_forwards,
        )
    elif args.stage == "export":
        export_model(
            args.model,
            args.calibration,
            args.output,
            library=args.library,
            library_sha256=args.library_sha256,
        )
    elif args.stage == "verify":
        verify_export(args.model, args.artifact)
        print("Full portable export validation passed; not an E2E quality result")
    else:
        serve(args)


if __name__ == "__main__":
    main()
