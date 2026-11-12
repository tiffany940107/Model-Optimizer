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

"""Run with python -m vdn_frozen.cli from the H3 example directory."""

import argparse
from pathlib import Path

from .environment import initialize

__all__ = ["main"]


def main():
    """Independent stages use exclusive output paths and private local receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    init = sub.add_parser("init", help="CPU-only source inventory; no historical receipts needed")
    for name in ("vdn-root", "base", "checkpoint", "library", "fit-cache", "dev-cache", "run"):
        init.add_argument("--" + name, type=Path, required=True)
    init.add_argument("--library-sha256", required=True)
    init.add_argument("--fit-seed", type=int, required=True)
    init.add_argument("--dev-seed", type=int, required=True)
    for name in ("calibrate", "export-baseline", "reconstruct", "export-frozen", "render"):
        stage = sub.add_parser(name)
        stage.add_argument("--run", type=Path, required=True)
        if name == "render":
            stage.add_argument("--prompt-cache", type=Path, required=True)
            stage.add_argument("--seed", type=int, required=True)
            stage.add_argument("--mode", choices=("bf16", "frozen"), required=True)
            stage.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "init":
        from .pipeline import validate_cache

        validate_cache(args.fit_cache)
        validate_cache(args.dev_cache)
        initialize(args)
        return
    # Heavy GPU pipeline dependencies are deferred so init/help need no VDN install.
    from . import export_reconstructed, pipeline, reconstruct

    args.run = args.run.resolve(strict=True)
    if args.stage == "calibrate":
        pipeline.calibrate(args.run)
    elif args.stage == "export-baseline":
        pipeline.export_baseline(args.run)
    elif args.stage == "reconstruct":
        reconstruct.run(args.run, args.run / "reconstruction")
    elif args.stage == "export-frozen":
        export_reconstructed.run(args.run, args.run / "frozen")
    elif args.stage == "render":
        pipeline.render(
            args.run, args.prompt_cache.resolve(strict=True), args.seed, args.mode, args.output
        )


if __name__ == "__main__":
    main()
