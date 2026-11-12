#!/usr/bin/env bash
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

set -euo pipefail

usage() {
    echo "Usage: bash scripts/reproduce_frozen.sh [--from-cache | --help]"
    echo "Required environment: H3_VDN H3_BASE H3_CHECKPOINT H3_LIBRARY"
    echo "  H3_LIBRARY_SHA256 H3_RUN H3_INPUT; H3_TEXT is required for encoding."
    echo "H3_RUN must be new. Keep H3_RUN and H3_INPUT outside this repository."
    echo "Default: encode fit/dev/eval, then run all fresh calibration-to-render stages."
    echo "Inputs: H3_INPUT/{fit,dev,eval}.txt and {fit,dev,eval}{1,2,3}.png."
    echo "--from-cache uses H3_INPUT/{fit,dev,eval}-cache/prompt.pt instead."
    echo "Optional: H3_PYTHON (default python). Fixed seeds: fit=1121, dev=1122, eval=42."
}

encode_inputs=1
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --from-cache) encode_inputs=0 ;;
    "") ;;
    *) usage >&2; exit 2 ;;
esac

for name in H3_VDN H3_BASE H3_CHECKPOINT H3_LIBRARY H3_LIBRARY_SHA256 H3_RUN H3_INPUT; do
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: $name" >&2
        exit 2
    fi
done
if [[ -e "$H3_RUN" || -L "$H3_RUN" ]]; then
    echo "H3_RUN already exists; choose a fresh directory." >&2
    exit 2
fi
if (( encode_inputs )) && [[ -z "${H3_TEXT:-}" ]]; then
    echo "H3_TEXT is required for encoding." >&2
    exit 2
fi

# Validate all input names before launching any expensive encoding work.
for split in fit dev eval; do
    if (( encode_inputs )); then
        inputs=("$H3_INPUT/$split.txt" "$H3_INPUT/${split}1.png" "$H3_INPUT/${split}2.png" "$H3_INPUT/${split}3.png")
        if [[ -e "$H3_INPUT/$split-cache" || -L "$H3_INPUT/$split-cache" ]]; then
            echo "An encoding output already exists; use fresh paths or --from-cache." >&2
            exit 2
        fi
    else
        inputs=("$H3_INPUT/$split-cache/prompt.pt")
    fi
    for input in "${inputs[@]}"; do
        if [[ ! -f "$input" ]]; then
            echo "A required input is missing; see --help for the expected file names." >&2
            exit 2
        fi
    done
done

# Run from the documented example directory; do not switch caller-relative paths.
if [[ ! -f vdn_frozen/cli.py ]]; then
    echo "Run from examples/diffusers/h3_svdquant_ptq." >&2
    exit 2
fi
export PYTHONPATH="$PWD:$H3_VDN${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
h3_python="${H3_PYTHON:-python}"

if (( encode_inputs )); then
    for split in fit dev eval; do
        "$h3_python" -m vdn_frozen.encode \
            --vdn-root "$H3_VDN" --text-source "$H3_TEXT" --base "$H3_BASE" \
            --prompt-file "$H3_INPUT/$split.txt" \
            --refs "$H3_INPUT/${split}1.png" "$H3_INPUT/${split}2.png" "$H3_INPUT/${split}3.png" \
            --output "$H3_INPUT/$split-cache"
    done
fi

"$h3_python" -m vdn_frozen.cli init \
    --vdn-root "$H3_VDN" --base "$H3_BASE" --checkpoint "$H3_CHECKPOINT" \
    --library "$H3_LIBRARY" --library-sha256 "$H3_LIBRARY_SHA256" \
    --fit-cache "$H3_INPUT/fit-cache/prompt.pt" --fit-seed 1121 \
    --dev-cache "$H3_INPUT/dev-cache/prompt.pt" --dev-seed 1122 \
    --run "$H3_RUN"
for stage in calibrate export-baseline reconstruct export-frozen; do
    "$h3_python" -m vdn_frozen.cli "$stage" --run "$H3_RUN"
done
for mode in bf16 frozen; do
    "$h3_python" -m vdn_frozen.cli render --run "$H3_RUN" \
        --prompt-cache "$H3_INPUT/eval-cache/prompt.pt" --seed 42 \
        --mode "$mode" --output "$H3_RUN/evaluation/$mode"
done
