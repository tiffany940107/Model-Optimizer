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

"""Serial local Ref2VA request runner with explicit per-rank forward accounting."""

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from portable_io import digest, read_json, write_json
from portable_sglang import check_files

__all__ = ["check_disjoint", "load_requests", "run_requests"]


def load_requests(path):
    """Load user-owned image references relative to a strict request manifest."""
    path = Path(path).resolve(strict=True)
    data = read_json(path)
    if (
        set(data) != {"format", "purpose", "requests"}
        or data["format"] != "h3-ptq-requests-v1"
        or data["purpose"] not in {"calibration", "evaluation"}
        or not isinstance(data["requests"], list)
        or not data["requests"]
    ):
        raise ValueError("Invalid request manifest")
    media, ids = [], set()
    for row in data["requests"]:
        if set(row) != {
            "id",
            "prompt",
            "seed",
            "reference_images",
            "seconds",
            "short_edge",
            "aspect_ratio",
            "num_inference_steps",
            "expected_dit_calls",
        }:
            raise ValueError("Unexpected/missing request fields")
        if (
            not isinstance(row["id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["id"])
            or row["id"] in ids
            or not isinstance(row["prompt"], str)
            or not row["prompt"].strip()
            or type(row["seed"]) is not int
            or not 0 <= row["seed"] < 2**31
            or row["seconds"] != 5
            or row["short_edge"] != 768
            or row["aspect_ratio"] not in {"16:9", "9:16"}
            or row["num_inference_steps"] != 50
            or row["expected_dit_calls"] != 49
        ):
            raise ValueError(
                "Require unique IDs, valid seeds and the documented 5s/50-requested/49-call recipe"
            )
        ids.add(row["id"])
        refs = row["reference_images"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 3:
            raise ValueError("Require 1-3 user-owned reference images per request")
        for name in refs:
            if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("References must be local manifest-relative paths")
            reference = (path.parent / name).resolve(strict=True)
            if not reference.is_relative_to(path.parent) or not reference.is_file():
                raise ValueError("Reference escapes manifest directory or is not a file")
            media.append(reference)
    return data, media


def check_disjoint(first, second):
    """Reject reused prompts or image bytes across calibration and evaluation."""
    left, left_media = load_requests(first)
    right, right_media = load_requests(second)
    if {left["purpose"], right["purpose"]} != {"calibration", "evaluation"}:
        raise ValueError("Need one calibration and one held-out evaluation manifest")

    def prompts(data):
        return {" ".join(row["prompt"].lower().split()) for row in data["requests"]}

    if prompts(left) & prompts(right) or {digest(p) for p in left_media} & {
        digest(p) for p in right_media
    }:
        raise ValueError("Calibration/evaluation reuse detected (prompt or reference bytes)")
    return right_media


def _json_request(base, route, body=None):
    request = urllib.request.Request(
        base + route,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _counts(launch):
    counts = []
    for rank in range(launch["world_size"]):
        path = Path(launch["output"]) / f"forwards-rank{rank}.jsonl"
        content = path.read_text() if path.exists() else ""
        rows = content.splitlines()
        # A final partial write is not a successful forward receipt yet.
        if content and not content.endswith("\n"):
            rows = rows[:-1]
        parsed = [json.loads(row) for row in rows]
        if any(row != {"forward": i + 1, "mode": launch["mode"]} for i, row in enumerate(parsed)):
            raise ValueError("Unexpected forward audit sequence")
        counts.append(len(parsed))
    return counts


def _inspect_video(path, row):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    streams = json.loads(result.stdout)["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    expected = (1344, 768) if row["aspect_ratio"] == "16:9" else (768, 1344)
    if (
        len(video) != 1
        or (video[0]["width"], video[0]["height"]) != expected
        or int(video[0]["nb_read_frames"]) != 124
        or not any(stream["codec_type"] == "audio" for stream in streams)
    ):
        raise ValueError("Expected 124-frame 1344x768 (or portrait) video with audio")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ],
        check=True,
        timeout=120,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return streams


def run_requests(args):
    """Run once against a fresh foreground server; do not retry submitted requests."""
    launch = read_json(args.launch)
    data, _ = load_requests(args.requests)
    if (
        launch.get("format") != "h3-modelopt-portable-launch-v1"
        or digest(args.requests) != launch["requests_sha256"]
        or args.port != launch["port"]
    ):
        raise ValueError("Request manifest/port differs from launch")
    check_files(launch["files"])
    args.output.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    deadline = time.monotonic() + args.startup_timeout
    while True:
        try:
            _json_request(base, "/health")
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Server startup timed out; inspect server log and stop it manually"
                ) from None
            time.sleep(2)
    if any(_counts(launch)):
        raise ValueError(
            "Server has already executed requests; restart with a fresh output directory"
        )
    completed, expected = [], 0
    for row in data["requests"]:
        check_files(launch["files"])
        body = {
            "model": launch["model"],
            "prompt": row["prompt"],
            "task": "ref2va",
            "seconds": row["seconds"],
            "seed": row["seed"],
            "conditions": [
                {
                    "type": "image",
                    "role": "reference",
                    "uri": (args.requests.resolve().parent / name).resolve().as_uri(),
                }
                for name in row["reference_images"]
            ],
            "target": {
                "short_edge": row["short_edge"],
                "aspect_ratio": row["aspect_ratio"],
                "duration_seconds": 5.0,
            },
            "num_inference_steps": row["num_inference_steps"],
            "num_outputs_per_prompt": 1,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        }
        started = time.monotonic()
        submitted = _json_request(base, "/v1/videos", body)
        video_id = submitted["id"]
        if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", video_id):
            raise ValueError("Invalid server video ID")
        status = submitted
        reported = started
        while status.get("status") != "completed":
            if status.get("status") in {"failed", "cancelled"}:
                raise RuntimeError(f"Request {row['id']} failed: {status}")
            if time.monotonic() - started > args.request_timeout:
                raise TimeoutError(
                    "Generation timed out; stop/restart the server, do not reuse calibration"
                )
            time.sleep(2)
            status = _json_request(base, f"/v1/videos/{video_id}")
            if time.monotonic() - reported >= 30:
                print(
                    json.dumps(
                        {
                            "id": row["id"],
                            "status": status.get("status"),
                            "forwards_per_rank": _counts(launch),
                        }
                    ),
                    flush=True,
                )
                reported = time.monotonic()
        elapsed = time.monotonic() - started
        expected += row["expected_dit_calls"]
        if _counts(launch) != [expected] * launch["world_size"]:
            raise ValueError("Observed per-rank DiT calls differ from the recipe")
        destination = args.output / (row["id"] + ".mp4")
        with (
            urllib.request.urlopen(base + f"/v1/videos/{video_id}/content", timeout=60) as response,
            destination.open("xb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        streams = _inspect_video(destination, row)
        completed.append(
            {
                "id": row["id"],
                "video_id": video_id,
                "file": destination.name,
                "sha256": digest(destination),
                "api_wall_seconds": elapsed,
                "cumulative_forwards_per_rank": expected,
                "streams": streams,
            }
        )
        write_json(args.output / (row["id"] + ".json"), completed[-1])
    check_files(launch["files"])
    write_json(
        args.output / "complete.json",
        {
            "format": "h3-ptq-videos-v1",
            "mode": launch["mode"],
            "requests_sha256": digest(args.requests),
            "launch_sha256": digest(args.launch),
            "results": completed,
            "scope": "Media decoding and execution checks only; not a perceptual quality acceptance",
        },
    )


def main():
    """Submit a checked manifest after starting h3_ptq.py serve in another terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("launch", "requests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or min(args.startup_timeout, args.request_timeout) < 1:
        parser.error("Port and timeouts must be positive and valid")
    run_requests(args)


if __name__ == "__main__":
    main()
