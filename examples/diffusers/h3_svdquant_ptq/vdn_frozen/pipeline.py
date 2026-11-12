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

"""Portable calibration, baseline export and matched Frozen/BF16 generation."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from .environment import build, identity, native_modules, read, sha, write
from .policy import install, select
from .quantization import Calibration, export, load_methods
from .runtime import Dispatch, RoutedMethod

__all__ = ["calibrate", "export_baseline", "render", "validate_cache"]


def validate_cache(path):
    """Restrict this recipe to three-reference, portrait VDN conditioning."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("height"), value.get("width"), value.get("reference_size")) != (1344, 768, 768):
        raise ValueError("Need portrait768x1344 cache with768-short-edge references")
    if value.get("keyframe_anchors") != ["ref"] * 3:
        raise ValueError("Need exactly three ordered image references")
    embeds, tags = value["prompt_embeds"], value["text_token_tags"]
    if (
        embeds.ndim != 2
        or embeds.shape[1] != 5120
        or embeds.dtype != torch.bfloat16
        or tags.shape != embeds.shape[:1]
        or not torch.isfinite(embeds).all()
    ):
        raise ValueError("Invalid text conditioning")
    refs = value["condition_latents"]
    if len(refs) != 3 or any(
        t.ndim != 5 or tuple(t.shape[:3]) != (1, 24, 1) or not torch.isfinite(t).all() for t in refs
    ):
        raise ValueError("Invalid reference latent geometry/content")


def calibrate(run):
    """Collect pooled absmax and exact BF16 teachers on two eight-NFE requests."""
    config, bound = read(run / "config.json"), identity(run)
    output = run / "calibration"
    output.mkdir(exist_ok=False)
    native, load_prompt = native_modules(config)
    model = build(config)
    tracker = Calibration(model.transformer)
    teacher_hashes = {}
    for index, role in enumerate(("fit", "dev")):
        validate_cache(config[f"{role}_cache"])
        embeds, tags, conditions = load_prompt(config[f"{role}_cache"], "cuda")
        print(f"PHASE BF16 calibration trajectory {index + 1}/2", flush=True)
        video, audio = native.generate_latents(
            model.transformer,
            embeds,
            tags,
            124,
            8,
            config[f"{role}_seed"],
            "cuda",
            video_shift=12.0,
            audio_shift=3.0,
            conditions=conditions,
        )
        if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
            raise ValueError("Nonfinite teacher latents")
        path = output / f"teacher-{index}.safetensors"
        save_file({"video": video.cpu().contiguous(), "audio": audio.cpu().contiguous()}, str(path))
        teacher_hashes[str(index)] = sha(path)
    stats = output / "activation_absmax.safetensors"
    counts = tracker.finish(stats, 16)
    write(
        output / "complete.json",
        {
            "binding": bound,
            "teacher_sha256": teacher_hashes,
            "statistics_sha256": sha(stats),
            "counts": counts,
            "scope": "Two calibration/development prompts; not independent quality acceptance",
        },
    )


def export_baseline(run):
    """Actual ModelOpt roundtrip and original TRT-compatible NVFP4 export."""
    config, bound = read(run / "config.json"), identity(run)
    calibration = read(run / "calibration/complete.json")
    stats = run / "calibration/activation_absmax.safetensors"
    if calibration["binding"] != bound or calibration["statistics_sha256"] != sha(stats):
        raise ValueError("Calibration source/statistics mismatch")
    model = build(config)
    export(model.transformer, stats, run / "baseline", bound, Path(config["library"]))


def render(run, cache, seed, mode, output):
    """Render all eight NFEs; Frozen uses original BF16 only at indices0 and5."""
    if mode not in {"bf16", "frozen"}:
        raise ValueError("Expected bf16 or frozen")
    config, bound = read(run / "config.json"), identity(run)
    validate_cache(cache)
    native, load_prompt = native_modules(config)
    output.mkdir(parents=True, exist_ok=False)
    model = build(config, decoders=True)
    selected = select(model.transformer)
    dispatch = None
    if mode == "frozen":
        artifact = run / "frozen"
        manifest = read(artifact / "manifest.json")
        optimization = manifest.get("optimization", {})
        if (
            optimization.get("kind") != "fixed-rank32-canonical-B-reconstruction"
            or optimization.get("only_changed_field") != "svdquant_lora_b"
            or optimization.get("baseline_manifest_sha256") != sha(run / "baseline/manifest.json")
            or optimization.get("fit_receipt_sha256") != sha(run / "reconstruction/complete.json")
        ):
            raise ValueError("Need the checked B-only export, not an arbitrary quantized artifact")
        methods = load_methods(artifact, bound, Path(config["library"]))
        dispatch = Dispatch(selected)
        routes = {
            name: RoutedMethod(name, selected[name], method, dispatch)
            for name, method in methods.items()
        }
        install(model.transformer, routes)
        model.transformer.register_forward_pre_hook(dispatch.before)
        model.transformer.register_forward_hook(dispatch.after, always_call=True)
        dispatch.start(bf16_steps=(0, 5), bf16_groups=())
    calls = []

    def observe(_model, _args, result):
        if not all(torch.isfinite(t).all() for t in result):
            raise ValueError("Nonfinite transformer prediction")
        calls.append(len(calls))
        print(f"PHASE {mode} NFE {len(calls)}/8", flush=True)

    handle = model.transformer.register_forward_hook(observe)
    embeds, tags, conditions = load_prompt(str(cache), "cuda")
    times = []
    torch.cuda.reset_peak_memory_stats()
    video, audio = native.generate_latents(
        model.transformer,
        embeds,
        tags,
        124,
        8,
        seed,
        "cuda",
        video_shift=12.0,
        audio_shift=3.0,
        conditions=conditions,
        step_seconds=times,
    )
    handle.remove()
    if len(calls) != 8 or len(times) != 8:
        raise ValueError("Incomplete8NFE generation")
    counts = dispatch.finish() if dispatch else {}
    if dispatch and any(row != {"bf16": 2, "nvfp4": 6} for row in counts.values()):
        raise ValueError("Frozen must use BF16twice/NVFP4six times for every group")
    save_file(
        {"video": video.cpu().contiguous(), "audio": audio.cpu().contiguous()},
        str(output / "latents.safetensors"),
    )
    peak = torch.cuda.max_memory_allocated() / 2**30
    native.decode_and_save(
        video, audio, model.vae, model.audio_vae, str(output / "video.mp4"), "cuda"
    )
    write(
        output / "complete.json",
        {
            "binding": bound,
            "mode": mode,
            "seed": seed,
            "prompt_cache_sha256": sha(cache),
            "actual_nfe": len(calls),
            "counts": counts,
            "bf16_steps": [0, 5] if dispatch else list(range(8)),
            "bf16_groups": [],
            "video_sha256": sha(output / "video.mp4"),
            "latents_sha256": sha(output / "latents.safetensors"),
            "step_seconds_instrumented": times,
            "denoise_peak_allocated_gib": peak,
            "scope": "Serial eager accuracy run with originalBF16weights retained; not production latency/memory",
        },
    )
