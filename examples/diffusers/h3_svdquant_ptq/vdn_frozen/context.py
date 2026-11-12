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

"""Serial, source-bound context for the unchanged rank32 B fitting algorithm."""

import json
from pathlib import Path

import torch
from portable_io import local_file
from safetensors.torch import load_file, save_file

from .block_replay import generated_video_rows
from .environment import Method, build, identity, native_modules, read, sha, write
from .fit import replace_b
from .policy import groups, install, select
from .runtime import Dispatch, RoutedMethod, relative_l2, validate_payload

__all__ = ["Context"]


class Context:
    def __init__(self, root, output):
        self.root, self.output = root.resolve(strict=True), output
        config = read(self.root / "config.json")
        binding = identity(self.root)
        self.sha, self.write = sha, write
        self.native, self.load_prompt = native_modules(config)
        self.Method, self.groups = Method, groups()
        self.library = config["library"]
        self.artifact = self.root / "baseline"
        self.manifest = read(self.artifact / "manifest.json")
        if (
            self.manifest["binding"] != binding
            or self.manifest["rank"] != 32
            or self.manifest["library_sha256"] != config["library_sha256"]
            or len(self.manifest["layers"]) != 208
        ):
            raise ValueError("Baseline source/kernel/rank mismatch")
        self.identity = {
            "binding": binding,
            "baseline_manifest_sha256": sha(self.artifact / "manifest.json"),
            "kernel_sha256": config["library_sha256"],
        }
        self.target_count = generated_video_rows(self.native.video_latent_num_frames)
        if self.target_count != 37296:
            raise ValueError("Expected 124-frame portrait geometry")
        output.mkdir(parents=True, exist_ok=False)
        write(output / "started.json", self.identity)
        self.model = build(config)
        self.selected = select(self.model.transformer)
        self.payloads, self.methods = {}, {}
        for group, record in zip(self.groups, self.manifest["layers"], strict=True):
            if (record["name"], record["shape"], record["modules"]) != (
                group.name,
                list(group.shape),
                list(group.modules),
            ):
                raise ValueError("Packed geometry/order mismatch")
            path = local_file(self.artifact, record["file"])
            if sha(path) != record["sha256"]:
                raise ValueError("Packed content mismatch")
            tensors = load_file(str(path), device="cuda")
            validate_payload(tensors, group.shape)
            self.payloads[group.name] = tensors
            self.methods[group.name] = Method(tensors)
        self.prompts = [
            (Path(config[f"{role}_cache"]), {"seed": config[f"{role}_seed"]})
            for role in ("fit", "dev")
        ]
        self.name, self.step, self.inside, self.dispatch = None, 0, False, None
        self.model.transformer.register_forward_pre_hook(self.before, with_kwargs=True)
        self.model.transformer.register_forward_hook(self.after)

    def before(self, _m, _args, kwargs):
        if self.name is None or self.inside or self.step >= 8:
            raise RuntimeError("Unexpected root forward")
        self.inside = True
        indices = kwargs["video_indices"]
        self.sequence_rows = kwargs["position_ids"].shape[0]
        if indices.numel() <= self.target_count:
            raise ValueError("Expected references plus generated video")
        self.modalities = {
            "video": indices[-self.target_count :],
            "reference": indices[: -self.target_count],
            "audio": kwargs["audio_indices"],
            "text": kwargs["text_indices"],
        }

    def after(self, _m, _args, output):
        if not all(torch.isfinite(value).all() for value in output):
            raise ValueError("Nonfinite DiT output")
        self.step += 1
        self.inside = False
        print(f"PHASE {self.name} NFE {self.step}/8", flush=True)

    def reference(self, row):
        path = self.root / "calibration" / f"teacher-{row}.safetensors"
        receipt = read(self.root / "calibration/complete.json")
        if (
            receipt["binding"] != self.identity["binding"]
            or self.sha(path) != receipt["teacher_sha256"][str(row)]
        ):
            raise ValueError("BF16 teacher cache changed")
        return load_file(str(path))

    def activate(self, updates):
        if self.dispatch is None:
            self.dispatch = Dispatch(self.selected)
            self.routes = {
                name: RoutedMethod(name, self.selected[name], method, self.dispatch)
                for name, method in self.methods.items()
            }
            install(self.model.transformer, self.routes)
            self.model.transformer.register_forward_pre_hook(self.dispatch.before)
            self.model.transformer.register_forward_hook(self.dispatch.after, always_call=True)
        if self.dispatch.active or not set(updates) <= set(self.selected):
            raise ValueError("Cannot switch during a generation or to unknown groups")
        for name, route in self.routes.items():
            route.method = (
                self.Method(replace_b(self.payloads[name], updates[name].to("cuda")))
                if name in updates
                else self.methods[name]
            )

    def generate(self, name, row, *, steps=(), recovered=(), reference=None, exact=False):
        path = self.output / name
        path.mkdir(exist_ok=False)
        self.name, self.step = name, 0
        prompt, entry = self.prompts[row]
        embeds, tags, conditions = self.load_prompt(str(prompt), "cuda")
        if self.dispatch:
            self.dispatch.start(bf16_steps=steps, bf16_groups=recovered)
        elif steps or recovered:
            raise ValueError("No mixed dispatch installed")
        times = []
        torch.cuda.reset_peak_memory_stats()
        video, audio = self.native.generate_latents(
            self.model.transformer,
            embeds,
            tags,
            124,
            8,
            entry["seed"],
            "cuda",
            video_shift=12.0,
            audio_shift=3.0,
            conditions=conditions,
            step_seconds=times,
        )
        if self.step != 8 or self.inside:
            raise ValueError("Incomplete NFE")
        counts = self.dispatch.finish() if self.dispatch else {}
        self.name = None
        values = {"video": video.cpu().contiguous(), "audio": audio.cpu().contiguous()}
        save_file(values, str(path / "latents.safetensors"))
        result = {
            "name": name,
            "row": row,
            "seed": entry["seed"],
            "prompt_sha256": self.sha(prompt),
            "actual_nfe": 8,
            "steps_bf16": list(steps),
            "groups_bf16": list(recovered),
            "counts": counts,
            "step_seconds_instrumented": times,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "latents_sha256": self.sha(path / "latents.safetensors"),
        }
        if reference is not None:
            result["relative_l2"] = {
                key: relative_l2(values[key], reference[key]) for key in values
            }
            result["exact"] = {key: torch.equal(values[key], reference[key]) for key in values}
            if exact and not all(result["exact"].values()):
                raise ValueError("Baseline/teacher replay changed")
        self.write(path / "complete.json", result)
        print(
            "RESULT "
            + json.dumps({key: value for key, value in result.items() if key != "counts"}),
            flush=True,
        )
        return values, result
