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

"""Opt-in post-load hooks for a pinned SGLang checkout, TP1 and serial requests."""

import json
import os
import threading
from pathlib import Path

import torch
from h3_modelopt_ptq import h3_layer_names
from portable_calibration import ChannelMax, save_rank
from portable_io import digest, local_file, read_json, validate_order
from portable_runtime import PackedLinear, load_runtime
from safetensors.torch import load_file

__all__ = ["Controller", "check_files", "file_signature", "install"]


def file_signature(path):
    """Detect local source mutation after the launcher's full content checks."""
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]


def check_files(signatures):
    """Check trusted-local launch inputs have not changed since CPU preparation."""
    for path, expected in signatures.items():
        if file_signature(path) != expected:
            raise ValueError("Launch input changed: " + path)


class Controller:
    """Instrument real BF16 inputs, or install verified full-weight linear methods."""

    def __init__(self, model, config):
        self.config = config
        self.root = model
        self.refiner = model.token_refiner
        self.stack = []
        self.mode = config["mode"]
        if self.mode not in {"bf16", "collect", "kernel"}:
            raise ValueError("Unknown portable PTQ mode")
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if self.world != config["world_size"]:
            raise ValueError("Actual process count differs from launch contract")
        check_files(config["files"])
        self.directory = Path(config["output"])
        self.audit = self.directory / f"forwards-rank{self.rank}.jsonl"
        if self.audit.exists():
            raise FileExistsError(self.audit)
        selected = set(h3_layer_names())
        self.modules = {name: module for name, module in model.named_modules() if name in selected}
        shapes = validate_order(list(self.modules))
        for name, module in self.modules.items():
            if (
                tuple(module.weight.shape) != shapes[name]
                or module.weight.dtype != torch.bfloat16
                or getattr(module, "tp_size", 1) != 1
                or not hasattr(module, "quant_method")
            ):
                raise ValueError("Only original unsharded BF16 SGLang linears supported: " + name)
        self.collectors, self.handles = {}, []
        self.count = 0
        self.prepared = False
        self.lock = threading.Lock()
        self.owner = None
        self.failed = False
        self.skipped_finish = {}
        self.handles.append(model.register_forward_pre_hook(self.begin))
        self.handles.append(model.register_forward_hook(self.finish, always_call=True))
        # SGLang can refine request-static text before the first DiT forward.
        self.handles.append(self.refiner.register_forward_pre_hook(self.begin))
        self.handles.append(self.refiner.register_forward_hook(self.finish, always_call=True))

    def begin(self, model, args):
        """Reject concurrency and initialize tensors only after weights are loaded."""
        del args
        if (
            self.owner == threading.get_ident()
            and self.stack == [self.root]
            and model is self.refiner
        ):
            self.stack.append(model)
            return
        if not self.lock.acquire(blocking=False):
            thread = threading.get_ident()
            self.skipped_finish[thread] = self.skipped_finish.get(thread, 0) + 1
            raise RuntimeError("Concurrent/reentrant H3 execution is unsupported")
        self.owner = threading.get_ident()
        self.stack.append(model)
        try:
            if self.failed:
                raise RuntimeError("Previous forward failed; restart with fresh output")
            if self.mode == "collect" and self.count >= self.config["expected_forwards"]:
                raise ValueError("Calibration complete; restart server before other requests")
            if not self.prepared:
                if self.mode == "collect":
                    for name, module in self.modules.items():
                        collector = ChannelMax(module.weight.shape[1], module.weight.device)
                        self.collectors[name] = collector
                        self.handles.append(
                            module.register_forward_pre_hook(
                                lambda _module, inputs, collector=collector: collector.update(
                                    inputs[0]
                                )
                            )
                        )
                elif self.mode == "kernel":
                    manifest = self.config["artifact_manifest"]
                    load_runtime(self.config["library"], manifest["runtime"]["library_sha256"])
                    methods = {}
                    for name, module in self.modules.items():
                        entry = manifest["layers"][name]
                        path = local_file(self.config["artifact"], entry["file"])
                        if digest(path) != entry["sha256"]:
                            raise ValueError("Packed layer changed before load")
                        methods[name] = PackedLinear(
                            load_file(str(path), device=str(module.weight.device))
                        )
                    check_files(self.config["files"])
                    for name, method in methods.items():
                        self.modules[name].quant_method = method
                self.prepared = True
        except BaseException:
            self.failed = True
            self.stack.clear()
            self.owner = None
            self.lock.release()
            raise

    def finish(self, model, args, output):
        """Persist only successful forwards, releasing the lock also on failure."""
        del args
        thread = threading.get_ident()
        if self.skipped_finish.get(thread, 0):
            self.skipped_finish[thread] -= 1
            return
        if self.owner != threading.get_ident():
            return
        try:
            if output is None:
                self.failed = True
                raise RuntimeError("H3 forward failed; calibration is incomplete")
            if model is self.refiner:
                return
            self.count += 1
            if self.mode == "collect" and self.count == self.config["expected_forwards"]:
                save_rank(
                    self.directory,
                    self.collectors,
                    rank=self.rank,
                    world_size=self.world,
                    forwards=self.count,
                    source=self.config["source"],
                    request_sha256=self.config["requests_sha256"],
                    media_sha256=self.config["requests_media_sha256"],
                )
            with self.audit.open("a") as stream:
                stream.write(json.dumps({"forward": self.count, "mode": self.mode}) + "\n")
        except BaseException:
            self.failed = True
            raise
        finally:
            self.stack.pop()
            if not self.stack:
                self.owner = None
                self.lock.release()


def install(model):
    """Called once by the optional SGLang post_load_weights integration."""
    path = os.environ.get("H3_MODELOPT_PTQ_CONFIG")
    if not path:
        return
    if hasattr(model, "_modelopt_ptq_controller"):
        raise ValueError("Controller already installed")
    config = read_json(path)
    if config.get("format") != "h3-modelopt-portable-launch-v1":
        raise ValueError("Invalid launch contract")
    model._modelopt_ptq_controller = Controller(model, config)
