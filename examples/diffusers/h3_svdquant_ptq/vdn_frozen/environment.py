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

"""Explicit local inputs and source bindings, without archived experiment receipts."""

import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from portable_io import digest as sha
from portable_io import read_json as read
from portable_io import write_json as write
from portable_runtime import PackedLinear, load_runtime

__all__ = ["Method", "build", "identity", "initialize", "native_modules", "read", "sha", "write"]


class Method(PackedLinear):
    """Expose the unchanged runtime rank-down features to offline B fitting."""

    @property
    def lora_down(self):
        """Actual BF16 runtime features, not a full-precision surrogate."""
        return self.down


def source_files(config):
    """Inventory checkpoint, base model and VDN Python/config content."""
    result = {}
    for label in ("base", "checkpoint"):
        root = Path(config[label])
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix in {".safetensors", ".json", ".yaml", ".yml", ".pt", ".bin"}
        )
        if not files or not any(p.suffix in {".safetensors", ".pt", ".bin"} for p in files):
            raise ValueError(f"No model weights in {label}")
        for index, path in enumerate(files):
            print(f"PHASE source integrity {label} {index + 1}/{len(files)}", flush=True)
            result[f"{label}/{path.relative_to(root)}"] = sha(path)
    vdn = Path(config["vdn_root"])
    for path in sorted((vdn / "src").rglob("*.py")):
        result[f"vdn/{path.relative_to(vdn)}"] = sha(path)
    cfg = vdn / "configs/inference/8nfe.yaml"
    result["vdn/configs/inference/8nfe.yaml"] = sha(cfg)
    spec = importlib.util.find_spec("diffusers")
    if spec is None or spec.origin is None:
        raise ValueError("Install the pinned Diffusers checkout with the VDN AdaLN patch")
    diffusers_root = Path(spec.origin).parent
    for path in sorted(diffusers_root.rglob("*.py")):
        result[f"diffusers/{path.relative_to(diffusers_root)}"] = sha(path)
    return result


def initialize(args):
    """Create a private run inventory before allocating a GPU."""
    config: dict[str, Any] = {
        name: str(getattr(args, name).resolve(strict=True))
        for name in ("base", "checkpoint", "vdn_root", "library", "fit_cache", "dev_cache")
    }
    config.update(
        fit_seed=args.fit_seed,
        dev_seed=args.dev_seed,
        library_sha256=args.library_sha256,
        width=768,
        height=1344,
        frames=124,
        nfe=8,
        video_shift=12.0,
        audio_shift=3.0,
    )
    if sha(config["library"]) != args.library_sha256:
        raise ValueError("Kernel library does not match supplied SHA256")
    prompts = [sha(config[name]) for name in ("fit_cache", "dev_cache")]
    if prompts[0] == prompts[1]:
        raise ValueError("Use different fitting and development prompt caches")
    print("PHASE hash local source weights and VDN code (CPU)", flush=True)
    inventory = source_files(config)
    args.run.mkdir(parents=True, exist_ok=False)
    write(args.run / "config.json", config)
    write(args.run / "source.json", {"files": inventory, "prompt_sha256": prompts})


def identity(run):
    """Revalidate actual inputs; never accept a copied success receipt."""
    config, source = read(run / "config.json"), read(run / "source.json")
    if source_files(config) != source["files"]:
        raise ValueError("Source weights or VDN code changed; initialize a new run")
    if [sha(config[name]) for name in ("fit_cache", "dev_cache")] != source["prompt_sha256"]:
        raise ValueError("Fitting/development prompt caches changed")
    if sha(config["library"]) != config["library_sha256"]:
        raise ValueError("Kernel changed")
    files = sorted(Path(__file__).parent.glob("*.py"))
    helpers = {}
    for name in (
        "h3_modelopt_ptq",
        "modelopt_multiplier_compat",
        "recipe_comparison",
        "compat_rcp_triton",
        "portable_runtime",
    ):
        module = importlib.import_module(name)
        helpers[name] = sha(module.__file__)
    return {
        "config_sha256": sha(run / "config.json"),
        "source_sha256": sha(run / "source.json"),
        "code_sha256": {p.name: sha(p) for p in files},
        "helpers": helpers,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "triton",
                "flash-attn-4",
            )
        },
    }


def native_modules(config):
    """Import the explicitly supplied trusted VDN checkout, not an implicit clone."""
    root = Path(config["vdn_root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # VDN/diffusers are optional GPU integration dependencies, not CPU-test dependencies.
    native = importlib.import_module("src.inference.render")
    if native.__file__ is None or not Path(native.__file__).resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("Another src package shadows the supplied VDN checkout")
    prompt = importlib.import_module("src.inference.utils.prompt_cache")
    native.LATENT_H, native.LATENT_W = 84, 48
    return native, prompt.load_prompt


def build(config, *, decoders=False):
    """Assemble the BF16 hybrid with native checkpoint LoRAs merged before PTQ."""
    native_modules(config)
    # Optional model-loading dependencies are deliberately deferred until GPU execution.
    from omegaconf import OmegaConf
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model

    load_runtime(Path(config["library"]), config["library_sha256"])
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = OmegaConf.merge(
        OmegaConf.structured(InferenceConfig),
        OmegaConf.load(Path(config["vdn_root"]) / "configs/inference/8nfe.yaml"),
    )
    cfg.checkpoint = config["checkpoint"]
    cfg.base_source = cfg.vae_source = config["base"]
    cfg.render.num_frames, cfg.render.num_steps = 124, 8
    cfg.render.video_shift, cfg.render.audio_shift = 12.0, 3.0
    model = build_inference_model(cfg, "cuda", load_decoders=decoders)
    if not model.is_hybrid or model.fp8_linears or model.merged_lora_pairs < 1:
        raise ValueError("Requires merged BF16 VDN hybrid; not original Ref2VA or FP8 Linear")
    return model
