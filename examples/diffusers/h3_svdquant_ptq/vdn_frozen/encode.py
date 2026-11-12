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

"""Create private portrait three-reference caches using VDN's public helpers."""

import argparse
from pathlib import Path

import torch

from .environment import native_modules, sha, write

__all__ = ["main"]


def main():
    """Encode user-supplied text and ordered images; never download inputs implicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vdn-root", "text-source", "base", "prompt-file", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--refs", type=Path, nargs=3, required=True)
    args = parser.parse_args()
    native_modules({"vdn_root": str(args.vdn_root.resolve(strict=True))})
    # Optional large model dependencies, deferred until encoding rather than --help.
    import numpy as np
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from PIL import Image
    from src.inference.encode_keyframes import (
        build_presentation,
        normalize_references,
        qwen3vl_prompt_embeds,
    )
    from src.inference.render import PIXEL_MEAN, PIXEL_STD
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    args.output.mkdir(parents=True, exist_ok=False)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("Empty prompt")
    images = []
    for path in args.refs:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    images = normalize_references(images, 768)
    torch.set_grad_enabled(False)
    processor = Qwen3VLProcessor.from_pretrained(
        args.text_source, subfolder="processor", local_files_only=True
    )
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.text_source, subfolder="text_encoder", dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    encoder.eval().requires_grad_(False)
    ids, tags, vision = build_presentation(processor, prompt, images)
    embeds = qwen3vl_prompt_embeds(encoder, processor, ids, vision, "cuda")
    del encoder
    torch.cuda.empty_cache()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.base, subfolder="vae", local_files_only=True
    ).to("cuda")
    vae.eval().requires_grad_(False)
    latents = []
    for image in images:
        value = torch.from_numpy(np.array(image)).to("cuda").permute(2, 0, 1)[None, :, None]
        latents.append(encode_vae_condition(vae, value, PIXEL_MEAN, PIXEL_STD, 42).cpu())
    target = args.output / "prompt.pt"
    torch.save(
        {
            "prompt": prompt,
            "prompt_embeds": embeds.cpu(),
            "text_token_tags": torch.tensor(tags, dtype=torch.long),
            "keyframe_anchors": ["ref"] * 3,
            "condition_latents": latents,
            "height": 1344,
            "width": 768,
            "reference_size": 768,
        },
        target,
    )
    write(
        args.output / "complete.json",
        {
            "prompt_cache_sha256": sha(target),
            "prompt_file_sha256": sha(args.prompt_file),
            "references_sha256": [sha(p) for p in args.refs],
            "scope": "User-supplied private conditioning; not shipped with the code",
        },
    )


if __name__ == "__main__":
    main()
