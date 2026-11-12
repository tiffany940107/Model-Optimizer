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

"""Kernel-fixed B fitting, block gates and complete quantized trajectories."""

from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from .block_replay import replay, tensor_versions
from .context import Context
from .fit import (
    Statistics,
    improves,
    losses,
    mean_losses,
    replace_b,
    sample_indices,
    solve_delta,
    updated_b,
)
from .runtime import bf16_group

__all__ = ["run"]


def run(run_dir, output):
    args = SimpleNamespace(output=output)
    ctx = Context(run_dir, output)
    stats = {
        name: Statistics(group.shape[0], "cuda")
        for name, group in ((g.name, g) for g in ctx.groups)
    }
    samples = {name: [] for name in ctx.selected}
    handles = []

    def sampled_input(name, value):
        flat = value.reshape(-1, value.shape[-1])
        refiner = name.startswith("token_refiner.")
        if not refiner and flat.shape[0] != ctx.sequence_rows:
            raise ValueError("Packed sequence row order/geometry differs")
        indices, segments = sample_indices(flat.shape[0], ctx.modalities, refiner=refiner)
        return flat.index_select(0, indices).contiguous(), segments

    for name, layers in ctx.selected.items():

        def observe(_m, inputs, name=name, layers=layers):
            x, segments = sampled_input(name, inputs[0])
            teacher = bf16_group(layers, x)
            prediction = ctx.methods[name].apply(None, x)
            down = torch.mm(x, ctx.methods[name].lora_down)
            stats[name].add(down, teacher.float() - prediction.float(), teacher, segments)
            samples[name].append((x.cpu(), teacher.cpu(), segments))

        handles.append(layers[0].register_forward_pre_hook(observe))
    ctx.generate("fit-teacher", 0, reference=ctx.reference(0), exact=True)
    for handle in handles:
        handle.remove()
    if any(s.calls != 8 for s in stats.values()) or any(len(row) != 8 for row in samples.values()):
        raise ValueError("Fit coverage differs")
    save_file(
        {
            f"{name}.{key}": value.cpu().contiguous()
            for name, s in stats.items()
            for key, value in (("H", s.h), ("G", s.g))
        },
        str(args.output / "normal_equations.safetensors"),
    )
    candidates, records = {}, {}
    for ordinal, group in enumerate(ctx.groups):
        name = group.name
        print(f"PHASE fit/candidate real-kernel check {ordinal + 1}/208 {name}", flush=True)
        data = [(x.cuda(), y.cuda(), segments) for x, y, segments in samples[name]]
        original = ctx.payloads[name]["svdquant_lora_b"]

        def score(b):
            method = ctx.Method(replace_b(ctx.payloads[name], b))
            return mean_losses(
                [losses(method.apply(None, x), y, segments) for x, y, segments in data]
            )

        baseline = score(original)
        best, best_loss, best_spec = original, baseline, None
        trials = []
        for ridge in (0.01, 0.1, 1.0):
            delta = solve_delta(stats[name].h, stats[name].g, ridge)
            for shrink in (0.25, 0.5, 1.0):
                b = updated_b(original, delta, shrink)
                candidate_loss = score(b)
                trials.append({"ridge": ridge, "shrink": shrink, "loss": candidate_loss})
                if improves(candidate_loss, baseline) and sum(candidate_loss.values()) < sum(
                    best_loss.values()
                ):
                    best, best_loss, best_spec = (
                        b,
                        candidate_loss,
                        {"ridge": ridge, "shrink": shrink},
                    )
        if best_spec is not None:
            method = ctx.Method(replace_b(ctx.payloads[name], best))
            first = method.apply(None, data[0][0])
            if not torch.equal(first, method.apply(None, data[0][0])):
                raise ValueError("Candidate kernel repeat changed")
            candidates[name] = best.cpu().contiguous()
        records[name] = {
            "train_baseline": baseline,
            "train_selected": best_loss,
            "selected_parameters": best_spec,
            "relative_B_change": float(
                (best.float() - original.float()).norm() / original.float().norm().clamp_min(1e-30)
            ),
            "grid": trials,
        }
        data.clear()
        samples[name].clear()
    samples.clear()
    stats.clear()
    if candidates:
        save_file(candidates, str(args.output / "train_B.safetensors"))
    ctx.write(
        args.output / "fit.json",
        {
            "groups": records,
            "candidate_count": len(candidates),
            "normal_equations_sha256": ctx.sha(args.output / "normal_equations.safetensors"),
            "train_B_sha256": ctx.sha(args.output / "train_B.safetensors") if candidates else None,
        },
    )
    validation = {name: {"baseline": [], "candidate": []} for name in candidates}
    candidate_methods = {
        name: ctx.Method(replace_b(ctx.payloads[name], value.cuda()))
        for name, value in candidates.items()
    }
    handles = []
    for name in candidates:
        layers = ctx.selected[name]

        def observe(_m, inputs, name=name, layers=layers):
            x, segments = sampled_input(name, inputs[0])
            teacher = bf16_group(layers, x)
            validation[name]["baseline"].append(
                losses(ctx.methods[name].apply(None, x), teacher, segments)
            )
            validation[name]["candidate"].append(
                losses(candidate_methods[name].apply(None, x), teacher, segments)
            )

        handles.append(layers[0].register_forward_pre_hook(observe))
    ctx.generate("development-teacher", 1, reference=ctx.reference(1), exact=True)
    for handle in handles:
        handle.remove()
    accepted = {}
    validation_report = {}
    for name, row in validation.items():
        if any(len(rows) != 8 for rows in row.values()):
            raise ValueError("Development validation coverage differs")
        before, after = mean_losses(row["baseline"]), mean_losses(row["candidate"])
        passed = improves(after, before)
        validation_report[name] = {"baseline": before, "candidate": after, "passed": passed}
        if passed:
            accepted[name] = candidates[name]
    ctx.write(
        args.output / "group_validation.json",
        {
            "groups": validation_report,
            "accepted": len(accepted),
            "scope": "Development selection, not independent held-out quality",
        },
    )
    candidate_methods.clear()
    if not accepted:
        ctx.write(
            args.output / "complete.json",
            {
                **ctx.identity,
                "status": "no-group-candidate-passed",
                "fit_sha256": ctx.sha(args.output / "fit.json"),
                "validation_sha256": ctx.sha(args.output / "group_validation.json"),
                "accepted_groups": 0,
                "export_accepted": False,
                "scope": "This bounded B-only fit failed; not a general impossibility claim",
            },
        )
        return
    save_file(accepted, str(args.output / "group_B.safetensors"))
    block_groups = {}
    for group in ctx.groups:
        family = ".".join(group.name.split(".")[:-2])
        block_groups.setdefault(family, {})[group.name] = ctx.selected[group.name]
    blocks = {
        f"blocks.{i}": block for i, block in enumerate(ctx.model.transformer.transformer_blocks)
    }
    blocks.update(
        {
            f"token_refiner.blocks.{i}": block
            for i, block in enumerate(ctx.model.transformer.token_refiner.refiner_blocks)
        }
    )
    scores = {
        name: {"baseline": [], "candidate": []}
        for name, selected in block_groups.items()
        if set(selected) & set(accepted)
    }
    handles = []
    for name in scores:
        selected = block_groups[name]
        original_methods = {key: ctx.methods[key] for key in selected}
        fitted_methods = {
            key: ctx.Method(replace_b(ctx.payloads[key], accepted[key].cuda()))
            if key in accepted
            else ctx.methods[key]
            for key in selected
        }

        def observe_block(
            block,
            inputs,
            kwargs,
            teacher,
            name=name,
            selected=selected,
            original_methods=original_methods,
            fitted_methods=fitted_methods,
        ):
            if ctx.step not in (0, 3, 7):
                return
            print(f"PHASE block gate step={ctx.step} block={name}", flush=True)
            versions = tensor_versions((inputs, kwargs))
            repeated = block.forward(*inputs, **kwargs)
            if not torch.equal(repeated, teacher) or any(
                value._version != version for value, version in versions
            ):
                raise ValueError("BF16 block replay/input immutability failed")
            rows = (
                {"text": torch.arange(teacher.shape[1], device=teacher.device)}
                if name.startswith("token_refiner.")
                else ctx.modalities
            )
            scores[name]["baseline"].append(
                replay(block, inputs, kwargs, selected, original_methods, teacher, rows)
            )
            scores[name]["candidate"].append(
                replay(block, inputs, kwargs, selected, fitted_methods, teacher, rows)
            )

        handles.append(blocks[name].register_forward_hook(observe_block, with_kwargs=True))
    ctx.generate("block-gate-teacher", 1, reference=ctx.reference(1), exact=True)
    for handle in handles:
        handle.remove()
    block_accepted = {}
    block_report = {}
    for name, row in scores.items():
        if any(len(rows) != 3 for rows in row.values()):
            raise ValueError("Block gate rung coverage differs")
        before, after = mean_losses(row["baseline"]), mean_losses(row["candidate"])
        passed = improves(after, before)
        block_report[name] = {"baseline": before, "candidate": after, "passed": passed}
        if passed:
            block_accepted.update(
                {key: accepted[key] for key in block_groups[name] if key in accepted}
            )
    ctx.write(
        args.output / "block_validation.json",
        {
            "blocks": block_report,
            "accepted_groups": len(block_accepted),
            "scope": "Matched full-sequence block outputs, actual kernel; development selection, not rollout evidence",
        },
    )
    if block_accepted:
        save_file(block_accepted, str(args.output / "block_B.safetensors"))
    ctx.activate({})
    original_values, _ = ctx.generate("original-nvfp4", 1)
    ctx.generate("original-nvfp4-repeat", 1, reference=original_values, exact=True)
    reference = ctx.reference(1)
    outcomes = []
    for label, updates in (("group-fit", accepted), ("block-fit", block_accepted)):
        if not updates:
            continue
        if label == "block-fit" and set(updates) == set(accepted):
            continue
        ctx.activate(updates)
        _, result = ctx.generate(label, 1, reference=reference)
        outcomes.append(result)
    ctx.write(
        args.output / "complete.json",
        {
            **ctx.identity,
            "status": "B-fit-screening-completed-export-not-yet-accepted",
            "fit_sha256": ctx.sha(args.output / "fit.json"),
            "validation_sha256": ctx.sha(args.output / "group_validation.json"),
            "block_validation_sha256": ctx.sha(args.output / "block_validation.json"),
            "group_accepted": len(accepted),
            "block_accepted": len(block_accepted),
            "group_B_sha256": ctx.sha(args.output / "group_B.safetensors"),
            "block_B_sha256": ctx.sha(args.output / "block_B.safetensors")
            if block_accepted
            else None,
            "outcomes": [
                {key: value for key, value in row.items() if key != "counts"} for row in outcomes
            ],
            "export_accepted": False,
            "scope": (
                "Real-kernel B-only screening; ModelOpt roundtrip/export "
                "and independent expanded evaluation remain mandatory"
            ),
        },
    )
