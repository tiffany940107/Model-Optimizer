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

"""CPU coverage of representative schedule and matrix inventory."""

import pytest
from h3_modelopt_ptq import h3_layer_names
from regeneration_planning import layer_storage, representative_schedule


def test_representatives_keep_absolute_export_order_and_seed():
    names = list(reversed(h3_layer_names()))
    selected = representative_schedule(names, seed=1101)
    assert len(selected) == 4
    assert (
        len({name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1] for _, name, _ in selected})
        == 4
    )
    assert [index for index, _, _ in selected] == sorted(index for index, _, _ in selected)
    for index, name, seed in selected:
        assert names[index] == name
        assert seed == 1101 + index


def test_missing_projection_type_rejected():
    with pytest.raises(ValueError, match="Missing"):
        representative_schedule(
            [name for name in h3_layer_names() if not name.endswith(".mlp.fc2")]
        )


def test_largest_h3_projection_and_padding_inventory():
    inventory = [layer_storage(name) for name in h3_layer_names()]
    assert len(inventory) == 208
    assert max(row["dense_bf16_bytes"] for row in inventory) == 308281344
    assert all(row["no_unused_scale_padding"] for row in inventory)
