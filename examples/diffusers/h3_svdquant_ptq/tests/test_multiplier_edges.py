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

"""CPU checks for edge input construction; GPU exactness is tested separately."""

import pytest
import torch
from modelopt_multiplier_compat import MultiplierNVFP4QTensor
from scripts.check_multiplier_edges import edge_inputs


def test_edge_inputs_are_repeatable_without_mutating_default_rng():
    before = torch.random.get_rng_state().clone()
    first, second = edge_inputs(), edge_inputs()
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(first) == 10
    for (name, x, g, rejection), (other_name, other_x, other_g, other_rejection) in zip(
        first, second
    ):
        assert name == other_name and rejection == other_rejection
        assert torch.equal(x.view(torch.int16), other_x.view(torch.int16))
        assert torch.equal(g, other_g)
        assert x.shape == (128, 64) and x.dtype == torch.bfloat16


def test_edge_nonzero_scale_underflow_cannot_be_reported_as_compatible():
    _, x, g, rejection = next(row for row in edge_inputs() if row[0] == "nonzero_scale_underflow")
    with pytest.raises(ValueError, match=rejection):
        MultiplierNVFP4QTensor.quantize(x, g, variant="nested_rcp_rn")
