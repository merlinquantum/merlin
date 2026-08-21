# MIT License
#
# Copyright (c) 2026 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression tests for occupancy-readout mass handling.

These tests pin down the behaviour described in the "occupancy readout
silently renormalizes" bug: :class:`_OccupancyReadout` used to rescale
grouped probabilities back up to sum to 1, while the non-occupancy
``probs()`` path (``DistributionStrategy.process``) never renormalizes.
Under photon loss / detector mass-loss, the two paths would then disagree
about the total probability mass of an otherwise-identical distribution,
with no signal to the user. The fix makes occupancy grouping a
mass-preserving column sum, matching the non-occupancy path.
"""

import torch

from merlin.measurement.readouts import _OccupancyReadout


class TestOccupancyReadoutMassPreservation:
    def test_lossy_input_mass_is_preserved_not_renormalized(self):
        """Grouped probabilities should keep whatever total mass was present.

        This mimics photon loss / detector mass-loss upstream: the incoming
        probability tensor no longer sums to 1. The occupancy readout must
        not silently rescale it back up to 1, since the non-occupancy
        probs() path never does that either.
        """
        output_keys = [(0, 0), (1, 0), (2, 0), (0, 1)]
        readout = _OccupancyReadout(output_keys)

        # Row sums to 0.5, simulating upstream mass loss.
        probabilities = torch.tensor([[0.10, 0.20, 0.15, 0.05]])

        grouped = readout(probabilities)

        input_mass = probabilities.sum(dim=-1)
        output_mass = grouped.sum(dim=-1)
        assert torch.allclose(output_mass, input_mass, atol=1e-6)
        assert not torch.allclose(output_mass, torch.ones_like(output_mass))

    def test_grouped_values_match_column_sum_reference(self):
        """Occupancy grouping is a plain column sum, with no division step."""
        output_keys = [(0, 0), (1, 0), (2, 0), (0, 1)]
        readout = _OccupancyReadout(output_keys)

        probabilities = torch.tensor([
            [0.10, 0.20, 0.15, 0.05],
            [0.02, 0.03, 0.01, 0.04],
        ])

        grouped = readout(probabilities)
        expected = torch.tensor([
            [0.10, 0.05, 0.35],
            [0.02, 0.04, 0.04],
        ])

        assert torch.allclose(grouped, expected, atol=1e-6)

    def test_fully_normalized_input_still_sums_to_one(self):
        """Sanity check: mass-preserving grouping of a normalized input stays normalized."""
        output_keys = [(0, 0), (1, 0), (2, 0), (0, 1)]
        readout = _OccupancyReadout(output_keys)

        probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        grouped = readout(probabilities)

        assert torch.allclose(grouped.sum(dim=-1), torch.ones(1), atol=1e-6)

    def test_1d_lossy_input_mass_is_preserved(self):
        """The unbatched (1-D) code path must also skip renormalization."""
        output_keys = [(0, 0), (1, 0), (2, 0), (0, 1)]
        readout = _OccupancyReadout(output_keys)

        probabilities = torch.tensor([0.10, 0.20, 0.15, 0.05])
        grouped = readout(probabilities)

        assert torch.allclose(grouped.sum(), probabilities.sum(), atol=1e-6)

    def test_zero_mass_row_stays_zero(self):
        """A fully-lost row (all zeros) should not divide-by-zero or blow up."""
        output_keys = [(0, 0), (1, 0), (2, 0), (0, 1)]
        readout = _OccupancyReadout(output_keys)

        probabilities = torch.zeros(1, 4)
        grouped = readout(probabilities)

        assert torch.allclose(grouped, torch.zeros(1, len(readout.output_keys)))
