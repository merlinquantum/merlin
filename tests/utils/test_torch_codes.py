# MIT License
#
# Copyright (c) 2025 Quandela
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

import torch

from merlin.core.generators import CircuitType
from merlin.utils import FeatureEncoder


def _run_partial_test(
    n_modes, num_features, circuit_type, bandwidth_coeffs, expected_res
):
    batch_size = 2
    total_shifters = None

    encoder = FeatureEncoder(num_features)
    X_norm = torch.rand((batch_size, num_features))
    res = encoder.encode(
        X_norm, circuit_type, n_modes, bandwidth_coeffs, total_shifters
    )

    assert res.shape == expected_res.shape, (
        "partial_test: num_features: "
        + str(num_features)
        + " ; circuit type: "
        + str(circuit_type)
        + ":\n"
        f"shapes do not match :\n"
        f"expected : {expected_res.shape}\n"
        f"calculated  : {res.shape}"
        f"Tensors values :"
        f"expected : {expected_res}\n"
        f"calculated  : {res}"
    )

    assert torch.allclose(res, expected_res, atol=1e-4), (
        "partial_test: num_features: "
        + str(num_features)
        + "; circuit type: "
        + str(circuit_type)
        + ":\n"
        f"values do not match :\n"
        f"encoded features do not match :\n"
        f"expected : {expected_res}\n"
        f"calculated  : {res}"
    )


def test_torch_codes():
    """Test torch_codes"""

    n_modes = 4
    seed = 42
    torch.manual_seed(seed)

    num_features = 1
    circuit_type = CircuitType.SERIES
    bandwidth_coeffs = {i: (1.0 + i / n_modes) for i in range(n_modes)}
    expected_res = torch.tensor([[2.7717, 5.5435, 8.3152], [2.8746, 5.7491, 8.6237]])
    _run_partial_test(
        n_modes, num_features, circuit_type, bandwidth_coeffs, expected_res
    )

    num_features = 3
    circuit_type = CircuitType.SERIES
    bandwidth_coeffs = None
    expected_res = torch.tensor([[1.2028, 3.0137, 4.2165], [1.8878, 0.8060, 2.6938]])
    _run_partial_test(
        n_modes, num_features, circuit_type, bandwidth_coeffs, expected_res
    )

    num_features = 1
    circuit_type = CircuitType.PARALLEL
    expected_res = torch.tensor([[2.9555, 2.9555, 2.9555], [0.4184, 0.4184, 0.4184]])
    _run_partial_test(
        n_modes, num_features, circuit_type, bandwidth_coeffs, expected_res
    )

    num_features = 3
    circuit_type = CircuitType.PARALLEL
    expected_res = torch.tensor([[2.9361, 1.8648, 2.7313], [1.7835, 2.3282, 1.3490]])
    _run_partial_test(
        n_modes, num_features, circuit_type, bandwidth_coeffs, expected_res
    )

    num_features = 3
    circuit_type = CircuitType.PARALLEL_COLUMNS
    expected_res = torch.tensor([
        [
            8.7390,
            17.4779,
            26.2169,
            34.9559,
            5.6642,
            11.3284,
            16.9926,
            22.6568,
            2.6310,
            5.2621,
            7.8931,
            10.5242,
        ],
        [
            6.1927,
            12.3854,
            18.5780,
            24.7707,
            2.6612,
            5.3223,
            7.9835,
            10.6446,
            4.3561,
            8.7122,
            13.0683,
            17.4243,
        ],
    ])
    _run_partial_test(
        n_modes, num_features, circuit_type, bandwidth_coeffs, expected_res
    )
