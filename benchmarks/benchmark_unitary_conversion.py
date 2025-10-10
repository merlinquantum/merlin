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

import random

import perceval as pcvl
import pytest
import torch

from merlin.pcvl_pytorch import CircuitConverter


class UnitaryConversionBenchmarkRunner:
    """Utility/validation for unitary conversion benchmarks."""

    def __init__(self):
        self.results = []

    def validate_converter(
        self, converter: CircuitConverter, expect_list_rct: int | None = None
    ) -> bool:
        if expect_list_rct is not None:
            assert len(converter.list_rct) == expect_list_rct
        return True


runner = UnitaryConversionBenchmarkRunner()


def _generic_interferometer_weightparamweight(circuit_size):
    wl = pcvl.GenericInterferometer(
        circuit_size,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_li{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        depth=circuit_size // 2,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    c_var = pcvl.Circuit(circuit_size)
    for i in range(circuit_size):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i, pcvl.PS(px))
    wr = pcvl.GenericInterferometer(
        circuit_size,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_ri{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"theta_ro{i}")),
        depth=circuit_size // 2,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    return wl // c_var // wr


def prep_fullparameterized_inteferometer(circuit_size: int) -> pcvl.Circuit:
    return _generic_interferometer_weightparamweight(circuit_size)


def prep_quantumreservoir_inteferometer(circuit_size: int) -> pcvl.Circuit:
    circ = _generic_interferometer_weightparamweight(circuit_size)
    for param in circ.get_parameters():
        if param.name.startswith("px"):
            continue
        param.fix_value(random.uniform(0, 1))
    return circ


def prep_noparameterized_inteferometer(circuit_size: int) -> pcvl.Circuit:
    circ = _generic_interferometer_weightparamweight(circuit_size)
    for param in circ.get_parameters():
        param.fix_value(random.uniform(0, 1))
    assert len(circ.get_parameters()) == 0
    return circ


def build_torchunitary(
    converter: CircuitConverter, nparams: int
) -> torch.Tensor | None:
    if nparams == 0:
        return converter.to_tensor([])
    else:
        return converter.to_tensor(torch.rand(nparams))


# Configs and devices
BENCH_NMODE = [10, 20, 40, 100]
DEVICE_CONFIGS = ["cpu"]


@pytest.mark.parametrize("nmode", BENCH_NMODE)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_fullparameter_benchmark(benchmark, nmode: int, device: str):
    circ = prep_fullparameterized_inteferometer(nmode)
    converter = CircuitConverter(circ, [""], dtype=torch.float)
    _res = benchmark(build_torchunitary, converter, nparams=len(circ.get_parameters()))
    assert runner.validate_converter(converter)


@pytest.mark.parametrize("nmode", BENCH_NMODE)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_quantumreservoir_benchmark(benchmark, nmode: int, device: str):
    circ = prep_quantumreservoir_inteferometer(nmode)
    converter = CircuitConverter(circ, [""], dtype=torch.float)
    _res = benchmark(build_torchunitary, converter, nparams=len(circ.get_parameters()))
    assert runner.validate_converter(converter)


@pytest.mark.parametrize("nmode", BENCH_NMODE)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_noparameter_benchmark(benchmark, nmode: int, device: str):
    circ = prep_noparameterized_inteferometer(nmode)
    converter = CircuitConverter(circ, [], dtype=torch.float)
    # when there is no parameter, we should have a single precomputed unitary
    assert runner.validate_converter(converter, expect_list_rct=1)
    _res = benchmark(build_torchunitary, converter, nparams=len(circ.get_parameters()))


if __name__ == "__main__":
    import pytest as _pytest

    print("Running unitary conversion benchmarks...")
    code = _pytest.main([__file__, "-v", "--benchmark-only"])  # quick local run
    raise SystemExit(code)
