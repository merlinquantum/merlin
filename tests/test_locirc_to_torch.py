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
import numpy as np
import pytest
from perceval.components import Circuit, PS, BS, PERM, Unitary
from perceval.utils import Parameter, Matrix
from merlin.pcvl_pytorch import CircuitConverter


def test_ps_to_torch():
    c_ps = Circuit(1) // PS(Parameter("x"))
    params = {"x": torch.tensor([0.1], requires_grad=True)}

    torch_conv = CircuitConverter(c_ps, input_specs=list(params.keys()))
    torch_tensor = torch_conv.to_tensor(params["x"])

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 2

    torch_tensor.real.backward()  # compute gradient with torch
    expected_grad = np.real(1j*np.exp(1j*params["x"].item()))  # analytical

    assert expected_grad == params['x'].grad


@pytest.mark.parametrize("circuit_size", [2, 3, 5])
def test_2ps_circ_to_torch(circuit_size):
    c_ps = Circuit(circuit_size) // PS(Parameter("x")) // (1, PS(Parameter("y")))
    params = {"x": torch.tensor([0.1], requires_grad=True), "y": torch.tensor([0.2], requires_grad=True)}

    torch_conv = CircuitConverter(c_ps, input_specs=list(params.keys()))
    torch_tensor = torch_conv.to_tensor(list(params.values()))

    assert torch_tensor.shape == torch.Size(2*[c_ps.m])

    loss = torch_tensor.real.sum()
    loss.backward()
    expected_gradx = np.real(1j * np.exp(1j * params["x"].item()))
    expected_grady = np.real(1j * np.exp(1j * params["y"].item()))

    assert params['x'].grad == expected_gradx
    assert params['y'].grad == expected_grady

@pytest.mark.parametrize("circuit_size", [2, 3, 5])
def test_2ps_1tensor_circ_to_torch(circuit_size):
    c_ps = Circuit(circuit_size) // PS(Parameter("px")) // (1, PS(Parameter("py")))
    tensor_p = torch.tensor([0.1, 0.2], requires_grad=True)

    torch_conv = CircuitConverter(c_ps, input_specs=["p"])
    torch_tensor = torch_conv.to_tensor([tensor_p])

    assert torch_tensor.shape == torch.Size(2*[c_ps.m])

    loss = torch_tensor.real.sum()
    loss.backward()
    expected_gradx = np.real(1j * np.exp(1j * float(tensor_p[0])))
    expected_grady = np.real(1j * np.exp(1j * float(tensor_p[1])))

    assert torch.allclose(tensor_p.grad, torch.tensor([expected_gradx, expected_grady], dtype=torch.float32))


def test_incorrect_input_param():
    c_ps = Circuit(2) // PS(Parameter("x")) // (1, PS(Parameter("y")))
    torch_1_tensor = torch.tensor([0.1], requires_grad=True)
    torch_2_tensor = torch.tensor([0.1, 0.2], requires_grad=True)
    params = {"x": torch_1_tensor}

    torch_conv = CircuitConverter(c_ps, [""])
    with pytest.raises(AttributeError):
        # does not expect a dict but a list of tensors
        torch_conv.to_tensor({"x": torch_1_tensor})

    # expect a list of tensors, or a single tensor
    torch_conv.to_tensor(torch_2_tensor)

    with pytest.raises(IndexError):
        # missing parameters in the tensor
        torch_conv.to_tensor([torch_1_tensor])

    # should not raise an error - 2 parameters expected
    torch_conv.to_tensor([torch_2_tensor])

    # should not raise an error
    CircuitConverter(c_ps, ["x", "y"])

    # missing "x"
    with pytest.raises(ValueError):
        CircuitConverter(c_ps, ["y"])

    # extraneous parameter "z"
    with pytest.raises(ValueError):
        CircuitConverter(c_ps, ["x", "y", "z"])


@pytest.mark.parametrize("bs_type", [BS.Rx(Parameter("theta")), BS.Ry(Parameter("theta")), BS.H(Parameter("theta"))])
def test_bs_to_torch(bs_type):
    c_bs = Circuit(2) // bs_type
    param_theta = torch.tensor([np.pi/2], requires_grad=True)

    torch_conv = CircuitConverter(c_bs, input_specs=["theta"])
    torch_tensor = torch_conv.to_tensor([param_theta])

    assert torch_tensor.shape == torch.Size((2*[c_bs.m]))

    # compute unitary at the initial parameter value from perceval
    comp_param = c_bs.get_parameters()[0]
    comp_param.set_value(param_theta)
    exptd_u = torch.tensor(c_bs.compute_unitary(), dtype=torch.complex64)

    torch.allclose(torch_tensor, exptd_u)

@pytest.mark.parametrize("component", [PERM([1, 3, 4, 0, 2]), Unitary(Matrix.random_unitary(5))])
def test_non_param_comp_to_torch(component):
    circ = Circuit(5) // (0, PS(Parameter("x"))) // (0, component)
    params = {"x": torch.tensor([0.1], requires_grad=True)}

    torch_conv = CircuitConverter(circ, ["x"])
    torch_tensor = torch_conv.to_tensor(list(params.values()))

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 2
    assert torch_tensor.shape == torch.Size(2*[circ.m])

    comp_param = circ.get_parameters()[0]
    comp_param.set_value(params[comp_param.name])
    exptd_u = torch.tensor(circ.compute_unitary(), dtype=torch.complex64)
    torch.allclose(torch_tensor, exptd_u)

def test_mixed_comps():
    circ = Circuit(3) // (0, PERM([1, 2, 0])) // (0, PS(Parameter("x"))) // (1, BS(Parameter("theta")))
    params = {"x": torch.tensor([0.1], requires_grad=True), "theta": torch.tensor([np.pi/2], requires_grad=True)}

    torch_conv = CircuitConverter(circ, ["x", "theta"])
    torch_tensor = torch_conv.to_tensor(list(params.values()))

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() ==  2 # single element tensor

    for each_param in circ.get_parameters():
        each_param.set_value(params[each_param.name])
    exptd_u = torch.tensor(circ.compute_unitary(), dtype=torch.complex64)

    torch.allclose(torch_tensor, exptd_u)

def test_complex_circuit():
    c1 = Circuit(1) // PS(Parameter("x"))
    circ = Circuit(2) // (0, BS(Parameter("theta")))
    circ.add(0, c1)

    params = {"x": torch.tensor([0.1], requires_grad=True), "theta": torch.tensor([np.pi/2], requires_grad=True)}

    torch_conv = CircuitConverter(circ, ["x", "theta"])
    torch_tensor = torch_conv.to_tensor(list(params.values()))

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 2

    for each_param in circ.get_parameters():
        each_param.set_value(params[each_param.name])
    exptd_u = torch.tensor(circ.compute_unitary(), dtype=torch.complex64)

    torch.allclose(torch_tensor, exptd_u)

def test_torch_input():
    c_ps = Circuit(1) // PS(Parameter("x"))
    params = torch.tensor([0.1], requires_grad=True)

    torch_conv = CircuitConverter(c_ps, [""])
    torch_tensor = torch_conv.to_tensor([params])

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 2

    param_name = c_ps.get_parameters()[0]
    param_name.set_value(params[0])
    exptd_u = torch.tensor(c_ps.compute_unitary(), dtype=torch.complex64)

    torch.allclose(torch_tensor, exptd_u)

def test_torch_input_batch_ps():
    c_ps = Circuit(1) // PS(Parameter("x"))
    params = torch.tensor([[0.1], [0.2]], requires_grad=True)

    torch_conv = CircuitConverter(c_ps, [""])
    torch_tensor = torch_conv.to_tensor([params])

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 3
    assert torch_tensor.shape == torch.Size([2, 1, 1])

    param_name = c_ps.get_parameters()[0]
    for i in range(len(params)):
        param_name.set_value(params[0])
        exptd_u = torch.tensor(c_ps.compute_unitary(), dtype=torch.complex64)
        torch.allclose(torch_tensor[i], exptd_u)

@pytest.mark.parametrize("circuit_size", [2, 4])
def test_torch_input_batch_bs(circuit_size):
    c_ps = Circuit(circuit_size) // BS.H(Parameter("theta"))
    params = torch.tensor([[np.pi], [np.pi/2], [np.pi/4]], requires_grad=True)
    torch_conv = CircuitConverter(c_ps, ["theta"])
    torch_tensor = torch_conv.to_tensor([params])

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 3
    assert torch_tensor.shape == torch.Size([3, c_ps.m, c_ps.m])

    param_name = c_ps.get_parameters()[0]
    for i in range(len(params)):
        param_name.set_value(params[0])
        exptd_u = torch.tensor(c_ps.compute_unitary(), dtype=torch.complex64)
        torch.allclose(torch_tensor[i], exptd_u)

def test_psfixed_to_torch():
    c_ps = Circuit(1) // PS(Parameter("x")) // PS(np.pi)
    params = {"x": torch.tensor([0.1], requires_grad=True)}

    torch_conv = CircuitConverter(c_ps, ["x"])
    torch_tensor = torch_conv.to_tensor([params["x"]])

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 2

    comp_param = c_ps.get_parameters()[0]
    comp_param.set_value(params[comp_param.name])
    exptd_u = torch.tensor(c_ps.compute_unitary(), dtype=torch.complex64)

    torch.allclose(torch_tensor, exptd_u)

def test_circuit_with_subcircuit():
    c1 = Circuit(2) // (0, BS()) // (1, PS(Parameter("x"))) // (0, BS())
    circ = Circuit(2) // (0, PS(Parameter("theta")))
    circ.add(0, c1, merge=False)

    params = {"x": torch.tensor([0.1], requires_grad=True), "theta": torch.tensor([np.pi/2], requires_grad=True)}

    torch_conv = CircuitConverter(circ, ["x", "theta"])
    torch_tensor = torch_conv.to_tensor(*list(params.values()))

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 2

    for each_param in circ.get_parameters():
        each_param.set_value(params[each_param.name])
    exptd_u = torch.tensor(circ.compute_unitary(), dtype=torch.complex64)

    torch.allclose(torch_tensor, exptd_u)

def test_circuit_with_PERM_and_batch():
    c1 = Circuit(2) // (0, BS()) // (1, PS(Parameter("x"))) // (0, BS())
    circ = Circuit(2) // (0, PERM([1, 0]))
    circ.add(0, c1, merge=False)

    params = {"x": torch.tensor([[0.1], [0.2]], requires_grad=True)}

    torch_conv = CircuitConverter(circ, ["x"])
    torch_tensor = torch_conv.to_tensor(params["x"])

    assert torch_tensor.requires_grad
    assert torch_tensor.dim() == 3
    assert torch_tensor.shape == torch.Size([2, 2, 2])

    for batch_idx in range(params["x"].shape[0]):
        for each_param in circ.get_parameters():
            each_param.set_value(params[each_param.name][batch_idx])
        exptd_u = torch.tensor(circ.compute_unitary(), dtype=torch.complex64)

        torch.allclose(torch_tensor[batch_idx], exptd_u)
