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

"""Tests for the QCNNClassifier model class"""

import json

import pytest
import torch

from merlin.algorithms import QuantumLayer
from merlin.core import ComputationSpace, StateVector
from merlin.measurement import MeasurementStrategy
from merlin.models import QCNNClassifier


def test_qcnn_basic_api():
    accepted_input_shape = (4, 4)
    non_square_input_shape = (4, 8)
    three_tuple_input_shape = (4, 4, 4)
    list_input_shape = [4, 4]
    input_shape_above_former_limit = (29, 29)
    invalid_input_shape = "4,4"
    float_input_shape_v1 = (4.0, 4.0)
    float_input_shape_v2 = (float(4), float(4))
    negative_input_shape = (-4, -4)

    accepted_num_classes = 10
    zero_num_classes = 0
    negative_num_classes = -1
    float_num_classes_v1 = 4.0
    float_num_classes_v2 = float(4)

    # API tests for input_shape and num_classes

    qcnn_classifier = QCNNClassifier(accepted_input_shape, accepted_num_classes)

    with pytest.raises(ValueError, match="input_shape must represent a square"):
        QCNNClassifier(non_square_input_shape, accepted_num_classes)

    with pytest.raises(
        ValueError, match="input_shape must be a tuple or list of size 2"
    ):
        QCNNClassifier(three_tuple_input_shape, accepted_num_classes)

    with pytest.raises(TypeError, match="input_shape must have tuple or list type"):
        QCNNClassifier(invalid_input_shape, accepted_num_classes)

    with pytest.raises(TypeError, match="input_shape elements must have int type"):
        QCNNClassifier(float_input_shape_v1, accepted_num_classes)

    with pytest.raises(TypeError, match="input_shape elements must have int type"):
        QCNNClassifier(float_input_shape_v2, accepted_num_classes)

    with pytest.raises(
        ValueError, match="input_shape must contain values superior to 0"
    ):
        QCNNClassifier(negative_input_shape, accepted_num_classes)

    with pytest.raises(ValueError, match="num_classes must be superior to 0"):
        QCNNClassifier(accepted_input_shape, zero_num_classes)

    with pytest.raises(ValueError, match="num_classes must be superior to 0"):
        QCNNClassifier(accepted_input_shape, negative_num_classes)

    with pytest.raises(TypeError, match="num_classes must have int type"):
        QCNNClassifier(accepted_input_shape, float_num_classes_v1)

    with pytest.raises(TypeError, match="num_classes must have int type"):
        QCNNClassifier(accepted_input_shape, float_num_classes_v2)

    assert qcnn_classifier.input_shape == accepted_input_shape
    assert qcnn_classifier.num_classes == accepted_num_classes

    qcnn_classifier_from_list = QCNNClassifier(list_input_shape, accepted_num_classes)
    assert qcnn_classifier_from_list.input_shape == accepted_input_shape

    amplitude_basis_indices = qcnn_classifier_from_list._amplitude_basis_indices.clone()
    list_input_shape[0] = 2
    assert qcnn_classifier_from_list.input_shape == accepted_input_shape
    assert torch.equal(
        qcnn_classifier_from_list._amplitude_basis_indices, amplitude_basis_indices
    )
    with pytest.raises(AttributeError):
        qcnn_classifier_from_list.input_shape = (2, 2)
    
    qcnn_classifier_above_former_limit = QCNNClassifier(
        input_shape_above_former_limit,
        accepted_num_classes,
        [QCNNClassifier.QDense()],
    )
    assert (
        qcnn_classifier_above_former_limit.input_shape == input_shape_above_former_limit
    )

    # Modifying the input shape list after model initialization should not change the model's input shape
    input_shape = [2, 2]
    num_classes = 2
    qcnn = QCNNClassifier(input_shape, num_classes)
    original_input_shape = qcnn.input_shape

    input_shape[0] += 2
    input_shape[1] += 2

    assert qcnn.input_shape == original_input_shape


def test_qcnn_stage_api():
    accepted_kernel_size = 2
    zero_kernel_size = 0
    one_kernel_size = 1
    negative_kernel_size = -2
    float_kernel_size = float(2)

    accepted_stride = 2
    zero_stride = 0
    negative_Stride = -2
    float_stride = float(2)

    # QConv and QPool initializations
    qconv = QCNNClassifier.QConv(accepted_kernel_size, accepted_stride)
    assert str(qconv) == "QConv(kernel_size=2, stride=2)"

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QConv(zero_kernel_size, accepted_stride)

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QConv(negative_kernel_size, accepted_stride)

    with pytest.raises(ValueError, match="stride must be superior to 0"):
        QCNNClassifier.QConv(accepted_kernel_size, zero_stride)

    with pytest.raises(ValueError, match="stride must be superior to 0"):
        QCNNClassifier.QConv(accepted_kernel_size, negative_Stride)

    with pytest.raises(TypeError, match="kernel_size must have int type"):
        QCNNClassifier.QConv(float_kernel_size, accepted_stride)

    with pytest.raises(TypeError, match="stride must have int type"):
        QCNNClassifier.QConv(accepted_kernel_size, float_stride)

    assert qconv.type.value == "QConv"

    qpool = QCNNClassifier.QPool(accepted_kernel_size)
    assert qpool.type.value == "QPool"
    assert qpool.kernel_size == accepted_kernel_size
    assert str(qpool) == "QPool(kernel_size=2)"

    qdense = QCNNClassifier.QDense()
    assert str(qdense) == "QDense()"

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QPool(zero_kernel_size)

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QPool(one_kernel_size)

    with pytest.raises(ValueError, match="kernel_size must be superior to 1"):
        QCNNClassifier.QPool(negative_kernel_size)

    with pytest.raises(TypeError, match="kernel_size must have int type"):
        QCNNClassifier.QPool(float_kernel_size)

    accepted_input_shape = (4, 4)
    accepted_num_classes = 10
    qcnn_classifier = QCNNClassifier(accepted_input_shape, accepted_num_classes)

    # Default stages
    stages = []
    stages.append(QCNNClassifier.QConv(2, 2))
    stages.append(QCNNClassifier.QPool(2))
    stages.append(QCNNClassifier.QDense())

    assert qcnn_classifier.resolved_stages == stages
    assert qcnn_classifier.stages is None

    # QCNNClassifier.resolve_stages() cases
    valid_custom_stages = [
        QCNNClassifier.QConv(2, 1),
        QCNNClassifier.QPool(2),
        QCNNClassifier.QDense(),
    ]
    custom_qcnn = QCNNClassifier(
        accepted_input_shape, accepted_num_classes, valid_custom_stages
    )
    assert custom_qcnn.resolved_stages == valid_custom_stages

    with pytest.raises(ValueError, match="stages cannot be an empty list"):
        QCNNClassifier(accepted_input_shape, accepted_num_classes, [])

    with pytest.raises(ValueError, match="stage 0 has invalid type"):
        QCNNClassifier(accepted_input_shape, accepted_num_classes, [object()])

    with pytest.raises(ValueError, match="stage 2 has invalid type"):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [
                QCNNClassifier.QConv(2, 1),
                QCNNClassifier.QPool(2),
                object(),
                QCNNClassifier.QDense(),
            ],
        )

    with pytest.raises(ValueError, match="only last stage can be QDense"):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QDense(), QCNNClassifier.QDense()],
        )

    with pytest.raises(ValueError, match="last stage has to be QDense"):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(2, 1), QCNNClassifier.QPool(2)],
        )

    # Accepted
    QCNNClassifier(
        accepted_input_shape,
        accepted_num_classes,
        [QCNNClassifier.QConv(3, 1), QCNNClassifier.QDense()],
    )

    with pytest.raises(
        ValueError, match="must be divisible by the pooling kernel size"
    ):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QPool(3), QCNNClassifier.QDense()],
        )

    with pytest.raises(TypeError, match="stages must be None or have the list type"):
        QCNNClassifier(accepted_input_shape, accepted_num_classes, ())

    # kernel_size > input_shape[0]
    with pytest.raises(
        ValueError, match="must be superior or equal to the convolution kernel size"
    ):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(5, 1), QCNNClassifier.QDense()],
        )

    # Accepted
    QCNNClassifier(
        accepted_input_shape,
        accepted_num_classes,
        [QCNNClassifier.QConv(4, 1), QCNNClassifier.QDense()],
    )

    # stride > kernel_size
    with pytest.raises(
        ValueError, match="must be superior or equal to convolution stride"
    ):
        QCNNClassifier(
            accepted_input_shape,
            accepted_num_classes,
            [QCNNClassifier.QConv(2, 3), QCNNClassifier.QDense()],
        )

    # QCNNClassifier._resolved_stages is read only
    with pytest.raises(AttributeError):
        qcnn_classifier.resolved_stages = []

    resolved_stages = qcnn_classifier.resolved_stages
    resolved_stages.append(QCNNClassifier.QDense())
    assert qcnn_classifier.resolved_stages == stages

    resolved_stages[0].kernel_size = 4
    assert qcnn_classifier.resolved_stages == stages

    valid_custom_stages.append(QCNNClassifier.QDense())
    assert custom_qcnn.resolved_stages == [
        QCNNClassifier.QConv(2, 1),
        QCNNClassifier.QPool(2),
        QCNNClassifier.QDense(),
    ]

    # Try to access stages class through import without QCNNClassifier
    # Have to ignore ruff here or else it replaces unused imports with `pass` statements
    with pytest.raises(ImportError):
        from merlin import QConv  # noqa: F401

    with pytest.raises(ImportError):
        from merlin.models import QPool  # noqa: F401

    with pytest.raises(ImportError):
        from merlin.models.qcnn import QDense  # noqa: F401

    # Try to access private classes through import without QCNNClassifier
    with pytest.raises(ImportError):
        from merlin.models.qcnn import _QCNNStageTypes  # noqa: F401

    with pytest.raises(ImportError):
        from merlin.models.qcnn import _Stage  # noqa: F401

    # Cannot instanciate QCNNClassifier with _Stage objects directly
    qconv_type = QCNNClassifier._QCNNStageTypes.QConv
    qconv_stage = QCNNClassifier._Stage(qconv_type)

    qdense_type = QCNNClassifier._QCNNStageTypes.QDense
    qdense_stage = QCNNClassifier._Stage(qdense_type)

    stages = [qconv_stage, qdense_stage]

    with pytest.raises(ValueError, match="Invalid stage type"):
        QCNNClassifier((4, 4), 2, stages)


def test_qcnn_resolved_stages_isolated_from_input_stages_mutation():
    stages = [QCNNClassifier.QConv(2, 2), QCNNClassifier.QDense()]
    model = QCNNClassifier((4, 4), 2, stages=stages)
    expected_summary = model.summary()
    expected_config = model.export_config()
    expected_layers = list(model.layers._modules)
    expected_stages = [QCNNClassifier.QConv(2, 2), QCNNClassifier.QDense()]

    stages.append(QCNNClassifier.QPool(2))
    stages[0].kernel_size = 3
    stages[0].stride = 1

    assert model.resolved_stages == expected_stages
    assert model.summary() == expected_summary
    assert model.export_config() == expected_config
    assert list(model.layers._modules) == expected_layers


def test_qcnn_resolved_stages_snapshot_mutation_does_not_change_model_metadata():
    model = QCNNClassifier((4, 4), 2)
    expected_stages = [
        QCNNClassifier.QConv(2, 2),
        QCNNClassifier.QPool(2),
        QCNNClassifier.QDense(),
    ]
    expected_summary = model.summary()
    expected_config = model.export_config()
    expected_layers = list(model.layers._modules)

    resolved_stages = model.resolved_stages
    resolved_stages.append(QCNNClassifier.QDense())
    resolved_stages[0].kernel_size = 4
    resolved_stages[0].stride = 1
    resolved_stages[1] = QCNNClassifier.QConv(2, 1)

    assert model.resolved_stages == expected_stages
    assert model.summary() == expected_summary
    assert model.export_config() == expected_config
    assert list(model.layers._modules) == expected_layers


def test_qcnn_invalid_stage_object_raises_clear_value_error():
    with pytest.raises(ValueError) as exc_info:
        QCNNClassifier((4, 4), 2, stages=[object()])

    message = str(exc_info.value)
    assert "stage 0 has invalid type" in message
    assert "stages must be of type QConv, QPool or QDense" in message


def test_qcnn_summary():
    qcnn_classifier = QCNNClassifier((4, 4), 10)

    expected_summary = (
        "QCNNClassifier("
        "input_shape=(4, 4), "
        "num_classes=10, "
        "stages=[QConv(kernel_size=2, stride=2) -> QPool(kernel_size=2) -> QDense()]"
        ")"
    )
    assert qcnn_classifier.summary() == expected_summary


def test_qcnn_export_config():
    stages = [
        QCNNClassifier.QConv(4, 1),
        QCNNClassifier.QDense(),
    ]
    qcnn_classifier = QCNNClassifier((4, 4), 3, stages=stages)

    expected_config = {
        "input_shape": (4, 4),
        "num_classes": 3,
        "stages": [
            {"type": "QConv", "kernel_size": 4, "stride": 1},
            {"type": "QDense"},
        ],
    }
    assert qcnn_classifier.export_config() == expected_config

    # Round trip test
    config = qcnn_classifier.export_config()

    new_qcnn_classifier = QCNNClassifier.from_config(config)

    assert new_qcnn_classifier.input_shape == qcnn_classifier.input_shape
    assert new_qcnn_classifier.num_classes == qcnn_classifier.num_classes
    assert new_qcnn_classifier.resolved_stages == qcnn_classifier.resolved_stages

    json_config = json.loads(json.dumps(config))
    restored_qcnn_classifier = QCNNClassifier.from_config(json_config)

    assert isinstance(json_config["input_shape"], list)
    assert restored_qcnn_classifier.input_shape == qcnn_classifier.input_shape
    assert restored_qcnn_classifier.num_classes == qcnn_classifier.num_classes
    assert restored_qcnn_classifier.resolved_stages == qcnn_classifier.resolved_stages


def test_qcnn_amplitude_encoding():
    input_shape = (4, 4)
    num_classes = 2
    qcnn = QCNNClassifier(input_shape, num_classes)

    # Build input to amplitude encode
    x_tensor_0 = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]).unsqueeze(0)
    x_tensor_1 = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]).unsqueeze(0)
    x_tensor_2 = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]).unsqueeze(0)
    x_tensor_3 = torch.rand((4, 4)).unsqueeze(0)

    x_tensor = torch.cat([x_tensor_0, x_tensor_1, x_tensor_2, x_tensor_3], dim=0)
    x_tensor = x_tensor.unsqueeze(1)

    assert x_tensor.shape == (4, 1, 4, 4)
    assert qcnn._amplitude_basis_indices.shape == (input_shape[0] * input_shape[1],)
    assert qcnn._amplitude_basis_indices.device.type == "cpu"

    amplitude_encoded_state_vector = qcnn.amplitude_encode(x_tensor)

    assert isinstance(amplitude_encoded_state_vector, StateVector)
    basis_size = StateVector(
        torch.tensor([]), n_modes=sum(input_shape), n_photons=2
    ).basis_size
    assert amplitude_encoded_state_vector.shape == (4, basis_size)
    assert amplitude_encoded_state_vector.is_normalized

    # Verify that the obtained state vector is the one expected
    first_state_tensor = amplitude_encoded_state_vector.tensor[0]
    first_expected_state = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    ]  # By definition of the amplitude encoding used (2 registers)
    first_expected_tensor = StateVector.from_basic_state(first_expected_state).tensor
    assert torch.allclose(
        first_state_tensor.to_dense(),
        first_expected_tensor.to_dense(),
        atol=1e-6,
        rtol=1e-6,
    )

    # Verify that the obtained state vector is the one expected
    second_state_tensor = amplitude_encoded_state_vector.tensor[1]
    second_expected_state = [
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
    ]  # By definition of the amplitude encoding used (2 registers)
    second_expected_tensor = StateVector.from_basic_state(second_expected_state).tensor
    assert torch.allclose(
        second_state_tensor.to_dense(),
        second_expected_tensor.to_dense(),
        atol=1e-6,
        rtol=1e-6,
    )

    # Verify that the obtained state vector is a uniform combination of all allowed basic states
    third_state_tensor = amplitude_encoded_state_vector.tensor[2]
    all_basic_states_indices = []
    for i in range(4):
        for j in range(4):
            state = [0] * 8
            state[i] = 1
            state[4 + j] = 1
            basic_state_index = amplitude_encoded_state_vector.index(state)
            all_basic_states_indices.append(basic_state_index)

    allowed_amplitudes = []
    forbidden_amplitudes = []
    for index, amplitude in enumerate(third_state_tensor):
        if index in all_basic_states_indices:
            allowed_amplitudes.append(amplitude)
        else:
            forbidden_amplitudes.append(amplitude)

    allowed = torch.tensor(allowed_amplitudes).to_dense()
    forbidden = torch.tensor(forbidden_amplitudes).to_dense()

    assert torch.allclose(
        allowed[0].unsqueeze(0).repeat(len(allowed)), allowed, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(torch.zeros_like(forbidden), forbidden, atol=1e-6, rtol=1e-6)

    # Verify that forbidden basic states are not returned
    fourth_state_tensor = amplitude_encoded_state_vector.tensor[3]
    for index, amplitude in enumerate(fourth_state_tensor):
        # Use the same basic state indices
        if index in all_basic_states_indices:
            allowed_amplitudes.append(amplitude)
        else:
            forbidden_amplitudes.append(amplitude)

    allowed = torch.tensor(allowed_amplitudes).to_dense()
    forbidden = torch.tensor(forbidden_amplitudes).to_dense()

    assert torch.allclose(torch.zeros_like(forbidden), forbidden, atol=1e-6, rtol=1e-6)


def test_full_default_qcnn():
    # Set torch random seed for reproducibility
    torch.manual_seed(42)

    input_shape = (4, 4)
    num_classes = 2
    qcnn = QCNNClassifier(input_shape, num_classes)

    initial_param_values = {}
    for name, param in qcnn.named_parameters():
        before = param.detach().clone()
        initial_param_values[name] = before

    # Generate data and labels
    x_tensor_0 = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]).unsqueeze(0)
    x_tensor_1 = torch.tensor([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]).unsqueeze(0)
    x_tensor_2 = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]).unsqueeze(0)
    x_tensor_3 = torch.rand((4, 4)).unsqueeze(0)

    x_tensor = torch.cat([x_tensor_0, x_tensor_1, x_tensor_2, x_tensor_3], dim=0)
    x_tensor = x_tensor.unsqueeze(1)

    y_0 = torch.tensor([1, 0], dtype=torch.float).unsqueeze(0)
    y_1 = torch.tensor([1, 0], dtype=torch.float).unsqueeze(0)
    y_2 = torch.tensor([0, 1], dtype=torch.float).unsqueeze(0)
    y_3 = torch.tensor([0, 1], dtype=torch.float).unsqueeze(0)

    y_tensor = torch.cat([y_0, y_1, y_2, y_3], dim=0)

    assert x_tensor.shape == (4, 1, 4, 4)
    assert y_tensor.shape == (4, 2)

    optimizer = torch.optim.Adam(qcnn.parameters(), lr=1e-1)

    qcnn.train()
    optimizer.zero_grad()
    logits = qcnn(x_tensor)

    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(logits, y_tensor)
    loss.backward()

    # Check that gradients exist and are defined.
    any_grad_non_zero = False
    for name, param in qcnn.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} gradient has NaN/Inf"
        any_grad_non_zero = any_grad_non_zero or (
            param.grad.detach().norm().cpu() > 1e-8
        )

    assert any_grad_non_zero, "no gradients were non-zero"

    optimizer.step()

    # Check that at least one parameter value changed after optimizer step.
    any_parameter_updated = False
    for name, param in qcnn.named_parameters():
        before = initial_param_values[name]
        after = param.detach().clone()
        any_parameter_updated = any_parameter_updated or not torch.allclose(
            before, after, atol=1e-4, rtol=1e-4
        )

    assert any_parameter_updated, "no parameter changed after optimizer step"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_qcnn_forward_keeps_cuda_device():
    device = torch.device("cuda")
    qcnn = QCNNClassifier((4, 4), 2).to(device)
    x = torch.rand((2, 1, 4, 4), device=device)

    assert x.is_cuda
    assert qcnn._amplitude_basis_indices.is_cuda
    for parameter in qcnn.parameters():
        assert parameter.is_cuda
    for layer in qcnn.layers:
        if isinstance(layer, QuantumLayer):
            assert torch.device(layer.device).type == "cuda"

    logits = qcnn(x)

    assert logits.is_cuda
    assert logits.device.type == device.type
    assert logits.shape == (2, 2)

    loss = logits.sum()
    loss.backward()
    for parameter in qcnn.parameters():
        assert parameter.grad is not None
        assert parameter.grad.is_cuda


def test_qcnn_layers_public_sequential_api():
    qcnn = QCNNClassifier((4, 4), 3)

    assert isinstance(qcnn.layers, torch.nn.Sequential)
    named_layers = dict(qcnn.layers.named_children())
    assert list(named_layers) == ["QConv_1", "QPool_1", "QDense", "Readout"]

    qconv = qcnn.layers[0]
    qpool = qcnn.layers[1]
    qdense = qcnn.layers[2]
    readout = qcnn.layers[3]

    assert named_layers["QConv_1"] is qconv
    assert named_layers["QPool_1"] is qpool
    assert named_layers["QDense"] is qdense
    assert named_layers["Readout"] is readout

    assert isinstance(qconv, QuantumLayer)
    assert isinstance(qpool, QuantumLayer)
    assert isinstance(qdense, QuantumLayer)
    assert isinstance(readout, torch.nn.Linear)

    assert qconv.circuit.m == 8
    assert {param.name for param in qconv.circuit.get_parameters()} == {
        "px_first_register",
        "px_second_register",
    }
    assert qpool.circuit.m == 8
    assert qdense.circuit.m == 4

    assert isinstance(qconv.output_size, int)
    assert isinstance(qpool.output_size, int)
    assert isinstance(qdense.output_size, int)
    assert qconv.output_size > 0
    assert qpool.output_size > 0
    assert qdense.output_size > 0

    assert isinstance(qconv.measurement_strategy, MeasurementStrategy)
    assert qconv.measurement_strategy.type.value == "AMPLITUDES"
    assert qconv.measurement_strategy.computation_space == ComputationSpace.FOCK
    assert isinstance(qpool.measurement_strategy, MeasurementStrategy)
    assert qpool.measurement_strategy.type.value == "PARTIAL"
    assert qpool.measurement_strategy.measured_modes == (0, 2, 4, 6)
    assert isinstance(qdense.measurement_strategy, MeasurementStrategy)
    assert qdense.measurement_strategy.type.value == "PROBABILITIES"

    assert readout.in_features == qdense.output_size
    assert readout.out_features == qcnn.num_classes


def test_qcnn_layers_custom_stage_names_and_dense_access():
    stages = [
        QCNNClassifier.QConv(2, 1),
        QCNNClassifier.QConv(2, 2),
        QCNNClassifier.QDense(),
    ]
    qcnn = QCNNClassifier((4, 4), 2, stages=stages)

    named_layers = dict(qcnn.layers.named_children())
    assert list(named_layers) == ["QConv_1", "QConv_2", "QDense", "Readout"]
    assert qcnn.layers[0] is named_layers["QConv_1"]
    assert qcnn.layers[1] is named_layers["QConv_2"]

    first_conv = qcnn.layers[0]
    second_conv = qcnn.layers[1]
    dense = qcnn.layers[2]

    assert isinstance(first_conv, QuantumLayer)
    assert isinstance(second_conv, QuantumLayer)
    assert isinstance(dense, QuantumLayer)

    assert isinstance(first_conv.measurement_strategy, MeasurementStrategy)
    assert first_conv.measurement_strategy.type.value == "AMPLITUDES"
    assert isinstance(second_conv.measurement_strategy, MeasurementStrategy)
    assert second_conv.measurement_strategy.type.value == "AMPLITUDES"
    assert isinstance(dense.measurement_strategy, MeasurementStrategy)
    assert dense.measurement_strategy.type.value == "PROBABILITIES"

    assert first_conv.circuit.m == 8
    assert second_conv.circuit.m == 8
    assert dense.circuit.m == 8
    assert qcnn.layers[3].in_features == dense.output_size


def test_qcnn_state_dict_round_trip_with_export_config():
    torch.manual_seed(1234)
    stages = [
        QCNNClassifier.QConv(2, 1),
        QCNNClassifier.QPool(2),
        QCNNClassifier.QDense(),
    ]
    qcnn = QCNNClassifier((4, 4), 3, stages=stages)

    with torch.no_grad():
        for index, parameter in enumerate(qcnn.parameters()):
            parameter.fill_(0.1 * (index + 1))

    config = qcnn.export_config()
    state_dict = {
        key: value.detach().clone() for key, value in qcnn.state_dict().items()
    }

    restored_qcnn = QCNNClassifier.from_config(config)
    load_result = restored_qcnn.load_state_dict(state_dict)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert restored_qcnn.export_config() == config

    assert qcnn.state_dict().keys() == restored_qcnn.state_dict().keys()
    for name, expected_tensor in state_dict.items():
        restored_tensor = restored_qcnn.state_dict()[name]
        assert torch.allclose(restored_tensor, expected_tensor)

    for (name, expected), (restored_name, actual) in zip(
        qcnn.named_parameters(), restored_qcnn.named_parameters(), strict=True
    ):
        assert restored_name == name
        assert torch.allclose(actual, expected)

    x = torch.rand((2, 1, 4, 4))
    qcnn.eval()
    restored_qcnn.eval()
    with torch.no_grad():
        expected_logits = qcnn(x)
        restored_logits = restored_qcnn(x)

    assert torch.allclose(restored_logits, expected_logits, atol=1e-6, rtol=1e-6)

