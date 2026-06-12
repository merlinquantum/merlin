import pytest
import perceval as pcvl
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor
from merlin.measurement.strategies import MeasurementStrategy


def _build_merlin_processor() -> MerlinProcessor:
    rp = pcvl.Processor("SLOS")
    return MerlinProcessor(
        processor=rp,
        microbatch_size=32,
        timeout=3600.0,
        max_shots_per_call=None,
        chunk_concurrency=1,
    )


def _build_quantum_layer() -> QuantumLayer:
    builder = CircuitBuilder(n_modes=6)
    builder.add_rotations(trainable=True, name="theta")
    builder.add_angle_encoding(modes=[0, 1], name="px")
    builder.add_entangling_layer()

    return QuantumLayer(
        input_size=2,
        builder=builder,
        n_photons=2,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED,
        ),
        dtype=torch.float32,
    )


def _random_one_hot_targets_like(output: torch.Tensor) -> torch.Tensor:
    labels = torch.randint(low=0, high=output.shape[1], size=(output.shape[0],))
    targets = torch.zeros_like(output)
    targets[torch.arange(output.shape[0]), labels] = 1.0
    return targets


def test_processor_of_hybrid_model_gradient_fail_without_eval():
    proc = _build_merlin_processor()
    quantum_layer = _build_quantum_layer()

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False, dtype=torch.float32),
        quantum_layer,
        nn.Linear(15, 4, bias=False, dtype=torch.float32),
    )

    X = torch.rand(8, 3)
    model.eval()
    y = proc.forward(model, X, nsample=5000)

    labels_to_check = _random_one_hot_targets_like(y)
    loss = torch.nn.MSELoss()(labels_to_check, y)

    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


def test_hybrid_model_of_processor_gradient_passes():
    proc = _build_merlin_processor()
    quantum_layer = _build_quantum_layer()

    class MyModel(nn.Module):
        def __init__(self, qlayer: QuantumLayer, processor: MerlinProcessor) -> None:
            super().__init__()
            self.layer_1 = nn.Linear(3, 2, bias=False, dtype=torch.float32)
            self.layer_2 = nn.Linear(15, 4, bias=False, dtype=torch.float32)
            self.qlayer = qlayer.eval()
            self.processor = processor

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out_1 = self.layer_1(x)
            out_2 = self.processor.forward(self.qlayer, out_1, nsample=5000)
            return self.layer_2(out_2)

    model = MyModel(quantum_layer, proc)

    X = torch.rand(8, 3)
    y = model(X)

    labels_to_check = _random_one_hot_targets_like(y)
    loss = torch.nn.MSELoss()(labels_to_check, y)

    loss.backward()

    assert [p.grad for p in model.layer_1.parameters()] == [None]
    assert [p.grad for p in model.qlayer.parameters()] == [None, None]
    assert not [p.grad for p in model.layer_2.parameters()] == [None]
