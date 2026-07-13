import numpy as np
import perceval as pcvl
import pytest
import torch

from merlin import Combinadics, ComputationSpace
from merlin.algorithms.layer_utils import NoiseGroups, classify_noise
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter
from merlin.pcvl_pytorch.noisy_slos import (
    NoisyG2SLOSComputeGraph,
    NoisySLOSComputeGraph,
    _InputStateNoisySLOSComputeGraph,
)
from merlin.pcvl_pytorch.slos_torchscript import SLOSComputeGraph


@pytest.fixture
def entangling_circuit():
    # Create a 5-mode photonic processor
    circuit = pcvl.Circuit(5, name="5-Mode Fully Entangled Circuit")

    # Layer 1: Initial beam splitter network
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add((1, 2), pcvl.BS.H())
    circuit.add((2, 3), pcvl.BS.H())
    circuit.add((3, 4), pcvl.BS.H())

    # Phase shifts to generate nontrivial interference
    circuit.add(0, pcvl.PS(phi=0.3))
    circuit.add(1, pcvl.PS(phi=1.1))
    circuit.add(2, pcvl.PS(phi=2.0))
    circuit.add(3, pcvl.PS(phi=0.7))
    circuit.add(4, pcvl.PS(phi=1.7))

    # Layer 2: Cross-coupling for stronger entanglement
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add((1, 2), pcvl.BS.H())
    circuit.add((3, 4), pcvl.BS.H())

    # Additional phase tuning
    circuit.add(0, pcvl.PS(phi=2.4))
    circuit.add(2, pcvl.PS(phi=1.3))
    circuit.add(4, pcvl.PS(phi=0.9))

    # Final mixing layer
    circuit.add((0, 1), pcvl.BS.H())
    circuit.add((1, 2), pcvl.BS.H())
    circuit.add((2, 3), pcvl.BS.H())
    circuit.add((3, 4), pcvl.BS.H())

    return circuit


def test_identify_hom_one(entangling_circuit):

    circuit_to_analyze = entangling_circuit
    unitary = CircuitConverter(circuit_to_analyze).to_tensor([])

    # noisy computation
    groups = NoiseGroups(
        source={"indistinguishability": 1.0}, circuit=None, post_measurement=None
    )
    noisy_slos = NoisySLOSComputeGraph(
        groups, m=5, n_photons=2, computation_space=ComputationSpace.FOCK
    )

    # Noiseless computation
    slos = SLOSComputeGraph(m=5, n_photons=2, computation_space=ComputationSpace.FOCK)

    for state in Combinadics(scheme="fock", n=2, m=5).enumerate_states():
        noisy_output = noisy_slos.compute_probs(unitary, state)
        clean_output = slos.compute_probs(unitary, state)
        # Check that the output order follows combinatics
        assert noisy_output[0] == clean_output[0]
        assert (
            noisy_output[0] == Combinadics(scheme="fock", n=2, m=5).enumerate_states()
        )
        assert noisy_output[0] == noisy_slos.mapped_keys

        assert torch.allclose(noisy_output[1], clean_output[1])


def test_fully_distinguishable_hom_zero():
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    unitary = CircuitConverter(circuit).to_tensor([])

    noise_groups = classify_noise(pcvl.NoiseModel(indistinguishability=0.0))
    noisy_slos = NoisySLOSComputeGraph(
        noise_groups,
        m=2,
        n_photons=2,
        computation_space=ComputationSpace.FOCK,
        keep_keys=True,
        device=None,
        dtype=torch.float32,
    )

    keys, probs = noisy_slos.compute_probs(unitary, [1, 1])
    p = {k: probs[0, i].item() for i, k in enumerate(keys)}

    # Fully distinguishable photons -> classical split:
    # P(2,0)=0.25, P(1,1)=0.50, P(0,2)=0.25
    assert p[(2, 0)] == pytest.approx(0.25, abs=1e-6)
    assert p[(1, 1)] == pytest.approx(0.50, abs=1e-6)
    assert p[(0, 2)] == pytest.approx(0.25, abs=1e-6)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-7)


# Also works for the test_bunched_input_state and test_normalization functions
def test_against_perceval(entangling_circuit):
    circuit_to_analyze = entangling_circuit
    unitary = CircuitConverter(circuit_to_analyze).to_tensor([])

    for ind in torch.linspace(0.1, 0.99, 9):
        noise = pcvl.NoiseModel(indistinguishability=ind.item())
        groups = classify_noise(noise)
        noisy_slos = NoisySLOSComputeGraph(
            groups, m=5, n_photons=3, computation_space=ComputationSpace.FOCK
        )

        source = pcvl.Source.from_noise_model(noise)

        backend = pcvl.BackendFactory.get_backend("SLOS")
        sim = pcvl.Simulator(backend)

        sim.set_circuit(circuit_to_analyze)

        for state in Combinadics(scheme="fock", n=3, m=5).enumerate_states():
            noisy_output = noisy_slos.compute_probs(unitary, state)
            # Normalisation check
            assert torch.sum(noisy_output[1]).item() == pytest.approx(1.0, abs=1e-6)

            input_state = pcvl.BasicState(state)

            perceval_probs = sim.probs_svd((source, input_state))

            for out_state, noisy_slos_probability in zip(
                noisy_output[0], noisy_output[1][0], strict=True
            ):
                assert np.allclose(
                    perceval_probs["results"][pcvl.FockState(out_state)],
                    noisy_slos_probability.item(),
                    atol=1e-4,
                )


def test_against_perceval_batched_unitary(entangling_circuit):
    circuit_a = entangling_circuit
    circuit_b = pcvl.Circuit(5, name="5-Mode Alternate Entangled Circuit")
    circuit_b.add((0, 1), pcvl.BS.H())
    circuit_b.add((2, 3), pcvl.BS.H())
    circuit_b.add(0, pcvl.PS(phi=0.5))
    circuit_b.add(1, pcvl.PS(phi=1.2))
    circuit_b.add(3, pcvl.PS(phi=0.4))
    circuit_b.add((1, 2), pcvl.BS.H())
    circuit_b.add((3, 4), pcvl.BS.H())
    circuit_b.add(2, pcvl.PS(phi=2.1))
    circuit_b.add(4, pcvl.PS(phi=0.8))
    circuit_b.add((0, 1), pcvl.BS.H())
    circuit_b.add((1, 2), pcvl.BS.H())
    circuit_b.add((2, 3), pcvl.BS.H())

    unitary_a = CircuitConverter(circuit_a).to_tensor([])
    unitary_b = CircuitConverter(circuit_b).to_tensor([])
    unitary_batch = torch.stack((unitary_a, unitary_b), dim=0)

    noise = pcvl.NoiseModel(indistinguishability=0.4)
    groups = classify_noise(noise)
    noisy_slos = NoisySLOSComputeGraph(
        groups, m=5, n_photons=3, computation_space=ComputationSpace.FOCK
    )

    source_1 = pcvl.Source.from_noise_model(noise)
    source_2 = pcvl.Source.from_noise_model(noise)
    backend_1 = pcvl.BackendFactory.get_backend("SLOS")
    backend_2 = pcvl.BackendFactory.get_backend("SLOS")
    sim_a = pcvl.Simulator(backend_1)
    sim_b = pcvl.Simulator(backend_2)
    sim_a.set_circuit(circuit_a)
    sim_b.set_circuit(circuit_b)

    for state in Combinadics(scheme="fock", n=3, m=5).enumerate_states():
        noisy_keys, noisy_probs = noisy_slos.compute_probs(unitary_batch, state)

        # Batched call should return one probability row per input unitary.
        assert noisy_probs.shape[0] == 2
        assert not torch.allclose(noisy_probs[0], noisy_probs[1], atol=1e-6)

        input_state = pcvl.BasicState(state)
        perceval_probs_a = sim_a.probs_svd((source_1, input_state))
        perceval_probs_b = sim_b.probs_svd((source_2, input_state))
        perceval_vec_a = torch.tensor(
            [
                perceval_probs_a["results"][pcvl.FockState(out_state)]
                for out_state in noisy_keys
            ],
            dtype=noisy_probs.dtype,
        )
        perceval_vec_b = torch.tensor(
            [
                perceval_probs_b["results"][pcvl.FockState(out_state)]
                for out_state in noisy_keys
            ],
            dtype=noisy_probs.dtype,
        )
        assert torch.allclose(noisy_probs[0], perceval_vec_a, atol=1e-4)
        assert torch.allclose(noisy_probs[1], perceval_vec_b, atol=1e-4)


def test_deduplication():
    """
    Bunched input partitions are not double-counted (compare with manual weight)
    """
    test = _InputStateNoisySLOSComputeGraph(
        input_state=[3, 0, 2, 1, 0],
        indistinguishability=0.5,
        computation_space=ComputationSpace.FOCK,
    )
    partitions = test._generate_obb_partition(input_state=test.input_state, order=1)
    assert torch.allclose(
        torch.unique(partitions[0], return_counts=True, dim=0)[1],
        torch.ones_like(partitions[1]),
    )
    assert torch.allclose(
        partitions[0],
        torch.tensor(
            [
                [[2, 0, 2, 1, 0], [1, 0, 0, 0, 0]],
                [[3, 0, 1, 1, 0], [0, 0, 1, 0, 0]],
                [[3, 0, 2, 0, 0], [0, 0, 0, 1, 0]],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        partitions[1],
        torch.tensor(
            [3, 2, 1],
        ),
    )


def test_gradient_flows_through_hom():
    # 2-mode 50/50 beamsplitter
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    unitary = CircuitConverter(circuit).to_tensor([]).to(torch.complex128)

    input_state = [1, 1]

    def f(hom: torch.Tensor) -> torch.Tensor:
        noise_groups = NoiseGroups(
            source={"indistinguishability": hom},
            circuit=None,
            post_measurement=None,
        )

        graph = NoisySLOSComputeGraph(
            noise_groups,
            m=2,
            n_photons=2,
            computation_space=ComputationSpace.FOCK,
            keep_keys=True,
            device=None,
            dtype=torch.float64,
        )

        keys, probs = graph.compute_probs(unitary, input_state)
        key_to_idx = {k: i for i, k in enumerate(keys)}

        # Use coincidence probability as scalar objective
        return probs[0, key_to_idx[(1, 1)]]

    hom = torch.tensor(0.37, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        f,
        (hom,),
        eps=1e-4,
        atol=1e-3,
        rtol=1e-2,
        fast_mode=True,
    )


def test_noisy_slos_to_moves_cached_graph_and_preserves_probs():
    circuit = pcvl.Circuit(2)
    circuit.add((0, 1), pcvl.BS.H())
    unitary = CircuitConverter(circuit).to_tensor([])

    noise_groups = NoiseGroups(
        source={"indistinguishability": 0.35},
        circuit=None,
        post_measurement=None,
    )
    noisy_slos = NoisySLOSComputeGraph(
        noise_groups,
        m=2,
        n_photons=2,
        computation_space=ComputationSpace.FOCK,
        keep_keys=True,
        device=None,
        dtype=torch.float32,
    )

    keys_before, probs_before = noisy_slos.compute_probs(unitary, [1, 1])
    cached_graph = noisy_slos._slos_graph_per_input[(1, 1)]

    moved_graph = noisy_slos.to("cpu")

    assert moved_graph is noisy_slos
    assert noisy_slos.device == torch.device("cpu")
    assert cached_graph.device == torch.device("cpu")
    assert cached_graph._obb_input_states.device.type == "cpu"
    assert all(weight.device.type == "cpu" for weight in cached_graph._weights)
    assert all(
        partition[0].device.type == "cpu" and partition[1].device.type == "cpu"
        for partition in cached_graph._partitions
    )
    assert all(
        states.device.type == "cpu"
        for states in cached_graph._fock_states_per_n.values()
    )
    assert all(graph.device == torch.device("cpu") for graph in noisy_slos._slos_graphs)

    keys_after, probs_after = noisy_slos.compute_probs(unitary, [1, 1])
    assert keys_after == keys_before
    assert torch.allclose(probs_after, probs_before, atol=1e-6)


@pytest.mark.parametrize(
    "unsupported_computation_space",
    [ComputationSpace.UNBUNCHED, ComputationSpace.DUAL_RAIL],
)
def test_computation_space_and_indistinguishability_default_value(
    unsupported_computation_space,
):
    noise = pcvl.NoiseModel(indistinguishability=0.2)
    groups = classify_noise(noise)
    noisy_slos = NoisySLOSComputeGraph(
        groups, m=5, n_photons=3, computation_space=ComputationSpace.FOCK
    )
    assert noisy_slos.computation_space == ComputationSpace.FOCK

    with pytest.warns(
        UserWarning,
        match="Noisy simulations with source noise currently use ComputationSpace.FOCK. Other computation spaces are not yet supported for noise models.",
    ):
        noisy_slos = NoisySLOSComputeGraph(
            groups,
            m=5,
            n_photons=3,
            computation_space=unsupported_computation_space,
        )
    assert noisy_slos.computation_space == ComputationSpace.FOCK

    noise = pcvl.NoiseModel(g2=1.0)
    groups = classify_noise(noise)
    noisy_slos = NoisySLOSComputeGraph(
        groups, m=5, n_photons=3, computation_space=ComputationSpace.FOCK
    )
    assert noisy_slos.indistinguishability == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_noisy_g2_slos_unitary_cuda_graph_cpu():
    noise = pcvl.NoiseModel(indistinguishability=0.2, g2=0.7)
    groups = classify_noise(noise)
    noisy_slos = NoisyG2SLOSComputeGraph(
        groups, m=5, n_photons=3, computation_space=ComputationSpace.FOCK, device=None
    )

    unitary = torch.tensor(
        np.array(pcvl.Matrix.random_unitary(m=5), dtype=np.complex128),
        device=torch.device("cuda"),
    )

    probs = noisy_slos.compute_probs(unitary=unitary, input_state=[1, 0, 1, 0, 1])
    assert probs.sectors[0].tensor.device == unitary.device
