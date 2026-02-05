import perceval as pcvl
import pytest
import torch

from merlin import ComputationSpace, MeasurementStrategy, QuantumLayer
from merlin.bridge.quantum_bridge import QuantumBridge

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU not available"
)


def make_identity_layer(m: int, n_photons: int, device: torch.device) -> QuantumLayer:
    c = pcvl.Circuit(m)  # identity unitary
    return QuantumLayer(
        circuit=c,
        n_photons=n_photons,
        device=device,
        dtype=torch.float32,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED
        ),
        amplitude_encoding=True,
    )


def cpu_gpu_bridge_outputs(groups, statevec_fn, batch: int = 1):
    m = sum(2**g for g in groups)
    n_photons = len(groups)

    # CPU layer + bridge
    layer_cpu = make_identity_layer(m, n_photons, torch.device("cpu"))
    bridge_cpu = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=n_photons,
        wires_order="little",
        normalize=True,
        device=torch.device("cpu"),
        computation_space=ComputationSpace.UNBUNCHED,
    )

    # GPU layer + bridge
    device_gpu = torch.device("cuda")
    layer_gpu = make_identity_layer(m, n_photons, device_gpu)
    bridge_gpu = QuantumBridge(
        qubit_groups=groups,
        n_modes=m,
        n_photons=n_photons,
        wires_order="little",
        normalize=True,
        device=device_gpu,
        computation_space=ComputationSpace.UNBUNCHED,
    )

    psi_cpu = statevec_fn(torch.device("cpu"), batch)
    psi_gpu = statevec_fn(device_gpu, batch)

    payload_cpu = bridge_cpu(psi_cpu)
    payload_gpu = bridge_gpu(psi_gpu)

    out_cpu = layer_cpu(payload_cpu).detach().cpu()
    out_gpu = layer_gpu(payload_gpu)
    assert out_gpu.is_cuda, f"Expected output on CUDA, got {out_gpu.device}"
    return out_cpu, out_gpu.detach().cpu()


def test_bridge_gpu_matches_cpu_basis_state():
    groups = [1, 1]

    def statevec_fn(device, batch):
        psi = torch.zeros(4, dtype=torch.complex64, device=device)
        psi[1] = 1.0 + 0.0j  # |01>
        if batch > 1:
            return psi.unsqueeze(0).expand(batch, -1).clone()
        return psi

    out_cpu, out_gpu = cpu_gpu_bridge_outputs(groups, statevec_fn, batch=1)
    assert out_cpu.shape == out_gpu.shape
    torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-5, atol=1e-6)


def test_bridge_gpu_matches_cpu_superposition_batched():
    groups = [1, 1]

    def statevec_fn(device, batch):
        psi = torch.zeros(4, dtype=torch.complex64, device=device)
        psi[0] = 1.0 + 0.0j
        psi[3] = 0.0 + 1.0j
        if batch > 1:
            return psi.unsqueeze(0).expand(batch, -1).clone()
        return psi

    out_cpu, out_gpu = cpu_gpu_bridge_outputs(groups, statevec_fn, batch=3)
    assert out_cpu.shape == out_gpu.shape
    torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-5, atol=1e-6)
