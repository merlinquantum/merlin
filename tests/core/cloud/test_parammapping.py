"""
tests/core/cloud/test_parammapping.py

Covers input parameter ordering consistency across:

Perceval user-built circuits:
  Perceval Circuit -> CircuitConverter.spec_mappings -> QuantumLayer.export_config()["input_param_order"]
  -> MerlinProcessor._extract_input_params -> correct column routing

CircuitBuilder declarative circuits:
  CircuitBuilder -> QuantumLayer(builder=...) -> export_config()["input_param_order"]
  -> MerlinProcessor routing + trainables do not leak into input params

Cloud tests (real): local forward vs remote MerlinProcessor.forward equivalence
for >10 modes, for both UNBUNCHED and FOCK, and for multi-prefix circuits.

Run:
  pytest tests/core/cloud/test_parammapping.py -v

Cloud:
  pytest --run-cloud-tests tests/core/cloud/test_parammapping.py -k Cloud -v
"""

from __future__ import annotations

from math import comb
from unittest.mock import MagicMock, patch

import numpy as np
import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor, ValidatedLayerConfig
from merlin.measurement import MeasurementStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_dist_size(space: ComputationSpace, m: int, n: int) -> int:
    """
    Distribution size for:
      - UNBUNCHED: collision-free states => C(m, n)
      - FOCK: bunched states => C(m+n-1, n)
    """
    if space == ComputationSpace.UNBUNCHED:
        return comb(m, n)
    if space == ComputationSpace.FOCK:
        return comb(m + n - 1, n)
    raise ValueError(f"Unsupported computation space: {space}")


def _make_perceval_circuit_single_prefix(
    n_logical: int, n_physical: int | None = None, prefix: str = "px"
):
    if n_physical is None:
        n_physical = n_logical

    U = pcvl.Matrix.random_unitary(n=n_physical)
    c_var = pcvl.Circuit(n_physical)
    for i in range(n_logical):
        c_var.add(i, pcvl.PS(pcvl.P(f"{prefix}{i + 1}")))
    return pcvl.Unitary(U) // c_var // pcvl.Unitary(U.copy())


def _make_perceval_layer(
    n_logical: int,
    n_physical: int | None = None,
    n_photons: int = 1,
    prefix: str = "px",
    input_parameters: list[str] | None = None,
    computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
):
    if n_physical is None:
        n_physical = n_logical

    circuit = _make_perceval_circuit_single_prefix(n_logical, n_physical, prefix=prefix)

    if n_photons <= 0:
        raise ValueError("n_photons must be >= 1")

    # Deterministic photon placement.
    input_state = [0] * n_physical
    if n_photons == 1:
        input_state[0] = 1
    else:
        step = (n_physical - 1) / (n_photons - 1)
        for k in range(n_photons):
            input_state[int(round(k * step))] = 1

    if input_parameters is None:
        input_parameters = [prefix]

    return QuantumLayer(
        input_size=n_logical,
        circuit=circuit,
        trainable_parameters=[],
        input_parameters=input_parameters,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space
        ),
    ).eval()


def _make_perceval_layer_two_prefixes(prefixes: list[str], counts: list[int], m: int):
    """
    Used for unit tests around ordering. Keeps 1 photon (local-only tests).
    """
    assert len(prefixes) == len(counts)
    c = pcvl.Circuit(m)

    mode = 0
    for pref, n in zip(prefixes, counts, strict=True):
        for i in range(n):
            c.add(mode, pcvl.PS(pcvl.P(f"{pref}{i + 1}")))
            mode += 1

    input_state = [0] * m
    input_state[0] = 1

    return QuantumLayer(
        input_size=sum(counts),
        circuit=c,
        trainable_parameters=[],
        input_parameters=list(prefixes),
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.UNBUNCHED
        ),
    ).eval()


def _make_builder_layer(
    n_modes: int,
    *,
    include_trainable: bool = True,
    scale: float = 1.0,
    n_photons: int = 1,
    computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
):
    b = CircuitBuilder(n_modes=n_modes)
    if include_trainable:
        b.add_entangling_layer(trainable=True, name="W")
    b.add_angle_encoding(modes=list(range(n_modes)), name="px", scale=scale)

    layer = QuantumLayer(
        input_size=n_modes,
        builder=b,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=computation_space
        ),
    ).eval()
    return layer, b


def _mock_processor() -> MerlinProcessor:
    mock_rp = MagicMock(spec=pcvl.RemoteProcessor)
    mock_rp.name = "sim:slos"
    mock_rp.available_commands = ["probs"]
    mock_rp.proxies = None
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value="fake"):
        return MerlinProcessor(remote_processor=mock_rp)


def _expected_from_converter(layer: QuantumLayer) -> list[str]:
    cc = layer.computation_process.converter.spec_mappings
    out: list[str] = []
    for p in list(layer.input_parameters or []):
        out.extend(cc.get(p, []))
    return out


def _assert_config_contract(cfg: ValidatedLayerConfig):
    assert isinstance(cfg.input_param_order, list)
    assert all(isinstance(x, str) for x in cfg.input_param_order)


# ---------------------------------------------------------------------------
# 1) Perceval user-built circuits: numeric order + prefix order + routing
# ---------------------------------------------------------------------------


class TestPercevalUserBuilt:
    @pytest.mark.parametrize("n", [3, 5, 10, 12])
    def test_export_matches_converter_and_is_numeric(self, n):
        layer = _make_perceval_layer(n, prefix="px")
        cfg = ValidatedLayerConfig(layer.export_config())
        _assert_config_contract(cfg)

        assert cfg.input_param_order == _expected_from_converter(layer)
        assert cfg.input_param_order == [f"px{i + 1}" for i in range(n)]

    def test_two_prefixes_12_each(self):
        layer = _make_perceval_layer_two_prefixes(
            prefixes=["a", "b"], counts=[12, 12], m=24
        )
        cfg = ValidatedLayerConfig(layer.export_config())
        _assert_config_contract(cfg)

        expected = [f"a{i + 1}" for i in range(12)] + [f"b{i + 1}" for i in range(12)]
        assert cfg.input_param_order == expected

    def test_reversed_prefix_order(self):
        layer = _make_perceval_layer_two_prefixes(
            prefixes=["beta", "alpha"], counts=[4, 4], m=8
        )
        cfg = ValidatedLayerConfig(layer.export_config())
        _assert_config_contract(cfg)

        expected = [f"beta{i + 1}" for i in range(4)] + [
            f"alpha{i + 1}" for i in range(4)
        ]
        assert cfg.input_param_order == expected

    @pytest.mark.parametrize("n", [10, 12])
    def test_merlinprocessor_extract_and_route(self, n):
        layer = _make_perceval_layer(n, prefix="px")
        cfg = ValidatedLayerConfig(layer.export_config())
        proc = _mock_processor()

        names = proc._extract_input_params(cfg)
        assert names == cfg.input_param_order

        row = np.array([(j + 1) * 0.1 for j in range(n)], dtype=float)
        params = {name: float(row[j]) for j, name in enumerate(names)}

        assert params["px1"] == pytest.approx(0.1)
        assert params["px2"] == pytest.approx(0.2)
        assert params["px10"] == pytest.approx(1.0)
        assert params[f"px{n}"] == pytest.approx(n * 0.1)

    def test_user_scenario_2ph_12logical_24modes(self):
        layer = _make_perceval_layer(12, n_physical=24, n_photons=2, prefix="px")
        cfg = ValidatedLayerConfig(layer.export_config())
        proc = _mock_processor()
        names = proc._extract_input_params(cfg)

        assert names == [f"px{i + 1}" for i in range(12)]

        row = np.array([(j + 1) * 0.1 for j in range(12)], dtype=float)
        params = {name: float(row[j]) for j, name in enumerate(names)}
        assert params["px2"] == pytest.approx(0.2)
        assert params["px10"] == pytest.approx(1.0)
        assert params["px12"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# 2) Builder declarative circuits: export order + routing + no trainable leakage
# ---------------------------------------------------------------------------


class TestBuilderDeclarative:
    @pytest.mark.parametrize("n", [5, 10, 12])
    def test_builder_export_matches_converter_and_routes(self, n):
        layer, _b = _make_builder_layer(n, include_trainable=True, scale=1.0)
        cfg = ValidatedLayerConfig(layer.export_config())
        _assert_config_contract(cfg)

        assert cfg.input_param_order == _expected_from_converter(layer)
        assert len(cfg.input_param_order) == n

        proc = _mock_processor()
        names = proc._extract_input_params(cfg)
        assert names == cfg.input_param_order

        row = np.array([(j + 1) * 0.1 for j in range(n)], dtype=float)
        params = {name: float(row[j]) for j, name in enumerate(names)}
        for j, name in enumerate(names):
            assert params[name] == pytest.approx((j + 1) * 0.1)

    def test_builder_no_trainable_leakage(self):
        layer, _b = _make_builder_layer(12, include_trainable=True, scale=1.0)
        cfg = ValidatedLayerConfig(layer.export_config())
        _assert_config_contract(cfg)

        for name in cfg.input_param_order:
            assert not name.startswith(
                "W"
            ), f"trainable leaked into input_param_order: {name}"

    def test_builder_input_only_no_trainables(self):
        layer, _b = _make_builder_layer(8, include_trainable=False, scale=1.0)
        cfg = ValidatedLayerConfig(layer.export_config())
        _assert_config_contract(cfg)

        assert cfg.input_param_order == _expected_from_converter(layer)
        assert len(cfg.input_param_order) == 8

    def test_builder_scale_does_not_change_order(self):
        layer1, b1 = _make_builder_layer(10, include_trainable=True, scale=1.0)
        layer2, b2 = _make_builder_layer(10, include_trainable=True, scale=2.5)

        cfg1 = layer1.export_config()
        cfg2 = layer2.export_config()
        _assert_config_contract(ValidatedLayerConfig(cfg1))
        _assert_config_contract(ValidatedLayerConfig(cfg2))

        assert cfg1["input_param_order"] == cfg2["input_param_order"]
        assert cfg1["input_param_order"] == _expected_from_converter(layer1)
        assert cfg2["input_param_order"] == _expected_from_converter(layer2)

        # builder bookkeeping
        assert "px" in b2._angle_encoding_scales
        assert b2._angle_encoding_scales["px"] == dict.fromkeys(range(10), 2.5)


# ---------------------------------------------------------------------------
# 3) Cloud tests — both UNBUNCHED and FOCK (>10 modes)
# ---------------------------------------------------------------------------


class TestCloudBothSpaces:
    """
    Real cloud tests for both UNBUNCHED and FOCK with >10 modes.

    Run with:
      pytest --run-cloud-tests tests/core/cloud/test_parammapping.py -k Cloud -v
    """

    @pytest.mark.parametrize(
        "space", [ComputationSpace.UNBUNCHED, ComputationSpace.FOCK]
    )
    @pytest.mark.parametrize(
        "n_modes,n_photons",
        [
            (11, 2),
            (12, 2),
            (15, 2),
            (12, 3),
        ],
    )
    def test_perceval_direct_local_vs_remote(
        self,
        remote_processor,
        space,
        n_modes,
        n_photons,
    ):
        # cloud requires >=2 photons
        n_photons = max(2, int(n_photons))

        layer = _make_perceval_layer(
            n_logical=n_modes,
            n_physical=n_modes,
            n_photons=n_photons,
            prefix="px",
            computation_space=space,
        )

        expected_dist = _expected_dist_size(space, n_modes, n_photons)

        torch.manual_seed(42)
        X = torch.rand(4, n_modes) * torch.pi

        y_local = layer(X)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        y_remote = proc.forward(layer, X, nsample=10_000)

        assert y_local.shape == (4, expected_dist)
        assert y_remote.shape == (4, expected_dist)

        assert torch.allclose(y_local.sum(dim=1), torch.ones(4), atol=1e-4)
        assert torch.allclose(y_remote.sum(dim=1), torch.ones(4), atol=0.1)

        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.1

    @pytest.mark.parametrize(
        "space", [ComputationSpace.UNBUNCHED, ComputationSpace.FOCK]
    )
    @pytest.mark.parametrize(
        "n_modes,n_photons",
        [
            (11, 2),
            (12, 2),
            (15, 2),
            (12, 3),
        ],
    )
    def test_builder_local_vs_remote(
        self,
        remote_processor,
        space,
        n_modes,
        n_photons,
    ):
        n_photons = max(2, int(n_photons))

        layer, _b = _make_builder_layer(
            n_modes,
            include_trainable=True,
            scale=1.0,
            n_photons=n_photons,
            computation_space=space,
        )

        expected_dist = _expected_dist_size(space, n_modes, n_photons)

        torch.manual_seed(123)
        X = torch.rand(4, n_modes) * torch.pi

        y_local = layer(X)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        y_remote = proc.forward(layer, X, nsample=10_000)

        assert y_local.shape == (4, expected_dist)
        assert y_remote.shape == (4, expected_dist)

        assert torch.allclose(y_local.sum(dim=1), torch.ones(4), atol=1e-4)
        assert torch.allclose(y_remote.sum(dim=1), torch.ones(4), atol=0.1)

        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.1


# ---------------------------------------------------------------------------
# 4) Cloud tests — multi-prefix circuits (>10 each), both spaces
# ---------------------------------------------------------------------------


class TestCloudMultiPrefixBothSpaces:
    """
    Multi-prefix circuits with >=10 params per prefix.
    Tests both UNBUNCHED and FOCK.

    Run with:
      pytest --run-cloud-tests tests/core/cloud/test_parammapping.py -k MultiPrefix -v
    """

    @pytest.mark.parametrize(
        "space", [ComputationSpace.UNBUNCHED, ComputationSpace.FOCK]
    )
    @pytest.mark.parametrize("n_each", [10, 12])
    def test_multi_prefix_local_vs_remote(
        self,
        remote_processor,
        space,
        n_each,
    ):
        total_modes = 2 * n_each
        n_photons = 2  # cloud-acceptable minimum

        c = pcvl.Circuit(total_modes)
        for i in range(n_each):
            c.add(i, pcvl.PS(pcvl.P(f"a{i + 1}")))
        for i in range(n_each):
            c.add(i + n_each, pcvl.PS(pcvl.P(f"b{i + 1}")))

        input_state = [0] * total_modes
        input_state[0] = 1
        input_state[-1] = 1

        layer = QuantumLayer(
            input_size=total_modes,
            circuit=c,
            trainable_parameters=[],
            input_parameters=["a", "b"],
            input_state=input_state,
            measurement_strategy=MeasurementStrategy.probs(computation_space=space),
        ).eval()

        expected_dist = _expected_dist_size(space, total_modes, n_photons)

        torch.manual_seed(123)
        X = torch.rand(4, total_modes) * torch.pi

        y_local = layer(X)

        proc = MerlinProcessor(remote_processor, timeout=300.0)
        y_remote = proc.forward(layer, X, nsample=10_000)

        assert y_local.shape == (4, expected_dist)
        assert y_remote.shape == (4, expected_dist)

        assert torch.allclose(y_local.sum(dim=1), torch.ones(4), atol=1e-4)
        assert torch.allclose(y_remote.sum(dim=1), torch.ones(4), atol=0.1)

        diff = (y_local - y_remote).abs().mean().item()
        assert diff < 0.1
