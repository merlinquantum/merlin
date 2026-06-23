from __future__ import annotations

import inspect

import numpy as np
import perceval as pcvl
import pytest
import torch
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

import merlin
from merlin.models.reservoir_classifier import ReservoirClassifier


def _toy_data():
    X = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.5],
            [0.2, 0.1, 0.9, 0.8],
            [0.8, 0.9, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)
    return X, y


class DummyProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def forward(self, module, inputs):
        self.calls += 1
        return module(inputs)


class FitTransformlessReduction(BaseEstimator):
    def __init__(self, n_components=2) -> None:
        self.n_components = n_components
        self.fit_calls = 0
        self.transform_calls = 0

    def fit(self, X):
        self.fit_calls += 1
        self.offset_ = np.mean(X[:, : self.n_components], axis=0)
        return self

    def transform(self, X):
        self.transform_calls += 1
        return X[:, : self.n_components] - self.offset_


def test_public_import_surface():
    assert merlin.models.ReservoirClassifier is ReservoirClassifier
    assert merlin.ReservoirClassifier is ReservoirClassifier


def test_constructor_hides_layer_configuration_from_public_api():
    signature = inspect.signature(ReservoirClassifier)

    assert "measurement_strategy" not in signature.parameters
    assert "n_modes" not in signature.parameters
    assert "noise_model" not in signature.parameters


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("in_features", 3.2),
        ("out_features", 2.5),
        ("n_photons", 4.5),
        ("in_features", "3"),
        ("n_photons", True),
        ("out_features", 0),
    ],
)
def test_constructor_rejects_non_positive_integer_dimensions(parameter, value):
    kwargs = {
        "in_features": 4,
        "out_features": 2,
        "n_photons": 1,
    }
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        ReservoirClassifier(**kwargs)


def test_to_updates_classifier_dtype_and_delegates_quantum_layer(monkeypatch):
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
    )
    quantum_to_calls = []
    original_quantum_to = model._quantum_layer.to

    def _count_quantum_to(*args, **kwargs):
        quantum_to_calls.append((args, kwargs))
        return original_quantum_to(*args, **kwargs)

    monkeypatch.setattr(model._quantum_layer, "to", _count_quantum_to)

    returned = model.to("cpu", dtype=torch.float64)

    assert returned is model
    assert len(quantum_to_calls) == 1
    assert model.device.type == "cpu"
    assert model.dtype is torch.float64
    assert model._quantum_layer.device.type == "cpu"
    assert model._quantum_layer.dtype is torch.float64
    assert model.readout.weight.dtype is torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_cuda_updates_runtime_device_for_reservoir_inputs(monkeypatch):
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        cache=True,
    )

    model.to("cuda")
    original_forward = model._quantum_layer.forward
    calls = {"count": 0}

    def _assert_cuda_forward(*input_parameters, **kwargs):
        calls["count"] += 1
        assert input_parameters[0].device.type == "cuda"
        output = original_forward(*input_parameters, **kwargs)
        assert output.device.type == "cuda"
        return output

    monkeypatch.setattr(model._quantum_layer, "forward", _assert_cuda_forward)

    model.fit_reservoir(X)
    logits = model.predict(X + 0.05)

    assert calls["count"] == 2
    assert model.device.type == "cuda"
    assert model.layer.device.type == "cuda"
    assert model.readout.weight.device.type == "cuda"
    assert logits.device.type == "cpu"


def test_accepts_fastica_reduction():
    X, y = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=FastICA(n_components=2, random_state=0),
    )

    model.fit_reservoir(X)
    dataset = model.make_dataset(X, y)
    features, targets = dataset.tensors
    logits = model.predict(X)

    assert model.layer.input_size == 2
    assert model.layer.n_modes == 3
    assert features.shape == (len(X), 4 + model.layer.output_size)
    assert targets.shape == (len(y),)
    assert logits.shape == (len(X), 2)


def test_reduction_only_needs_fit_and_transform(monkeypatch):
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=FitTransformlessReduction(n_components=2),
    )

    assert not hasattr(model.reduction, "fit_transform")

    def _fake_encode_quantum(X_reduced_normalized, processor=None):
        del processor
        return torch.as_tensor(X_reduced_normalized, dtype=model.dtype)

    monkeypatch.setattr(model, "_encode_quantum", _fake_encode_quantum)

    model.fit_reservoir(X)

    assert model.reduction.fit_calls == 1
    assert model.reduction.transform_calls == 1


def test_rejects_non_decomposition_reduction():
    class FakeReduction:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    with pytest.raises(TypeError, match="decomposition"):
        ReservoirClassifier(
            in_features=4,
            out_features=2,
            n_photons=1,
            reduction=FakeReduction(),
        )

    with pytest.raises(TypeError, match="decomposition"):
        ReservoirClassifier(
            in_features=4,
            out_features=2,
            n_photons=1,
            reduction=StandardScaler(),
        )


def test_warns_for_large_mode_count_without_reduction():
    with pytest.warns(UserWarning, match="20 modes"):
        ReservoirClassifier(
            in_features=19,
            out_features=2,
            n_photons=1,
            reduction=None,
        )


def test_fit_required_before_use():
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
    )
    X, y = _toy_data()

    with pytest.raises(RuntimeError, match="fit_reservoir"):
        model.make_dataset(X, y)
    with pytest.raises(RuntimeError, match="fit_reservoir"):
        model.predict(X)


def test_parameters_only_expose_readout():
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
    )

    params = list(model.parameters())
    assert len(params) == 2
    assert all(param.requires_grad for param in params)
    assert sum(1 for param in model.layer.parameters() if param.requires_grad) == 0


def test_fit_make_dataset_and_predict_shapes():
    X, y = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        concatenate=True,
        seed=42,
    )

    model.fit_reservoir(X)
    dataset = model.make_dataset(X, y)
    features, targets = dataset.tensors
    logits = model.predict(X)

    assert model.layer.input_size == 2
    assert features.shape == (len(X), 4 + model.layer.output_size)
    assert targets.shape == (len(y),)
    assert logits.shape == (len(X), 2)


def test_concatenate_false_uses_only_quantum_features():
    X, y = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        concatenate=False,
    )

    model.fit_reservoir(X)
    dataset = model.make_dataset(X, y)
    features, _ = dataset.tensors
    assert features.shape == (len(X), model.layer.output_size)


def test_cache_reuses_training_quantum_features(monkeypatch):
    X, y = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        cache=True,
    )
    model.fit_reservoir(X)

    def _fail(*_args, **_kwargs):
        raise AssertionError("Quantum encoding should not run on a cache hit.")

    monkeypatch.setattr(model, "_encode_quantum", _fail)
    dataset = model.make_dataset(X, y)
    assert dataset.tensors[0].shape[0] == len(X)


def test_cache_miss_for_large_input_change_outside_old_sample(monkeypatch):
    X = np.arange(2001 * 4, dtype=np.float32).reshape(2001, 4)
    X_changed = X.copy()
    X_changed[1, 0] += 10.0

    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=None,
        concatenate=False,
        cache=True,
    )
    calls = {"count": 0}

    def _fake_encode_quantum(X_reduced_normalized, processor=None):
        del processor
        calls["count"] += 1
        return torch.as_tensor(X_reduced_normalized, dtype=model.dtype)

    monkeypatch.setattr(model, "_encode_quantum", _fake_encode_quantum)

    model.fit_reservoir(X)
    model.transform_reservoir(X_changed)

    assert calls["count"] == 2


def test_cache_false_always_encodes(monkeypatch):
    X, y = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        cache=False,
    )

    calls = {"count": 0}
    original = model._encode_quantum

    def _count(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_encode_quantum", _count)
    model.fit_reservoir(X)
    assert calls["count"] == 0
    assert model._quantum_mean is None
    assert model._quantum_std is None

    model.make_dataset(X, y)
    assert calls["count"] == 1
    assert model._quantum_mean is not None
    assert model._quantum_std is not None

    model.make_dataset(X, y)
    assert calls["count"] == 2


def test_cache_false_requires_training_transform_before_new_inputs():
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        cache=False,
    )

    model.fit_reservoir(X)

    with pytest.raises(RuntimeError, match="training data"):
        model.transform_reservoir(X + 0.05)


def test_transform_reservoir_returns_quantum_embeddings_only():
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        concatenate=True,
        cache=True,
    )

    model.fit_reservoir(X)
    embeddings = model.transform_reservoir(X)
    features = model._make_feature_tensor(X)

    assert embeddings.shape == (len(X), model.layer.output_size)
    assert features.shape == (len(X), model.in_features + model.layer.output_size)
    assert torch.allclose(embeddings, features[:, model.in_features :].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_encode_quantum_computes_features_on_cuda():
    device = torch.device("cuda")
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        device=device,
    )
    reduced_normalized = np.array([[0.0], [1.0]], dtype=np.float32)

    quantum_features = model._encode_quantum(reduced_normalized)

    assert model.device.type == "cuda"
    assert model.layer.device.type == "cuda"
    assert next(model.readout.parameters()).device.type == "cuda"
    assert quantum_features.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_reservoir_uses_cuda_then_keeps_cache_on_cpu(monkeypatch):
    X, y = _toy_data()
    device = torch.device("cuda")
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        device=device,
        cache=True,
    )
    original_forward = model._quantum_layer.forward
    calls = {"count": 0}

    def _assert_cuda_forward(*input_parameters, **kwargs):
        calls["count"] += 1
        assert input_parameters[0].device.type == "cuda"
        output = original_forward(*input_parameters, **kwargs)
        assert output.device.type == "cuda"
        return output

    monkeypatch.setattr(model._quantum_layer, "forward", _assert_cuda_forward)

    model.fit_reservoir(X)
    embeddings = model.transform_reservoir(X)
    dataset = model.make_dataset(X, y)
    features, targets = dataset.tensors

    assert calls["count"] == 1
    assert model._fit_quantum_cache is not None
    assert model._fit_quantum_cache.device.type == "cpu"
    assert model._quantum_mean is not None
    assert model._quantum_mean.device.type == "cpu"
    assert model._quantum_std is not None
    assert model._quantum_std.device.type == "cpu"
    assert embeddings.device.type == "cpu"
    assert features.device.type == "cpu"
    assert targets.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_predict_uses_cuda_reservoir_and_returns_cpu_logits(monkeypatch):
    X, _ = _toy_data()
    device = torch.device("cuda")
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        device=device,
        cache=False,
    )
    model.fit_reservoir(X)
    _ = model.transform_reservoir(X)

    original_forward = model._quantum_layer.forward
    calls = {"count": 0}

    def _assert_cuda_forward(*input_parameters, **kwargs):
        calls["count"] += 1
        assert input_parameters[0].device.type == "cuda"
        output = original_forward(*input_parameters, **kwargs)
        assert output.device.type == "cuda"
        return output

    monkeypatch.setattr(model._quantum_layer, "forward", _assert_cuda_forward)

    logits = model.predict(X + 0.05)

    assert calls["count"] == 1
    assert logits.device.type == "cpu"
    assert next(model.readout.parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_device_override_restores_reservoir_on_cuda(tmp_path):
    X, _ = _toy_data()
    device = torch.device("cuda")
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
    )
    model.fit_reservoir(X)

    path = tmp_path / "reservoir_cpu.pt"
    model.save(path)
    restored = ReservoirClassifier.load(path, device=device)
    logits = restored.predict(X)

    assert restored.device.type == "cuda"
    assert restored.layer.device.type == "cuda"
    assert next(restored.readout.parameters()).device.type == "cuda"
    assert logits.device.type == "cpu"


def test_processor_is_used_for_cache_false_transform_and_predict():
    X, _ = _toy_data()
    processor = DummyProcessor()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        cache=False,
    )

    model.fit_reservoir(X, processor=processor)
    model.transform_reservoir(X, processor=processor)
    _ = model.predict(X + 0.05, processor=processor)
    assert processor.calls == 2


def test_layer_processor_is_used_for_cache_false_transform_and_predict():
    X, _ = _toy_data()
    processor = DummyProcessor()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        cache=False,
    )

    model.layer.processor = processor
    model.fit_reservoir(X)
    model.transform_reservoir(X)
    _ = model.predict(X + 0.05)

    assert processor.calls == 2


def test_processor_argument_overrides_layer_processor_with_warning():
    X, _ = _toy_data()
    argument_processor = DummyProcessor()
    layer_processor = DummyProcessor()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        cache=False,
    )
    model.layer.processor = layer_processor

    model.fit_reservoir(X, processor=argument_processor)
    assert argument_processor.calls == 0
    assert layer_processor.calls == 0

    with pytest.warns(UserWarning, match="processor argument"):
        model.transform_reservoir(X, processor=argument_processor)
    with pytest.warns(UserWarning, match="processor argument"):
        _ = model.predict(X + 0.05, processor=argument_processor)

    assert argument_processor.calls == 2
    assert layer_processor.calls == 0


def test_fit_reservoir_works_with_local_perceval_aprocessor_backend():
    X, _ = _toy_data()
    direct_model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        cache=True,
        seed=13,
    )
    processor_model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        cache=True,
        seed=13,
    )
    processor = merlin.MerlinProcessor(processor=pcvl.Processor("SLOS"))

    direct_model.fit_reservoir(X)
    processor_model.fit_reservoir(X, processor=processor)

    assert processor.backend_kind == "local_processor"
    assert processor.available_commands == ("probs",)
    assert processor_model._fit_quantum_cache is not None
    assert direct_model._fit_quantum_cache is not None
    torch.testing.assert_close(
        processor_model._fit_quantum_cache,
        direct_model._fit_quantum_cache,
        rtol=1e-6,
        atol=1e-6,
    )


def test_layer_processor_runs_cache_false_reservoir_with_local_perceval_aprocessor():
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=1),
        concatenate=False,
        cache=False,
        seed=17,
    )
    model.layer.processor = merlin.MerlinProcessor(processor=pcvl.Processor("SLOS"))

    model.fit_reservoir(X)
    embeddings = model.transform_reservoir(X)
    logits = model.predict(X + 0.05)

    assert embeddings.shape == (len(X), model.layer.output_size)
    assert logits.shape == (len(X), 2)
    assert model._quantum_mean is not None
    assert model._quantum_std is not None
    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(logits).all()


def test_layer_processor_survives_reservoir_rebuild():
    processor = DummyProcessor()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=None,
    )

    model.layer.processor = processor
    model.layer.n_modes = 5

    assert model.layer.processor is processor


def test_layer_processor_is_not_saved(tmp_path):
    processor = DummyProcessor()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
    )
    model.layer.processor = processor

    path = tmp_path / "reservoir_processor.pt"
    model.save(path)
    restored = ReservoirClassifier.load(path)

    assert restored.layer.processor is None


def test_save_load_roundtrip(tmp_path):
    X, y = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        seed=7,
    )
    model.fit_reservoir(X)
    dataset = model.make_dataset(X, y)
    features, _ = dataset.tensors

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    logits = model(features)
    loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y))
    loss.backward()
    optimizer.step()

    before = model.predict(X)
    path = tmp_path / "reservoir.pt"
    model.save(path)
    restored = ReservoirClassifier.load(path)
    after = restored.predict(X)

    assert torch.allclose(before, after, atol=1e-6)
    assert restored._is_fitted is True


def test_save_load_preserves_custom_mode_count(tmp_path):
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=2,
        reduction=None,
    )
    model.layer.n_modes = 6

    path = tmp_path / "reservoir_modes.pt"
    model.save(path)
    restored = ReservoirClassifier.load(path)

    assert restored.layer.n_modes == 6
    assert restored.layer.input_size == 4
    assert restored.quantum_input_features == 6
    assert restored.encoded_input_features == 4


def test_save_load_preserves_measurement_strategy_override(tmp_path):
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=2,
        reduction=None,
    )
    strategy = merlin.MeasurementStrategy.probs(
        computation_space=merlin.ComputationSpace.FOCK,
    )
    model.layer.measurement_strategy = strategy

    path = tmp_path / "reservoir_strategy.pt"
    model.save(path)
    restored = ReservoirClassifier.load(path)

    assert (
        restored.layer.measurement_strategy.computation_space
        == merlin.ComputationSpace.FOCK
    )
    assert restored.layer.output_size == 15


def test_same_seed_same_predictions():
    X, _ = _toy_data()
    model_a = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        seed=123,
    )
    model_b = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
        seed=123,
    )

    model_a.fit_reservoir(X)
    model_b.fit_reservoir(X)

    assert np.allclose(model_a._unitary_matrix, model_b._unitary_matrix)
    assert torch.allclose(model_a.predict(X), model_b.predict(X), atol=1e-6)


def test_layer_measurement_strategy_rebuilds_layer_and_invalidates_fit():
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=2,
        reduction=None,
    )
    model.fit_reservoir(X)

    old_output_size = model.layer.output_size
    model.layer.measurement_strategy = merlin.MeasurementStrategy.probs(
        computation_space=merlin.ComputationSpace.FOCK
    )

    assert model.layer.output_size != old_output_size
    assert model.layer.output_size == 15
    assert model._is_fitted is False
    assert model._fit_quantum_cache is None


def test_grouped_measurement_strategy_updates_lazylinear_input_width():
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=2,
        reduction=None,
        concatenate=True,
    )
    model.fit_reservoir(X)

    assert model.layer.n_modes == 5
    assert model.layer.output_size == 10
    assert model.readout.in_features == 14

    model.layer.measurement_strategy = merlin.MeasurementStrategy.probs(
        computation_space=merlin.ComputationSpace.UNBUNCHED,
        grouping=merlin.ModGrouping(10, 3),
    )
    model.fit_reservoir(X)
    grouped_features = model._make_feature_tensor(X)

    assert grouped_features.shape[1] == 7
    assert model.readout.in_features == 7
    assert model.predict(X).shape == (len(X), 2)


def test_layer_n_modes_rebuilds_layer_without_mutating_reduction():
    X, _ = make_blobs(
        n_samples=20,
        n_features=6,
        centers=2,
        random_state=0,
    )
    X = X.astype(np.float32)
    model = ReservoirClassifier(
        in_features=6,
        out_features=2,
        n_photons=2,
        reduction=PCA(n_components=4),
    )
    model.fit_reservoir(X)

    model.layer.n_modes = 6

    assert model.quantum_input_features == 6
    assert model.layer.n_modes == 6
    assert model.layer.input_size == 4
    assert model.encoded_input_features == 4
    assert model._reduction_template.n_components == 4
    assert model._is_fitted is False
    assert model._fit_quantum_cache is None


def test_layer_n_modes_without_reduction_can_grow_modes():
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=2,
        reduction=None,
    )

    model.layer.n_modes = 6

    assert model.quantum_input_features == 6
    assert model.layer.n_modes == 6
    assert model.layer.input_size == 4
    assert model.encoded_input_features == 4


def test_layer_n_modes_cannot_shrink_below_circuit_width():
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=2,
        reduction=None,
    )

    with pytest.raises(ValueError, match="encoded input features plus one"):
        model.layer.n_modes = 4


def test_layer_noise_model_rebuilds_layer_and_invalidates_fit():
    X, _ = _toy_data()
    model = ReservoirClassifier(
        in_features=4,
        out_features=2,
        n_photons=1,
        reduction=PCA(n_components=2),
    )
    model.fit_reservoir(X)

    noise_model = pcvl.NoiseModel(brightness=0.9)
    model.layer.noise = noise_model

    assert model.layer.noise is noise_model
    assert model._quantum_layer.noise is noise_model
    assert model._is_fitted is False
    assert model._fit_quantum_cache is None


def test_reservoir_memory_cache_stays_bounded(monkeypatch):
    # Use a larger synthetic dataset than the unit tests above so the cache path
    # is exercised with a non-trivial number of samples.
    X, y = make_blobs(
        n_samples=4000,
        n_features=64,
        centers=4,
        random_state=0,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    model = ReservoirClassifier(
        in_features=64,
        out_features=4,
        n_photons=1,
        reduction=PCA(n_components=8),
        cache=True,
    )

    quantum_width = 32

    # Replace the quantum forward by a deterministic lightweight tensor
    # expansion so this test measures the cache behavior rather than the
    # runtime cost of the photonic simulation itself.
    def _fake_encode_quantum(X_reduced_normalized, processor=None):
        del processor
        reduced = torch.as_tensor(X_reduced_normalized, dtype=model.dtype)
        repeats = (quantum_width + reduced.shape[1] - 1) // reduced.shape[1]
        tiled = reduced.repeat(1, repeats)
        return tiled[:, :quantum_width]

    monkeypatch.setattr(model, "_encode_quantum", _fake_encode_quantum)

    model.fit_reservoir(X)
    dataset = model.make_dataset(X, y)
    features, targets = dataset.tensors

    assert model._fit_quantum_cache is not None
    # The cached tensor should occupy exactly N x quantum_width float entries,
    # with no hidden copy of the raw input dataset kept on the model.
    cache_bytes = (
        model._fit_quantum_cache.element_size() * model._fit_quantum_cache.numel()
    )
    expected_bytes = (
        X.shape[0] * quantum_width * torch.tensor([], dtype=model.dtype).element_size()
    )

    assert cache_bytes == expected_bytes
    # Keep the threshold modest: this is a regression guard against accidental
    # memory blow-ups, not a benchmark of the real photonic workload.
    assert cache_bytes < 1024 * 1024  # ~0.5 MB for this dummy configuration
    assert not hasattr(model, "_fit_raw_cache")
    assert features.shape == (len(X), X.shape[1] + quantum_width)
    assert targets.shape == (len(y),)
