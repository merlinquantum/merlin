import numpy as np
import pytest

from merlin.datasets import DatasetMetadata, spiral


@pytest.mark.parametrize(
    "num_instances,num_features,num_classes",
    [
        (300, 6, 3),  # divisible for exact size
        (400, 10, 4),  # divisible for exact size
    ],
)
def test_spiral_shapes_classes_and_metadata(num_instances, num_features, num_classes):
    X, y, md = spiral.get_data(
        num_instances=num_instances, num_features=num_features, num_classes=num_classes
    )

    # Shapes
    assert X.shape == (num_instances, num_features)
    assert y.shape == (num_instances,)

    # Finite values
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()

    # Class labels
    classes = np.unique(y)
    assert len(classes) == num_classes
    assert set(classes) == set(range(num_classes))

    # Metadata
    assert isinstance(md, DatasetMetadata)
    assert md.name.startswith("Quantum-Inspired Spiral Dataset") or "Spiral" in md.name
    assert md.num_instances == num_instances
    assert md.num_features == num_features
    assert md.num_classes == num_classes
