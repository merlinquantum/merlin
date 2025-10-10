import numpy as np

from merlin.datasets import DatasetMetadata, iris


def test_iris_train_test_shapes_and_ranges():
    X_train, y_train, md_train = iris.get_data_train()
    X_test, y_test, md_test = iris.get_data_test()

    # Shapes
    assert X_train.shape == (120, 4)
    assert y_train.shape == (120,)
    assert X_test.shape == (30, 4)
    assert y_test.shape == (30,)

    # Value ranges after Min-Max scaling
    assert np.all(X_train >= 0.0) and np.all(X_train <= 1.0)
    assert np.all(X_test >= 0.0) and np.all(X_test <= 1.0)

    # Labels
    assert set(np.unique(y_train)).issubset({0, 1, 2})
    assert set(np.unique(y_test)).issubset({0, 1, 2})

    # Metadata
    assert isinstance(md_train, DatasetMetadata)
    assert isinstance(md_test, DatasetMetadata)
    assert md_train.name == "Iris Plants Dataset"
    assert md_test.name == "Iris Plants Dataset"
    assert md_train.num_classes == 3
    assert md_test.num_classes == 3
    assert md_train.num_features == 4
    assert md_test.num_features == 4
    assert md_train.subset == "train"
    assert md_test.subset == "test"
    assert md_train.normalization is not None
    assert md_train.normalization.method == "min-max"
    assert tuple(md_train.normalization.range) == (0, 1)
    assert md_train.normalization.per_feature is True
