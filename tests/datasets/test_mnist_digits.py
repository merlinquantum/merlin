from pathlib import Path

import numpy as np
import pandas as pd

from merlin.datasets import DatasetMetadata, fashion_mnist, k_mnist, mnist_digits, utils

# --- Helpers to build tiny IDX files (no network) ---


def _write_idx_file(path: Path, dtype_code: int, dims: list[int], data: np.ndarray):
    """Write a minimal IDX file at path.

    dtype_code:
      0x08: unsigned byte, 0x09: signed byte, 0x0B: >i2, 0x0C: >i4, 0x0D: >f4, 0x0E: >f8
    dims: list of dimension sizes
    data: flat numpy array with correct dtype and matching prod(dims)
    """
    # Magic number: 00 00 <dtype_code> <ndims>
    ndims = len(dims)
    with open(path, "wb") as f:
        f.write((0).to_bytes(1, "big"))
        f.write((0).to_bytes(1, "big"))
        f.write(dtype_code.to_bytes(1, "big"))
        f.write(ndims.to_bytes(1, "big"))
        for d in dims:
            f.write(int(d).to_bytes(4, "big"))
        # Write raw data buffer
        f.write(data.tobytes(order="C"))


def _make_tiny_mnist_image_file(path: Path, n: int = 2, h: int = 3, w: int = 3):
    dims = [n, h, w]
    # uint8 image values 0..255
    arr = (np.arange(n * h * w) % 256).astype(np.uint8)
    _write_idx_file(path, 0x08, dims, arr)


def _make_tiny_mnist_label_file(path: Path, n: int = 2):
    dims = [n]
    labels = (np.arange(n) % 10).astype(np.uint8)
    _write_idx_file(path, 0x08, dims, labels)


def _make_percevalquest_csv(path: Path, n: int):
    # Build CSV with 28x28 flattened pixel list as string and a label column
    pixels = [0.0] * (28 * 28)
    rows = [{"image": str(pixels), "label": int(i % 10)} for i in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_mnist_like_datasets_apis_offline(tmp_path: Path, monkeypatch):
    # Create tiny offline IDX files
    train_images = tmp_path / "train-images-idx3-ubyte"
    train_labels = tmp_path / "train-labels-idx1-ubyte"
    test_images = tmp_path / "t10k-images-idx3-ubyte"
    test_labels = tmp_path / "t10k-labels-idx1-ubyte"

    _make_tiny_mnist_image_file(train_images, n=3, h=3, w=3)
    _make_tiny_mnist_label_file(train_labels, n=3)
    _make_tiny_mnist_image_file(test_images, n=2, h=3, w=3)
    _make_tiny_mnist_label_file(test_labels, n=2)

    # Map URLs used in mnist_digits to our local files
    def _fake_fetch(url: str, data_dir=None, force: bool = False):
        if "train-images-idx3-ubyte" in url:
            return train_images
        if "train-labels-idx1-ubyte" in url:
            return train_labels
        if "t10k-images-idx3-ubyte" in url:
            return test_images
        if "t10k-labels-idx1-ubyte" in url:
            return test_labels
        raise ValueError(f"Unexpected URL: {url}")

    # Monkeypatch the fetch function inside the mnist_digits module
    monkeypatch.setattr(utils, "fetch", _fake_fetch)

    # MNIST original
    # Train
    Xtr, ytr, md_tr = mnist_digits.get_data_train_original()
    assert Xtr.shape == (3, 3, 3)
    assert ytr.shape == (3,)
    assert Xtr.dtype == np.uint8 and ytr.dtype == np.uint8
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3

    # Test
    Xte, yte, md_te = mnist_digits.get_data_test_original()
    assert Xte.shape == (2, 3, 3)
    assert yte.shape == (2,)
    assert Xte.dtype == np.uint8 and yte.dtype == np.uint8
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "test"
    assert md_te.num_instances == 2

    # Fashion MNIST
    # Train
    Xtr, ytr, md_tr = fashion_mnist.get_data_train()
    assert Xtr.shape == (3, 3, 3)
    assert ytr.shape == (3,)
    assert Xtr.dtype == np.uint8 and ytr.dtype == np.uint8
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3

    # Test
    Xte, yte, md_te = fashion_mnist.get_data_test()
    assert Xte.shape == (2, 3, 3)
    assert yte.shape == (2,)
    assert Xte.dtype == np.uint8 and yte.dtype == np.uint8
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "test"
    assert md_te.num_instances == 2

    # K-MNIST
    # Train
    Xtr, ytr, md_tr = k_mnist.get_data_train()
    assert Xtr.shape == (3, 3, 3)
    assert ytr.shape == (3,)
    assert Xtr.dtype == np.uint8 and ytr.dtype == np.uint8
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3

    # Test
    Xte, yte, md_te = k_mnist.get_data_test()
    assert Xte.shape == (2, 3, 3)
    assert yte.shape == (2,)
    assert Xte.dtype == np.uint8 and yte.dtype == np.uint8
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "test"
    assert md_te.num_instances == 2


def test_mnist_percevalquest_apis_offline(tmp_path: Path, monkeypatch):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _make_percevalquest_csv(train_csv, n=3)
    _make_percevalquest_csv(val_csv, n=2)

    def _fake_fetch(url: str, data_dir=None, force: bool = False):
        if url.endswith("/data/train.csv"):
            return train_csv
        if url.endswith("/data/val.csv"):
            return val_csv
        raise ValueError(f"Unexpected URL: {url}")

    monkeypatch.setattr(mnist_digits, "fetch", _fake_fetch)

    Xtr, ytr, md_tr = mnist_digits.get_data_train_percevalquest()
    assert Xtr.shape == (3, 28, 28)
    assert ytr.shape == (3,)
    assert Xtr.dtype in (np.float32, np.float64)
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3

    Xte, yte, md_te = mnist_digits.get_data_test_percevalquest()
    assert Xte.shape == (2, 28, 28)
    assert yte.shape == (2,)
    assert Xte.dtype in (np.float32, np.float64)
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "val"
    assert md_te.num_instances == 2
