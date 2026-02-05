from pathlib import Path
import hashlib
import json

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

def _hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def _hash_dict(d: dict) -> str:
    return _hash_string(json.dumps(d, sort_keys=True))

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
    md_str_hash = _hash_string(str(md_tr))
    md_dict_hash = _hash_dict(md_tr.to_dict())
    REF_STR_HASH = "b4e78db840b411cf28b4635f542de8d49b689c2650fed4c36fae9d30511ad475"
    REF_DICT_HASH = "6d5d1385ef88741d8ca8e8fe3a6e32c361c1cb6e87c7c25d8a9e56759a5e9036"
    assert Xtr.shape == (3, 3, 3)
    assert ytr.shape == (3,)
    assert Xtr.dtype == np.uint8 and ytr.dtype == np.uint8
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3
    assert md_str_hash == REF_STR_HASH, "MNIST / Train: Mismatch in str(METADATA): Expected:" + REF_STR_HASH + " calculated:" + md_str_hash
    assert md_dict_hash == REF_DICT_HASH, "MNIST / Train: Mismatch in dict(METADATA): Expected:" + REF_DICT_HASH + " calculated:" + md_dict_hash

    # Test
    Xte, yte, md_te = mnist_digits.get_data_test_original()
    md_str_hash = _hash_string(str(md_tr))
    md_dict_hash = _hash_dict(md_tr.to_dict())
    assert Xte.shape == (2, 3, 3)
    assert yte.shape == (2,)
    assert Xte.dtype == np.uint8 and yte.dtype == np.uint8
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "test"
    assert md_te.num_instances == 2
    assert md_str_hash == REF_STR_HASH, "MNIST / Test: Mismatch in str(METADATA): Expected:" + REF_STR_HASH + " calculated:" + md_str_hash
    assert md_dict_hash == REF_DICT_HASH, "MNIST / Test: Mismatch in dict(METADATA): Expected:" + REF_DICT_HASH + " calculated:" + md_dict_hash

    # Fashion MNIST
    # Train
    Xtr, ytr, md_tr = fashion_mnist.get_data_train()
    md_str_hash = _hash_string(str(md_tr))
    md_dict_hash = _hash_dict(md_tr.to_dict())
    REF_STR_HASH = "1f53f37e14a7b8278637f4ab5dca4e922fdda4cb64cf14fcfe47bf7a717aeb9e"
    REF_DICT_HASH = "61ef906cbde33f50c9c9b44b5e1849e4cc5f8d1fee474b55e3a15d3b8895c895"
    assert Xtr.shape == (3, 3, 3)
    assert ytr.shape == (3,)
    assert Xtr.dtype == np.uint8 and ytr.dtype == np.uint8
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3
    assert md_str_hash == REF_STR_HASH, "Fashion-MNIST / Train: Mismatch in str(METADATA): Expected:" + REF_STR_HASH + " calculated:" + md_str_hash
    assert md_dict_hash == REF_DICT_HASH, "Fashion-FMNIST / Train: Mismatch in dict(METADATA): Expected:" + REF_DICT_HASH + " calculated:" + md_dict_hash

    # Test
    Xte, yte, md_te = fashion_mnist.get_data_test()
    md_str_hash = _hash_string(str(md_tr))
    md_dict_hash = _hash_dict(md_tr.to_dict())
    assert Xte.shape == (2, 3, 3)
    assert yte.shape == (2,)
    assert Xte.dtype == np.uint8 and yte.dtype == np.uint8
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "test"
    assert md_te.num_instances == 2
    assert md_str_hash == REF_STR_HASH, "Fashion-MNIST / Test: Mismatch in str(METADATA): Expected:" + REF_STR_HASH + " calculated:" + md_str_hash
    assert md_dict_hash == REF_DICT_HASH, "Fashion-MNIST / Test: Mismatch in dict(METADATA): Expected:" + REF_DICT_HASH + " calculated:" + md_dict_hash

    # K-MNIST
    # Train
    Xtr, ytr, md_tr = k_mnist.get_data_train()
    md_str_hash = _hash_string(str(md_tr))
    md_dict_hash = _hash_dict(md_tr.to_dict())
    REF_STR_HASH = "8c01b556077483c07904ea62765c65893236ef72b01583881a830ee5e568c5f9"
    REF_DICT_HASH = "3ea97b89f76c68c4be7713b7be6793253ce7cf5a4a0543adcac09d35cebb20d9"
    assert Xtr.shape == (3, 3, 3)
    assert ytr.shape == (3,)
    assert Xtr.dtype == np.uint8 and ytr.dtype == np.uint8
    assert isinstance(md_tr, DatasetMetadata)
    assert md_tr.subset == "train"
    assert md_tr.num_instances == 3
    assert md_str_hash == REF_STR_HASH, "K-MNIST / Train: Mismatch in str(METADATA): Expected:" + REF_STR_HASH + " calculated:" + md_str_hash
    assert md_dict_hash == REF_DICT_HASH, "K-MNIST / Train: Mismatch in dict(METADATA): Expected:" + REF_DICT_HASH + " calculated:" + md_dict_hash

    # Test
    Xte, yte, md_te = k_mnist.get_data_test()
    md_str_hash = _hash_string(str(md_tr))
    md_dict_hash = _hash_dict(md_tr.to_dict())
    assert Xte.shape == (2, 3, 3)
    assert yte.shape == (2,)
    assert Xte.dtype == np.uint8 and yte.dtype == np.uint8
    assert isinstance(md_te, DatasetMetadata)
    assert md_te.subset == "test"
    assert md_te.num_instances == 2
    assert md_str_hash == REF_STR_HASH, "K-MNIST / Test: Mismatch in str(METADATA): Expected:" + REF_STR_HASH + " calculated:" + md_str_hash
    assert md_dict_hash == REF_DICT_HASH, "K-MNIST / Test: Mismatch in dict(METADATA): Expected:" + REF_DICT_HASH + " calculated:" + md_dict_hash


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
