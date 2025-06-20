# MIT License
#
# Copyright (c) 2025 Quandela
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

import numpy as np
from .utils import fetch, read_idx
from pathlib import Path
import pandas as pd
from merlin.datasets import DatasetMetadata
import ast


MNIST_METADATA = {
    "name": "MNIST Database of Handwritten Digits",
    "description": "The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels).",
    "features": [
        {
            "name": "pixel_values",
            "description": "Grayscale image of handwritten digit",
            "type": "uint8",
            "value_range": (0, 255),
            "unit": None
        },
        {
            "name": "label",
            "description": "Digit class label",
            "type": "uint8",
            "value_range": (0, 9),
            "unit": None
        }
    ],
    "num_instances": 70000,  # 60000 training + 10000 test
    "task_type": ["classification"],
    "num_classes": 10,
    "characteristics": ["image", "handwritten"],
    "homepage": "http://yann.lecun.com/exdb/mnist/",
    "license": "Creative Commons Attribution-Share Alike 3.0",
    "citation": """@article{lecun2010mnist,
      title={MNIST handwritten digit database},
      author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
      journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
      volume={2},
      year={2010}
    }""",
    "creators": ["Yann LeCun", "Corinna Cortes", "CJ Burges"],
    "year": 1994
}

MNIST_METADATA_PERCEVALQUEST = {
    "name": "MNIST Subset for First Perceval Quest",
    "description": "A reduced subset of the MNIST database specifically curated for the First Perceval Quest, a hybrid quantum-classical machine learning challenge. The dataset consists of 6,000 training images and 600 validation images, carefully selected to explore quantum machine learning approaches using the Perceval framework for photonic quantum computing.",
    "features": [
        {
            "name": "pixel_values",
            "description": "Flattened grayscale image of handwritten digit",
            "type": "float",
            "value_range": (0, 255),
            "unit": None
        },
        {
            "name": "label",
            "description": "Digit class label",
            "type": "int",
            "value_range": (0, 9),
            "unit": None
        }
    ],
    "num_instances": 6600,  # 6000 training + 600 validation
    "task_type": ["classification"],
    "num_classes": 10,
    "characteristics": ["image", "handwritten", "subset"],
    "homepage": "https://github.com/Quandela/HybridAIQuantum-Challenge",
    "license": None,
    "citation": None,
    "creators": ["Quandela"],
    "year": 2024
}


def read_mnist_images(filepath: Path) -> np.ndarray:
    """
    Read MNIST images file and return a numpy array of shape (n_images, 28, 28).

    Args:
        filepath: Path to the MNIST images file

    Returns:
        np.ndarray: Array of images with shape (n_images, 28, 28)
    """
    data, metadata = read_idx(filepath)

    # Verify this is an image file (3 dimensions: n_images, height, width)
    if len(metadata['dims']) != 3:
        raise ValueError(f"Expected 3 dimensions for images, got {len(metadata['dims'])}")

    return data


def read_mnist_labels(filepath: Path) -> np.ndarray:
    """
    Read MNIST labels file and return a numpy array of labels.

    Args:
        filepath: Path to the MNIST labels file

    Returns:
        np.ndarray: Array of labels
    """
    data, metadata = read_idx(filepath)

    # Verify this is a labels file (1 dimension)
    if len(metadata['dims']) != 1:
        raise ValueError(f"Expected 1 dimension for labels, got {len(metadata['dims'])}")

    return data


def get_data_train_original():
    train_images_path = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz")
    train_labels_path = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz")
    X = read_mnist_images(train_images_path)
    y = read_mnist_labels(train_labels_path)
    MNIST_METADATA["num_instances"] = len(X)
    MNIST_METADATA["subset"] = "train"
    return X, y, DatasetMetadata.from_dict(MNIST_METADATA)


def get_data_test_original():
    train_images_path = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz")
    train_labels_path = fetch("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz")
    X = read_mnist_images(train_images_path)
    y = read_mnist_labels(train_labels_path)
    MNIST_METADATA["num_instances"] = len(X)
    MNIST_METADATA["subset"] = "test"
    return X, y, DatasetMetadata.from_dict(MNIST_METADATA)


def get_data_train_percevalquest():
    train = fetch("https://raw.githubusercontent.com/Quandela/HybridAIQuantum-Challenge/refs/heads/main/data/train.csv")
    #val = fetch("https://github.com/Quandela/HybridAIQuantum-Challenge/blob/main/data/val.csv")
    df_train = pd.read_csv(train)
    X = np.stack([
        np.array(ast.literal_eval(img), dtype=float).reshape(28, 28)
        for img in df_train['image']
    ])
    y = df_train['label'].to_numpy()
    MNIST_METADATA_PERCEVALQUEST["num_instances"] = len(X)
    MNIST_METADATA_PERCEVALQUEST["subset"] = "train"
    return X, y, DatasetMetadata.from_dict(MNIST_METADATA_PERCEVALQUEST)


def get_data_test_percevalquest():
    val = fetch("https://raw.githubusercontent.com/Quandela/HybridAIQuantum-Challenge/refs/heads/main/data/val.csv")
    df_val = pd.read_csv(val)
    X = np.stack([
        np.array(ast.literal_eval(img), dtype=float).reshape(28, 28)
        for img in df_val['image']
    ])
    y = df_val['label'].to_numpy()
    MNIST_METADATA_PERCEVALQUEST["num_instances"] = len(X)
    MNIST_METADATA_PERCEVALQUEST["subset"] = "val"
    return X, y, DatasetMetadata.from_dict(MNIST_METADATA_PERCEVALQUEST)


# Example usage
if __name__ == "__main__":
    X, y, metadata = get_data_train_percevalquest()
    Xtest, ytest, _ = get_data_test_percevalquest()
    print(len(X), len(Xtest))
    print(metadata)
