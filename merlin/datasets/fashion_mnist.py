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

from .utils import get_data_generic

FASHION_MNIST_METADATA = {
    "name": "Fashion-MNIST",
    "description": "Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.",
    "features": [
        {
            "name": "pixel_values",
            "description": "Grayscale image of Zalando's article",
            "type": "uint8",
            "value_range": (0, 255),
            "unit": None,
        },
        {
            "name": "label",
            "description": "Article class label",
            "type": "uint8",
            "value_range": (0, 9),
            "unit": None,
        },
    ],
    "num_instances": 70000,  # 60000 training + 10000 test
    "task_type": ["classification"],
    "num_classes": 10,
    "characteristics": ["image"],
    "homepage": "https://github.com/zalandoresearch/fashion-mnist",
    "license": "MIT",
    "citation": """@online{fashionmnist2017,
        author       = {Zalando Research},
        title        = {Fashion MNIST},
        year         = {2017},
    }""",
    "creators": [
        "Zalando Research",
    ],
    "year": 2017,
}


def get_data_train():
    return get_data_generic(
        subset="train",
        url_images="https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/train-images-idx3-ubyte.gz",
        url_labels="https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/train-labels-idx1-ubyte.gz",
        metadata=FASHION_MNIST_METADATA,
    )


def get_data_test():
    return get_data_generic(
        subset="test",
        url_images="https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/t10k-images-idx3-ubyte.gz",
        url_labels="https://github.com/zalandoresearch/fashion-mnist/raw/refs/heads/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        metadata=FASHION_MNIST_METADATA,
    )


__all__ = [
    "get_data_train",
    "get_data_test",
]

# Example usage
if __name__ == "__main__":
    X, y, metadata = get_data_train()
    Xtest, ytest, _ = get_data_test()
    print(len(X), len(Xtest))
    # Mean of per-pixel standard deviations – Helps characterize/identify the dataset by capturing pixel-level variability.
    X_mean_std_per_pixel = np.std(X.reshape(X.shape[0], -1), axis=0).mean()
    Xtest_mean_std_per_pixel = np.std(Xtest.reshape(Xtest.shape[0], -1), axis=0).mean()
    print(X_mean_std_per_pixel, Xtest_mean_std_per_pixel)
    print(metadata)
