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

K_MNIST_METADATA = {
    "name": "Kuzushiji-MNIST",
    "description": "Kuzushiji-MNIST is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images). Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.",
    "features": [
        {
            "name": "pixel_values",
            "description": "Grayscale image of handwritten character",
            "type": "uint8",
            "value_range": (0, 255),
            "unit": None,
        },
        {
            "name": "label",
            "description": "Character class label",
            "type": "uint8",
            "value_range": (0, 9),
            "unit": None,
        },
    ],
    "num_instances": 70000,  # 60000 training + 10000 test
    "task_type": ["classification"],
    "num_classes": 10,
    "characteristics": ["image", "handwritten"],
    "homepage": "https://github.com/rois-codh/kmnist",
    "license": "Creative Commons Attribution-Share Alike 4.0",
    "citation": """@online{clanuwat2018deep,
        author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
        title        = {Deep Learning for Classical Japanese Literature},
        date         = {2018-12-03},
        year         = {2018},
        eprintclass  = {cs.CV},
        eprinttype   = {arXiv},
        eprint       = {cs.CV/1812.01718},
    }""",
    "creators": [
        "Tarin Clanuwat",
        "Mikel Bober-Irizar",
        "Asanobu Kitamoto",
        "Alex Lamb",
        "Kazuaki Yamamoto",
        "David Ha",
    ],
    "year": 2018,
}


def get_data_train():
    return get_data_generic(
        subset="train",
        url_images="https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
        url_labels="https://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
        metadata=K_MNIST_METADATA,
    )


def get_data_test():
    return get_data_generic(
        subset="test",
        url_images="https://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
        url_labels="https://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
        metadata=K_MNIST_METADATA,
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
