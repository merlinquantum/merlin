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

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from merlin.datasets import DatasetMetadata

iris_data = load_iris()
features = iris_data.data
labels = iris_data.target

# normalize the features
features = MinMaxScaler().fit_transform(features)

# split train, test
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=123
)

IRIS_METADATA = {
    "name": "Iris Plants Dataset",
    "description": "The famous Iris database, first used by Sir R.A. Fisher in 1936. Contains measurements of three different Iris flower species. One class is linearly separable from the other two, but the latter are not linearly separable from each other.",
    "normalization": {
        "method": "min-max",
        "range": (0, 1),
        "per_feature": True
    },
    "features": [
        {
            "name": "sepal_length",
            "description": "Length of sepal (normalized)",
            "type": "float",
            "value_range": (0, 1),
            "unit": "normalized",
            "normalization": {
                "original_unit": "cm",
                "scale_factor": 3.6,  # max-min = 7.9-4.3
                "offset": 4.3         # original min
            }
        },
        {
            "name": "sepal_width",
            "description": "Width of sepal (normalized)",
            "type": "float",
            "value_range": (0, 1),
            "unit": "normalized",
            "normalization": {
                "original_unit": "cm",
                "scale_factor": 2.4,  # max-min = 4.4-2.0
                "offset": 2.0         # original min
            }
        },
        {
            "name": "petal_length",
            "description": "Length of petal (normalized)",
            "type": "float",
            "value_range": (0, 1),
            "unit": "normalized",
            "normalization": {
                "original_unit": "cm",
                "scale_factor": 5.9,  # max-min = 6.9-1.0
                "offset": 1.0         # original min
            }
        },
        {
            "name": "petal_width",
            "description": "Width of petal (normalized)",
            "type": "float",
            "value_range": (0, 1),
            "unit": "normalized",
            "normalization": {
                "original_unit": "cm",
                "scale_factor": 2.4,  # max-min = 2.5-0.1
                "offset": 0.1         # original min
            }
        }
    ],
    "num_instances": 150,
    "task_type": ["classification"],
    "num_classes": 3,
    "characteristics": ["multivariate", "real"],
    "homepage": None,
    "license": None,
    "citation": """Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936)""",
    "creators": ["R.A. Fisher"],
    "year": 1936
}

def get_data_train():
    IRIS_METADATA["subset"] = "train"
    return train_features, train_labels, DatasetMetadata.from_dict(IRIS_METADATA)

def get_data_test():
    IRIS_METADATA["subset"] = "test"
    return test_features, test_labels, DatasetMetadata.from_dict(IRIS_METADATA)
