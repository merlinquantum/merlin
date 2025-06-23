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
from sklearn.utils import shuffle
from merlin.datasets import DatasetMetadata

def get_data(num_instances=1500, num_features=10, num_classes=3, random_seed=42):
    """
    Generate a spiral dataset inspired by the paper "Computational Advantage in Hybrid Quantum Neural Networks:
    Myth or Reality?" (https://arxiv.org/abs/2412.04991), generalized to support any number of spiral arms.

    This function creates a dataset with multiple interleaved spirals in a high-dimensional space.
    Each spiral represents a different class. The first two dimensions form the base spiral pattern,
    while additional dimensions are created through nonlinear combinations of these base features.
    The noise in the data scales with the distance from the origin, making points further from
    the center more difficult to classify.

    Args:
        num_instances (int, optional): Total number of samples to generate. Defaults to 1500.
        num_features (int, optional): Number of features for each sample. Defaults to 10.
        num_classes (int, optional): Number of spiral arms (classes). Defaults to 3.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X, y) where:
            - X (np.ndarray): Array of shape (n_samples, n_features) containing the feature data
            - y (np.ndarray): Array of shape (n_samples,) containing the class labels (0 to n_classes-1)

    Features:
        - First two dimensions: Form the base spiral pattern
        - Additional dimensions: Created through nonlinear combinations of the base features
        - Noise: Scales with distance from origin (harder to classify points further from center)
        - Multiple classes: Represented by interleaved spirals
    """
    np.random.seed(random_seed)

    # Ensure n_samples is divisible by n_classes
    samples_per_class = num_instances // num_classes
    num_instances = samples_per_class * num_classes  # Adjust total samples to be exact

    # Parameter space up to 3π - might need adjustment for very high number of classes
    t = np.linspace(0, 3 * np.pi / 2, samples_per_class)

    X = np.zeros((num_instances, num_features))
    y = np.zeros(num_instances)

    # Base noise level that increases with dimensionality
    base_noise = 0.1 + 0.003 * num_features

    for i in range(num_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class

        # Generate spiral pattern in first two dimensions
        r = 4 * t / 3 / np.pi  # Scale radius to get range approximately [-2,2]
        # Adjust angle spacing based on number of classes
        angle_offset = i * 2.0 * np.pi / num_classes
        X[start_idx:end_idx, 0] = r * np.cos(t + angle_offset)
        X[start_idx:end_idx, 1] = r * np.sin(t + angle_offset)

        # Calculate distance-dependent noise
        distances = np.sqrt(X[start_idx:end_idx, 0] ** 2 + X[start_idx:end_idx, 1] ** 2)
        point_noise = base_noise * distances[:, np.newaxis]

        # Generate additional features through nonlinear combinations
        if num_features > 2:
            for j in range(2, num_features):
                # Create interactions using sine and cosine functions
                # Use modulo with n_classes to create varying patterns
                base = np.sin(X[start_idx:end_idx, 0] * (j % num_classes + 1))
                phase = np.cos(X[start_idx:end_idx, 1] * ((j + 1) % num_classes + 1))
                X[start_idx:end_idx, j] = base * phase

        # Add distance-dependent noise to all features
        X[start_idx:end_idx] += np.random.normal(0, point_noise,
                                                 (samples_per_class, num_features))
        y[start_idx:end_idx] = i

    # Shuffle the data to avoid order class bias
    X, y = shuffle(X, y, random_state=random_seed)

    _metadata = {
        "name": "Quantum-Inspired Spiral Dataset",
        "description": "A synthetic dataset featuring high-dimensional spiral patterns inspired by quantum neural network research. The dataset consists of interleaved spirals where each spiral represents a different class, with noise that scales with distance from origin. The first two dimensions form the base spiral pattern, while additional dimensions are created through nonlinear combinations of these base features.",
        "feature_relationships": """Features organization:
    - Features 1-2: Base spiral pattern coordinates
    - Features 3+: Nonlinear combinations of base features using sine and cosine functions
    - Class labels: Integer values from 0 to (n_classes-1)""",
        "features": [
            {
                "name": "spiral_base_x",
                "description": "First coordinate of base spiral pattern",
                "type": "numeric",
                "value_range": (0,3),
                "unit": None
            },
            {
                "name": "spiral_base_y",
                "description": "Second coordinate of base spiral pattern",
                "type": "numeric",
                "value_range": (0,3),
                "unit": None
            },
            {
                "name": "nonlinear_features",
                "description": "Additional dimensions created through nonlinear combinations using sine and cosine functions",
                "type": "numeric",
                "value_range": None,
                "unit": None
            }
        ],
        "num_instances": num_instances,
        "num_features": num_features,
        "task_type": ["classification"],
        "num_classes": num_classes,
        "characteristics": ["multivariate", "synthetic", "non-linear"],
        "homepage": None,
        "license": None,
        "citation": "Cite: Computational Advantage in Hybrid Quantum Neural Networks: Myth or Reality? (arXiv:2412.04991)",
        "creators": ["Muhammad Kashif", "Alberto Marchisio", "Muhammad Shafique"],
        "year": 2024
    }

    return X, y, DatasetMetadata.from_dict(_metadata)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # Test with different numbers of classes
    n_classes_list = [3, 4, 5, 8]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    for idx, n_classes in enumerate(n_classes_list):
        # Generate dataset
        X, y, md = get_data(
                     num_instances=1500,
                     num_features=20,
                     num_classes=n_classes
        )
        print(md)

        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create color maps for train and validation sets
        train_colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        # Create slightly darker colors for validation set
        val_colors = plt.cm.rainbow(np.linspace(0, 1, n_classes)) * 0.7

        # Plot
        for i in range(n_classes):
            mask_train = y_train == i
            axes[idx].scatter(X_train[mask_train, 0], X_train[mask_train, 1],
                              c=[train_colors[i]], marker='o',
                              label=f'Train Class {i}', alpha=0.6)

            mask_val = y_val == i
            axes[idx].scatter(X_val[mask_val, 0], X_val[mask_val, 1],
                              c=[val_colors[i]], marker='x',
                              label=f'Val Class {i}', alpha=0.8)

        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].set_title(f'Spiral Dataset: {n_classes} Classes')
        # Only show legend for first plot to avoid clutter
        if idx == 0:
            axes[idx].legend()

    plt.tight_layout()
    plt.show()