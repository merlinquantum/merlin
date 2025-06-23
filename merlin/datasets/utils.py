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

import gzip
import hashlib
import os
import site
import sys
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def get_venv_data_dir() -> Path:
    """
    Get the data directory within the current virtual environment.
    Creates a 'datasets' directory in the site-packages folder.
    """
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a venv/virtualenv
        # Get the site-packages directory of the current environment
        site_packages = site.getsitepackages()[0]
        data_dir = Path(site_packages) / "datasets"
    else:
        # Fallback to user's home directory if not in a venv
        data_dir = Path.home() / ".cache" / "datasets"

    return data_dir


def url_to_filename(url: str) -> str:
    """
    Convert URL to a filename, using hash to ensure uniqueness while keeping it readable.

    Args:
        url: URL to convert

    Returns:
        str: Filename based on the URL
    """
    # Get the original filename from the URL
    original_filename = os.path.basename(url)

    # Create a hash of the full URL
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]

    # If the file is gzipped, insert hash before .gz
    if original_filename.endswith('.gz'):
        base = original_filename[:-3]
        return f"{base}_{url_hash}.gz"
    else:
        # Insert hash before the extension (or at the end if no extension)
        root, ext = os.path.splitext(original_filename)
        return f"{root}_{url_hash}{ext}"


def fetch(url: str, data_dir: Path = None, force: bool = False) -> Path:
    """
    Fetch a file from URL, storing it in the virtual environment's data directory.
    If the file already exists, return its path unless force=True.
    If the file is gzipped, extract it.

    Args:
        url: URL to fetch the file from
        data_dir: Optional override for the data directory
        force: If True, re-download even if file exists

    Returns:
        Path: Path to the downloaded (and potentially extracted) file
    """
    if data_dir is None:
        data_dir = get_venv_data_dir()

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Generate filename from URL
    filename = url_to_filename(url)
    filepath = data_dir / filename

    # If file is gzipped, get the name without .gz
    if filename.endswith('.gz'):
        # Include the hash in the extracted filename
        extracted_filename = filename[:-3]  # Remove .gz but keep the hash
        extracted_path = data_dir / extracted_filename

        if not force and extracted_path.exists():
            return extracted_path

        # If forcing re-download, remove existing files
        if force:
            if extracted_path.exists():
                extracted_path.unlink()
            if filepath.exists():
                filepath.unlink()
    else:
        if not force and filepath.exists():
            return filepath

        # If forcing re-download, remove existing file
        if force and filepath.exists():
            filepath.unlink()

    # Download the file
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)

    # If file is gzipped, extract it and remove the compressed file
    if filename.endswith('.gz'):
        print(f"Extracting {filename}...")
        with gzip.open(filepath, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                f_out.write(f_in.read())
        filepath.unlink()  # Remove the gzipped file
        return extracted_path

    return filepath


def read_idx(filepath: Path) -> Tuple[np.ndarray, dict]:
    """
    Read an IDX file format as used in MNIST dataset.

    Args:
        filepath: Path to the IDX file

    Returns:
        Tuple[np.ndarray, dict]: Tuple containing:
            - numpy array with the data
            - metadata dictionary with magic number, data type, and dimensions
    """
    # Data type mapping from IDX format to numpy
    dtype_map = {
        0x08: np.uint8,
        0x09: np.int8,
        0x0B: np.dtype('>i2'),  # short, big-endian
        0x0C: np.dtype('>i4'),  # int, big-endian
        0x0D: np.dtype('>f4'),  # float, big-endian
        0x0E: np.dtype('>f8'),  # double, big-endian
    }

    with open(filepath, 'rb') as f:
        # Read magic number
        magic = int.from_bytes(f.read(4), byteorder='big')
        data_type = (magic >> 8) & 0xff  # Third byte
        n_dims = magic & 0xff  # Fourth byte

        # Verify magic number format
        if (magic >> 16) != 0:
            raise ValueError(f"Invalid magic number (first 2 bytes must be 0): {magic:08x}")

        if data_type not in dtype_map:
            raise ValueError(f"Unknown data type: {data_type:02x}")

        # Read dimensions
        dims = []
        for i in range(n_dims):
            dim_size = int.from_bytes(f.read(4), byteorder='big')
            dims.append(dim_size)

        # Read the data
        dtype = dtype_map[data_type]
        data = np.frombuffer(f.read(), dtype=dtype)

        # Reshape according to dimensions
        data = data.reshape(dims)

        # Create metadata dictionary
        metadata = {
            'magic': magic,
            'data_type': data_type,
            'dtype': dtype,
            'dims': dims
        }

        return data, metadata


def df_to_xy(df: pd.DataFrame,
             feature_cols: list = None,
             label_cols: list = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pandas DataFrame to numpy arrays for features (X) and labels (y)

    Args:
        df: Input DataFrame
        feature_cols: List of column names to use as features. If None, uses all columns except label_cols
        label_cols: List of column names to use as labels. If None, assumes last column is label

    Returns:
        X: numpy array of features
        y: numpy array of labels
    """
    if feature_cols is None and label_cols is None:
        # Assume last column is label
        feature_cols = df.columns[:-1]
        label_cols = [df.columns[-1]]
    elif feature_cols is None:
        # Use all columns except label columns as features
        feature_cols = [col for col in df.columns if col not in label_cols]
    elif label_cols is None:
        # Use all columns except feature columns as labels
        label_cols = [col for col in df.columns if col not in feature_cols]

    X = df[feature_cols].to_numpy()
    y = df[label_cols].to_numpy()

    # If single label column, flatten the array
    if len(label_cols) == 1:
        y = y.ravel()

    return X, y

if __name__ == "__main__":
    def _test_fetch():
        """Test the fetch function with MNIST dataset files"""
        mnist_urls = [
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
        ]

        print(f"Data directory: {get_venv_data_dir()}")

        for url in mnist_urls:
            print(f"\nTesting fetch for {os.path.basename(url)}")
            try:
                path = fetch(url)
                print(f"Success! File available at: {path}")
                print(f"File size: {path.stat().st_size:,} bytes")

                # Verify file exists and is readable
                assert path.exists(), "File does not exist"
                assert path.is_file(), "Not a regular file"
                assert os.access(path, os.R_OK), "File is not readable"

                # Basic size sanity check
                assert path.stat().st_size > 0, "File is empty"

            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                raise

    print("Running fetch tests...")
    _test_fetch()
    print("\nAll tests completed successfully!")