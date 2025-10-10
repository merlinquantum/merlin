# MerLin Continuous Benchmarking Guide

This document provides a comprehensive guide for using and extending the MerLin continuous benchmarking system.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Creating New Benchmarks](#creating-new-benchmarks)
- [Updating GitHub Actions](#updating-github-actions)
- [Viewing Results](#viewing-results)
- [Best Practices](#best-practices)
- [Troubleshooting](...)

## 🎯 Overview

The MerLin continuous benchmarking system automatically tracks performance across the codebase, detecting regressions and monitoring improvements. It provides:

- **Automated Performance Testing**: Runs on every PR and main branch push
- **Regression Detection**: Alerts when performance degrades >20%
- **Performance Visualization**: Interactive charts on GitHub Pages
- **PR Integration**: Automatic comments with benchmark results
- **Baseline Protection**: Only main branch merges update the performance baseline

## 🏗️ System Architecture

```
benchmarks/                    # Benchmark files directory
├── benchmark_slos_core.py     # SLOS core function benchmarks
├── benchmark_no_bunching.py   # No bunching algorithm benchmarks  
├── benchmark_layer.py         # Quantum layer benchmarks
├── benchmark_robustness.py    # Robustness and stress test benchmarks
└── benchmark_unitary_conversion.py  # Unitary conversion (Perceval→Torch) benchmarks

.github/workflows/
└── benchmark.yml              # GitHub Actions workflow

GitHub Pages: https://merlinquantum.github.io/merlin/dev/bench/
```

## ⚡ How It Works

### For Pull Requests:
1. **Triggers**: Workflow runs when PR is created/updated
2. **Benchmarks Execute**: All benchmark files are run on github runners
3. **Comparison**: Results compared against main branch baseline
4. **Alerts**: PR gets automated comment if performance regression >20%
5. **Blocking**: Workflow fails if regression detected (configurable)
6. **No Baseline Update**: GitHub Pages remain unchanged

### For Main Branch (Merges):
1. **Triggers**: Workflow runs on push to main
2. **Benchmarks Execute**: Same benchmark execution as PRs
3. **Baseline Update**: Results become new performance baseline
4. **GitHub Pages Update**: Charts updated with new data points
5. **Navigation**: Dynamic index page regenerated

## 🔧 Creating New Benchmarks

### Step 1: Create Benchmark File

Create a new file in `/benchmarks/` following the naming pattern `benchmark_<category>.py`:

```python
# benchmarks/benchmark_your_feature.py

import pytest
import torch
import time
import json
import os
from typing import List, Tuple, Dict, Any

# Your imports here
import merlin as ML

class YourFeatureBenchmarkRunner:
    """Utility class for running and validating your feature benchmarks."""
    
    def __init__(self):
        self.results = []
    
    def validate_output_correctness(self, output, expected_shape) -> bool:
        """Validate that the output is correct."""
        # Add your validation logic
        return True

# Test configurations for different complexity levels
BENCHMARK_CONFIGS = [
    {"param1": "small", "param2": 10, "name": "small"},
    {"param1": "medium", "param2": 50, "name": "medium"},
    {"param1": "large", "param2": 100, "name": "large"},
]

DEVICE_CONFIGS = ["cpu"]

benchmark_runner = YourFeatureBenchmarkRunner()

@pytest.mark.parametrize("config", BENCHMARK_CONFIGS)
@pytest.mark.parametrize("device", DEVICE_CONFIGS)
def test_your_feature_benchmark(benchmark, config: Dict, device: str):
    """Benchmark your feature functionality."""
    
    # Setup test data
    test_data = create_test_data(config)
    
    def run_your_function():
        return your_function(test_data)
    
    # Run benchmark
    result = benchmark(run_your_function)
    
    # Validate correctness
    assert benchmark_runner.validate_output_correctness(result, expected_shape)

# Performance regression tests
class TestYourFeaturePerformanceRegression:
    """Test suite for detecting performance regressions."""
    
    def test_performance_bounds(self):
        """Test that your feature stays within reasonable time bounds."""
        # Add performance bounds testing
        pass

# Utility function to save benchmark results
def save_benchmark_results(results: List[Dict[str, Any]], output_path: str = "your-feature-results.json"):
    """Save benchmark results in the format expected by github-action-benchmark."""
    formatted_results = []
    
    for result in results:
        formatted_results.append({
            "name": result.get("name", "unknown"),
            "unit": "seconds", 
            "value": result.get("mean", 0),
            "extra": {
                "min": result.get("min", 0),
                "max": result.get("max", 0),
                "stddev": result.get("stddev", 0),
                "iterations": result.get("rounds", 1)
            }
        })
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    print("Running your feature benchmarks...")
    # Add standalone test execution
    print("Your feature benchmark tests completed!")
```

### Step 2: Key Requirements for Benchmark Files

#### ✅ **Required Elements:**
- **Parametrized tests**: Use `@pytest.mark.parametrize` for different configurations
- **Benchmark runner class**: For validation and utility functions
- **Performance bounds**: Add regression tests with time limits
- **Validation**: Ensure benchmark results are correct
- **Standalone execution**: `if __name__ == "__main__"` section for testing

#### ✅ **Best Practices:**
- **Multiple complexity levels**: Test small, medium, large configurations
- **Realistic workloads**: Use representative data sizes and patterns
- **Batched operations**: Test with batched inputs when applicable
- **Error handling**: Validate inputs and outputs thoroughly
- **Documentation**: Clear docstrings explaining what each benchmark tests

## 🚀 Updating GitHub Actions

### Step 1: Add New Benchmark Job

Add a new job to `.github/workflows/benchmark.yml`:

```yaml
  benchmark-your-feature:
    runs-on: ubuntu-latest
    needs: setup
    steps:
    - uses: actions/checkout@v4


    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install pytest pytest-benchmark numpy pandas psutil
        pip install perceval-quandela==0.13.1
        pip install -e .

    - name: Set performance environment variables
      run: |
        echo "OMP_NUM_THREADS=1" >> $GITHUB_ENV
        echo "MKL_NUM_THREADS=1" >> $GITHUB_ENV
        echo "NUMEXPR_NUM_THREADS=1" >> $GITHUB_ENV
        echo "OPENBLAS_NUM_THREADS=1" >> $GITHUB_ENV

    - name: Run your feature benchmarks
      run: |
        pytest benchmarks/benchmark_your_feature.py --benchmark-json=your-feature-results.json -v --benchmark-only

    - name: Store your feature benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: your-feature-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
        benchmark-data-dir-path: 'dev/bench/your-feature'
        fail-on-alert: true
        alert-threshold: '120%'
        comment-on-alert: true
        summary-always: true
```

### Step 2: Update Job Dependencies

Update the `create-index` job to include your new benchmark:

```yaml
  create-index:
    runs-on: ubuntu-latest
    needs: [benchmark-slos-core, benchmark-no-bunching, benchmark-layer, benchmark-robustness, benchmark-your-feature]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

### Step 3: Configuration Options

#### Alert Settings:
- `alert-threshold: '120%'` - Triggers alert at 20% performance regression
- `fail-on-alert: true` - Fails workflow on performance regression
- `comment-on-alert: true` - Posts comment on PR with regression details

#### Auto-push Logic:
```yaml
auto-push: ${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}
```
This ensures GitHub Pages only updates on main branch merges, not PRs.

## 📊 Viewing Results

### GitHub Pages Location:
- **Main Dashboard**: `https://merlinquantum.github.io/merlin/`
- **All Benchmarks Index**: `https://merlinquantum.github.io/merlin/dev/bench/`

### Automatic Navigation:
The system automatically generates navigation pages by scanning existing benchmark directories. New benchmarks appear automatically without manual configuration.

### Performance Charts:
Each benchmark category gets its own interactive chart showing:
- **Performance over time**: Track improvements and regressions
- **Multiple metrics**: Mean, min, max, standard deviation
- **Commit correlation**: Link performance changes to specific commits
- **Trend analysis**: Visual indicators of performance direction

## ✅ Best Practices

### Benchmark Design:
1. **Representative Workloads**: Use realistic data sizes and patterns
2. **Multiple Configurations**: Test across different complexity levels
3. **Batched Testing**: Include batch sizes relevant to real usage
4. **Validation**: Always verify benchmark results are correct
5. **Stability**: Ensure benchmarks are deterministic and repeatable

### Performance Thresholds:
1. **Conservative Alerts**: 20% regression threshold catches significant issues
2. **Contextual Bounds**: Set different thresholds for different benchmark types
3. **Baseline Quality**: Only merge performance improvements to maintain clean baselines

### Code Organization:
1. **Separate Directory**: Keep benchmarks in `/benchmarks/` not `/tests/`
2. **Clear Naming**: Use descriptive file and function names
3. **Modular Design**: Create reusable benchmark utilities
4. **Documentation**: Document what each benchmark measures

### CI/CD Integration:
1. **Parallel Execution**: Benchmarks run in parallel for speed
2. **Consistent Environment**: GitHub Actions runners provide stable, reproducible environments
3. **Conditional Updates**: Only update baselines from main branch
4. **Failure Handling**: Configure appropriate failure behaviors

## 🔧 Troubleshooting
### Debug Commands:

```bash
# Test benchmark locally
python benchmarks/benchmark_your_feature.py

# Run specific benchmark with pytest
python -m pytest benchmarks/benchmark_your_feature.py -v --benchmark-only

# Install benchmark dependencies
pip install pytest pytest-benchmark numpy pandas psutil perceval-quandela

# Validate benchmark results format
python -c "import json; print(json.load(open('results.json')))"
```