# MerLin - Photonic Quantum Machine Learning Framework

![Tests](https://img.shields.io/github/actions/workflow/status/merlinquantum/merlin/ci.yml?branch=main&style=flat-square&logo=github&label=tests)
![Coverage](https://img.shields.io/github/actions/workflow/status/merlinquantum/merlin/coverage.yml?branch=main&style=flat-square&logo=github&label=coverage)

MerLin brings quantum computing capabilities to AI practitioners through easy-to-use PyTorch integrations. Named after the legendary wizard, MerLin adds quantum wizardry to your AI toolkit with no quantum expertise required.

**Built for AI/ML practitioners**: MerLin is designed to feel familiar to PyTorch users while unlocking the potential of quantum computing. Under the hood, it leverages photonic quantum computing - a cutting-edge approach using single-photons that's hardware-aware and prepares your models for real quantum processors.

**Simulation-first with hardware bridges**: Optimized for classical simulation today, with connections to currently available photonic QPUs and pathways to next-generation quantum hardware.

**Key Goals:**

- **Paper Reproduction**: Simple tools to reproduce published quantum ML papers and benchmark algorithms - see our [reproduced papers](https://merlinquantum.ai/research/reproduced_papers.html) list.
- **Quantum Architecture Bridge**: Access to latest and next-gen quantum photonic architectures as a bridge between AI and quantum worlds - see our [quantum architectures](https://merlinquantum.ai/research/architectures.html).
- **GPU-Optimized Performance**: Fast simulation scaling up to 500+ mode chips with 10-20 photons near the simulability threshold - see [performance benchmarks](https://merlinquantum.ai/reference/performance.html).

Together, these provide researchers with comprehensive tools for exploring and developing new quantum-classical hybrid algorithms.

**Why Quantum Layers?** Enable non-conventional operations in hybrid workflows that can help classical ML models improve performance, learn faster, or use fewer parameters.

Advanced users can leverage the underlying [Perceval](https://perceval.quandela.net) framework for custom models or advanced functionality.

## Who Should Use MerLin?

- **AI/ML Practitioners**: Add quantum layers to existing PyTorch models
- **Quantum Researchers**: Experiment with photonic quantum computing  
- **Enterprise Teams**: Build future-proof quantum-AI applications

## Installation

Production installation:
```bash
pip install merlinquantum
```

Development (includes tests, benchmarks, lint & mypy):
```bash
git clone https://github.com/merlinquantum/merlin.git
cd merlin
pip install -e '.[dev]'
```

Examples environment (notebooks & plots):
```bash
pip install -e '.[examples]'
```

Build documentation locally:
```bash
pip install -e '.[docs]'
cd docs
make html  # or: make livehtml (if sphinx-autobuild added manually)
```

### Tests & Benchmarks

Run the full test suite:
```bash
pytest -q
```

Run only benchmarks (pytest-benchmark):
```bash
pytest --benchmark-only
```

Compare two branches (example):
```bash
pytest tests/test_sampling.py --benchmark-save current
# ... switch branch ...
pytest tests/test_sampling.py --benchmark-compare current
```

Quick quality checks:
```bash
ruff check .
ruff format --check .
mypy merlin
```

Tip: run `pytest -k <keyword>` to target a subset.

## Hello Quantum World!

The following shows how to create a very simple quantum layer using MerLin's high-level API. This layer can be
integrated into any PyTorch model, and supports usual PyTorch operations like training and inference.

``` python
   import merlin as ML # Package: merlinquantum, import: merlin
   import torch
   
   # Create a simple quantum layer
   quantum_layer = ML.QuantumLayer.simple(
       input_size=3,
       n_params=50  # Number of trainable quantum parameters
   )

   # Use it like any PyTorch layer
   x = torch.rand(10, 3)
   output = quantum_layer(x)
   print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

Under the hood, this simple interface wraps complex photonic quantum operations — including architecture selection, ansatz design, input encoding, and photon number configuration. Learn more in our [User Guide](https://merlinquantum.ai/user_guide/index.html).

## Learn More

- **Examples**: Check the ``examples/`` directory for tutorials
- **Notebooks**: Explore ``docs/source/notebooks/`` for interactive examples

## Roadmap

- **v0.1**: Initial release with core features
- In development:

  - More circuit types and ansatz configurations
  - Improved documentation and examples
  - Integration with Quandela's photonic hardware
  - additional machine learning models

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: ``git checkout -b feature-name``
3. **Test** your changes: ``pytest tests/``
4. **Submit** a pull request

See our [Contributing Guide](https://github.com/merlinquantum/merlin/blob/main/CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](https://github.com/merlinquantum/merlin/blob/main/LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/merlinquantum/merlin/issues)
- **Discussions**: [GitHub Discussions](https://github.com/merlinquantum/merlin/discussions)

## Test Coverage

MerLin uses automated test coverage tracking to maintain code quality:

**Coverage Reports:**
- 🎯 **Target Coverage:** 80% (warning threshold)
- 📊 **Reports Generated:** On every PR and commit
- 🚫 **Non-blocking:** Coverage checks don't prevent merges
- 📈 **Diff Coverage:** Shows coverage for changed files only

**Running Coverage Locally:**
```bash
# Quick coverage check
pytest tests/ --cov=merlin --cov-report=term | grep TOTAL

# Detailed coverage with missing lines  
pytest tests/ --cov=merlin --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=merlin --cov-report=html
# Then open htmlcov/index.html in browser

# Test specific module
pytest tests/test_layer.py --cov=merlin.core --cov-report=term
```

**Coverage Configuration:**
- Exclusions: Tests, migrations, virtual environments
- Formats: Terminal, HTML, XML reports
- Thresholds: 80% target (informational only)

Coverage data is automatically collected and reported in PRs without blocking development workflow.

----

**⚡ Ready to add quantum power to your AI models? Get started with MerLin! ⚡**
