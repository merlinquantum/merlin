:github_url: https://github.com/merlinquantum/merlin

MerLin - Photonic Quantum Machine Learning Framework
====================================================

MerLin brings quantum computing capabilities to AI practitioners through easy-to-use PyTorch integrations. Named after the legendary wizard, MerLin adds quantum wizardry to your AI toolkit with no quantum expertise required.

**Built for AI/ML practitioners**: MerLin is designed to feel familiar to PyTorch users while unlocking the potential of quantum computing. Under the hood, it leverages photonic quantum computing - a cutting-edge approach using single-photons that's hardware-aware and prepares your models for real quantum processors.

**Simulation-first with hardware bridges**: Optimized for classical simulation today, with connections to currently available photonic QPUs and pathways to next-generation quantum hardware.

**Key Goals:**

- **Paper Reproduction**: Simple tools to reproduce published quantum ML papers and benchmark algorithms - see our :doc:`reproduced papers </research/reproduced_papers>` list.
- **Quantum Architecture Bridge**: Access to latest and next-gen quantum photonic architectures as a bridge between AI and quantum worlds - see our :doc:`quantum architectures </research/architectures>`.
- **GPU-Optimized Performance**: Fast simulation scaling up to 500+ mode chips with 10-20 photons near the simulability threshold - see :doc:`performance benchmarks <./reference/performance>`.

Together, these provide researchers with comprehensive tools for exploring and developing new quantum-classical hybrid algorithms.

**Why Quantum Layers?** Enable non-conventional operations in hybrid workflows that can help classical ML models improve performance, learn faster, or use fewer parameters.

Advanced users can leverage the underlying `Perceval <https://perceval.quandela.net>`_ framework for custom models or advanced functionality.

.. toctree::
   :maxdepth: 3
   :hidden:

   Welcome <self>
   quickstart/index
   user_guide/index
   quantum_expert_area/index
   notebooks/index
   reproduced_papers/index
   api_reference/index
   performance/index

Who Should Use MerLin?
======================

- **AI/ML Practitioners**: Add quantum layers to existing PyTorch models
- **Quantum Researchers**: Experiment with photonic quantum computing  
- **Enterprise Teams**: Build future-proof quantum-AI applications

Installation
============

.. code-block:: bash

   pip install merlinquantum

For development:

.. code-block:: bash

   git clone https://github.com/merlinquantum/merlin.git
   cd merlin
   pip install -e ".[dev]"

Hello Quantum World!
====================

The following shows how to create a very simple quantum layer using MerLin's high-level API. This layer can be
integrated into any PyTorch model, and supports usual PyTorch operations like training and inference.

.. code-block:: python

   import merlin as ML # Package: merlinquantum, import: merlin
   import torch

   # Create a simple quantum layer
   quantum_layer = ML.QuantumLayer.simple(
       input_size=3,
   )

   # Use it like any PyTorch layer
   x = torch.rand(10, 3)
   output = quantum_layer(x)
   print(f"Input shape: {x.shape}, Output shape: {output.shape}")

Under the hood, this simple interface wraps complex photonic quantum operations — including architecture selection, ansatz design, input encoding, and photon number configuration. Learn more in our :doc:`User Guide </user_guide/index>`.

Learn More
==========

- **Examples**: Check the ``examples/`` directory for tutorials
- **Notebooks**: Explore ``docs/source/notebooks/`` for interactive examples

Roadmap
=======

- **v0.1**: Initial release with core features
- In development:

  - More circuit types and ansatz configurations
  - Improved documentation and examples
  - Integration with Quandela's photonic hardware
  - additional machine learning models
  -

- **v0.2**: current release with additional features such as
- new features :

  - feedforward
  - entangled boson sampling
  - quantum kernels
  - quantum bridge
  - remote processing

- **v0.3**
- In development:

  - Optimisation and consolidation of existing code
  - Bugfixes

Contributing
============

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: ``git checkout -b feature-name``
3. **Test** your changes: ``pytest tests/``
4. **Submit** a pull request

See our `Contributing Guide <https://github.com/merlinquantum/merlin/blob/main/CONTRIBUTING.md>`_ for detailed guidelines.

License
=======

MIT License - see `LICENSE <https://github.com/merlinquantum/merlin/blob/main/LICENSE>`_ for details.

Support
=======

- **Issues**: `GitHub Issues <https://github.com/merlinquantum/merlin/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/merlinquantum/merlin/discussions>`_

----

**⚡ Ready to add quantum power to your AI models? Get started with MerLin! ⚡**

