:github_url: https://github.com/merlinquantum/merlin

=====================================================
Template for Reproducing a Quantum ML Paper in MerLin
=====================================================

.. admonition:: Paper Information
   :class: note

   **Title**: [Full Paper Title]

   **Authors**: [Author 1], [Author 2], [Author 3]

   **Published**: [Journal/Conference Name], [Volume/Issue], [Pages] ([Year])

   **DOI**: `[DOI URL] <[DOI URL]>`_

   .. Citation tracking: register the paper in _data/citations/papers.json
      (key, title, authors_short, year, venue, doi, doc), run
      docs/fetch_citations.py, then replace this comment with the badge
      directive (two dots, one space):  merlin-citations-badge:: [paper-key]

   **Reproduction Status**: [✅ Complete | 🚧 In Progress | ⚠️ Partial | 📋 Planned]

   **Reproducer**: [Reproducer Name] ([Reproducer Email])

Abstract
========

[1-2 paragraph summary of the paper's main contributions and methodology. Focus on what makes this work significant for the quantum ML field.]

Significance
============

[Why is this paper important? What gap does it fill? How does it advance the field?]

MerLin Implementation
=====================

[Brief overview of how the paper is implemented in MerLin. What specific MerLin components are used? Any special considerations or adaptations made?]

Key Contributions Reproduced
============================

**[Contribution Category 1]**
  * [Specific achievement 1]
  * [Specific achievement 2]
  * [Specific achievement 3]

**[Contribution Category 2]**
  * [Specific achievement 1]
  * [Specific achievement 2]
  * [Specific achievement 3]

**[Contribution Category 3 - if applicable]**
  * [Specific achievement 1]
  * [Specific achievement 2]

Implementation Details
======================

[Technical details about the implementation approach:]

.. code-block:: python

   import merlin as ml

  # Example code showing key implementation
  builder = ml.CircuitBuilder(n_modes=[number])
  builder.add_entangling_layer(name="phi_")
  builder.add_rotations(role="input", name="pl", axis="z")
  builder.add_entangling_layer(name="phi_")
     circuit = builder.to_pcvl_circuit()

   # Additional setup code
   model = ml.[ModelType](
       circuit=circuit,
       parameter1=value1,
       parameter2=value2
   )

Experimental Results
====================

**[Dataset/Experiment 1]**

.. list-table:: [Dataset Name] Performance Comparison
   :header-rows: 1
   :widths: 30 25 25 20

   * - Method
     - [Metric 1]
     - [Metric 2]
     - [Metric 3]
   * - Original Paper
     - [value]%
     - [value]
     - [value]
   * - MerLin Reproduction
     - [value]%
     - [value]
     - [value]
   * - [Baseline Method]
     - [value]%
     - [value]
     - [value]

**[Dataset/Experiment 2 - if applicable]**

[Description of additional experimental results, key findings, and analysis]

Technical Implementation Details
================================

**[Technical Aspect 1]**
  * [Implementation detail 1]
  * [Implementation detail 2]
  * [Implementation detail 3]

**[Technical Aspect 2]**
  * [Implementation detail 1]
  * [Implementation detail 2]
  * [Implementation detail 3]

**[Technical Aspect 3 - if applicable]**
  * [Implementation detail 1]
  * [Implementation detail 2]

Performance Analysis
====================

**Advantages of [Method/Approach]**
  * [Advantage 1]
  * [Advantage 2]
  * [Advantage 3]

**Current Limitations**
  * [Limitation 1]
  * [Limitation 2]
  * [Limitation 3]

**Scaling Behavior**
  * [Scaling observation 1]
  * [Scaling observation 2]
  * [Trade-offs and considerations]

Interactive Exploration
=======================

**Jupyter Notebook**: _[LINK TO NOTEBOOK]_

The notebook provides:

* [Interactive feature 1]
* [Interactive feature 2]
* [Interactive feature 3]
* [Interactive feature 4]
* [Interactive feature 5]

Extensions and Future Work
==========================

The MerLin implementation extends beyond the original paper:

**Enhanced Capabilities**
  * [Enhancement 1]
  * [Enhancement 2]
  * [Enhancement 3]

**Experimental Extensions**
  * [Extension 1]
  * [Extension 2]
  * [Extension 3]

**Hardware Considerations**
  * [Hardware consideration 1]
  * [Hardware consideration 2]
  * [Hardware consideration 3]

Code Access and Documentation
=============================

**GitHub Repository**: `merlin/reproductions/[folder_name] <https://github.com/merlinquantum/merlin/tree/main/reproductions/[folder_name]>`_

The complete implementation includes:

* [Code component 1]
* [Code component 2]
* [Code component 3]
* [Code component 4]
* [Code component 5]

Citation
========

.. code-block:: bibtex

   @article{[citationkey][year][firstauthor],
     title={[Full Paper Title]},
     author={[Author 1] and [Author 2] and [Author 3]},
     journal={[Journal Name]},
     volume={[Volume]},
     number={[Number]},
     pages={[Pages]},
     year={[Year]},
     publisher={[Publisher]},
     doi={[DOI]}
   }

Related Reproductions
=====================

This work complements other reproductions in the MerLin ecosystem:

* **[Related Paper 1]**: [Brief description of relationship]
* **[Related Paper 2]**: [Brief description of relationship]
* **[Future Work]**: [Description of planned future reproductions that build on this]

Impact and Applications
=======================

The [method/approach] demonstrated in this reproduction has implications for:

* **[Application Area 1]**: [Description of impact]
* **[Application Area 2]**: [Description of impact]
* **[Application Area 3]**: [Description of impact]
* **[Application Area 4]**: [Description of impact]

----

.. note::
   **Template Usage Instructions**

   1. **Replace all bracketed placeholders** with actual content
   2. **Remove sections** that don't apply to your specific paper
   3. **Add custom sections** as needed for your paper's unique contributions
   4. **Update the toctree** in the main reproduced_papers.rst file
   5. **Create corresponding Jupyter notebook** in the notebooks/ directory
   6. **Add entry to the main table** in reproduced_papers.rst

   **Optional Sections** (remove if not applicable):
   - Multiple experimental results tables
   - Extensions and future work
   - Performance analysis
   - Extended technical implementation details

   **Required Sections**:
   - Paper Information admonition
   - Abstract
   - Key Contributions Reproduced
   - Citation