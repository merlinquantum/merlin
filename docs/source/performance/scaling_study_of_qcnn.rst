====================================
Quantum CNN (QCNN) Scaling Study
====================================

This page details the empirical complexities and scalability constraints of 
the Quantum Convolutional Neural Network (QCNN) implementation in MerLin.

Complexity and Scaling
---------------------------

After conducting a rigorous scaling study of the QCNN architecture, we have 
mapped its computational complexity along two main axes:

* **Batch Size**: The computational complexity scales **linearly** with respect 
  to the ``batch_size``.
* **Input Dimensions**: The complexity scales **quadratically** with respect 
  to the input dimensions.

Execution Constraints and Safeguards
------------------------------------

To ensure execution stability and prevent runtime failures, specific limits 
must be observed during configuration:

* **Batch Size**: The ``batch_size`` can safely be scaled beyond **100**, as its 
  overall impact on performance overhead remains minor.
* **Input Dimension**: The ``input dimension`` is not capped, however, we recommand
to the user to avoid going above **24** if it is possible.

Here are the different graphs representing the scaling:

.. figure:: /_static/img/graph_scaling_study.png
   :align: center
   :width: 1000px
   :alt: QCNN scaling study graphs
   
   **Figure:** Graph scaling study.

Running the Benchmark Study
---------------------------

To reproduce the scaling study or evaluate performance updates under this 
subsystem, navigate to the ``merlin`` repository and execute the following 
benchmark pipeline:

.. code-block:: bash

    python -m benchmarks.QCNN_scaling_study_benchmark
    python -m benchmarks.scaling_study_graphs

----------------

For structural architecture details and QCNN layer signatures, see the internal expert documentation section: :mod:`merlin.models.qcnn`. See also: :class:`merlin.models.qcnn.QCNNClassifier`
