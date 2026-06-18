merlin.core.partial_measurement
===============================

.. automodule:: merlin.core.partial_measurement
   :no-members:

.. currentmodule:: merlin.core.partial_measurement

PartialMeasurementBranch
------------------------

.. autoclass:: PartialMeasurementBranch
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:
   

PartialMeasurement
------------------

.. autoclass:: PartialMeasurement
   :members:
   :undoc-members:
   :member-order: bysource
   :show-inheritance:

.. autodata:: DetectorTransformOutput

Notes and Examples
------------------

Basic Structure
^^^^^^^^^^^^^^^

A :class:`PartialMeasurement` represents the outcome of measuring a subset of modes in a quantum system.
It consists of a collection of :class:`PartialMeasurementBranch` objects, each corresponding to a specific
measurement outcome on the measured modes, along with the conditional quantum state on the unmeasured modes.

.. code-block:: python

   from merlin.core.partial_measurement import PartialMeasurement, PartialMeasurementBranch
   from merlin.core.state_vector import StateVector
   import torch

   # Create branches manually
   outcome_1 = (1, 0)  # measurement result on measured modes 0 and 1
   prob_1 = torch.tensor([0.5])  # probability for this outcome
   amps_1 = StateVector.from_basic_state([1, 0], sparse=False)  # conditional state on unmeasured modes
   
   branch_1 = PartialMeasurementBranch(outcome_1, prob_1, amps_1)
   branch_2 = PartialMeasurementBranch((0, 1), torch.tensor([0.5]), StateVector.from_basic_state([0, 1], sparse=False))
   
   # Combine into PartialMeasurement
   pm = PartialMeasurement(
       branches=(branch_1, branch_2),
       measured_modes=(0, 1),
       unmeasured_modes=(2, 3)
   )

Accessing Measurement Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`PartialMeasurement` class provides convenient access to the measurement outcomes and their associated probabilities.

.. code-block:: python

   # Access properties
   print(f"Measured modes: {pm.measured_modes}")
   print(f"Unmeasured modes: {pm.unmeasured_modes}")
   print(f"Number of branches: {len(pm.branches)}")
   
   # Get probability distribution across all outcomes
   prob_tensor = pm.tensor  # shape: (batch_size, n_branches)
   
   # Get individual outcomes and amplitudes
   for outcome, branch in zip(pm.outcomes, pm.branches):
       print(f"Outcome {outcome}: probability={branch.probability}, amplitude shape={branch.amplitudes.shape}")

Working with Grouped Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`PartialMeasurement` supports optional grouping of probabilities, which allows you to aggregate outcomes
according to a custom grouping function (e.g., for classical post-processing or symmetry-based grouping).

.. code-block:: python

   from merlin.utils.grouping import ModGrouping
   
   # Define a grouping function
   grouping = ModGrouping(input_size = 4,output_size=2)  # example grouping
   
   # Apply grouping to PartialMeasurement
   pm.set_grouping(grouping)
   
   # Grouped probabilities now have shape (batch_size, output_size) instead of (batch_size, n_branches)
   grouped_probs = pm.probabilities
   print(grouped_probs.shape)  # (batch_size, 2)

Creating from DetectorTransform Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`PartialMeasurement` objects are typically created from the output of a detector transformation in the measurement pipeline.

.. code-block:: python

   from merlin.core.partial_measurement import PartialMeasurement
   
   # When DetectorTransform produces partial measurement output (partial_measurement=True),
   # it returns a structure that can be converted to PartialMeasurement
   detector_output = [...]  # output from DetectorTransform
   
   pm = PartialMeasurement.from_detector_transform_output(
       detector_output,
       grouping=None
   )

Batch Processing
^^^^^^^^^^^^^^^^

Probabilities are stored per-batch in each branch, allowing for efficient handling of batch-processed quantum circuits.

.. code-block:: python

   import torch
   
   # Batch-wise probabilities
   batch_probs = torch.tensor([[0.3, 0.7], [0.6, 0.4]])  # shape: (batch_size=2, n_outcomes=2)
   
   # When creating a branch with batched probabilities
   branch = PartialMeasurementBranch(
       outcome=(1, 0),
       probability=batch_probs[:, 0],  # shape: (batch_size,)
       amplitudes=StateVector.from_basic_state([1, 0])
   )
