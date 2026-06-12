merlin.builder.circuit_builder module
=====================================

.. automodule:: merlin.builder.circuit_builder
   :no-members:

.. currentmodule:: merlin.builder.circuit_builder

.. autoclass:: ModuleGroup
   :members:
   :undoc-members:

.. autoclass:: CircuitBuilder
   :members:
   :undoc-members:
   :show-inheritance:

--------------------------------------------------------------------------------
The different components of the builder with their arguments and example of code
--------------------------------------------------------------------------------

Below are the components available in the ``CircuitBuilder`` class.

**add_rotations**
-----------------------

Adds one or multiple phase shifters across the specified modes.

Arguments:

- ``modes`` (``int | list[int] | ModuleGroup | None``): Modes receiving the rotations. Defaults to all modes.
- ``axis`` (``str``): Axis of rotation. Default: ``"z"``.
- ``angle`` (``float``): Fixed rotation angle for non-trainable cases.
- ``trainable`` (``bool``): Promote the rotations to trainable parameters.
- ``name`` (``str``): Optional stem used for generated parameter names.
- ``role`` (:class:`~merlin.core.components.ParameterRole`): Explicitly set the parameter role.

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_rotations(modes=3, angle=np.pi / 4)

.. image:: ../../_static/img/builder_layer/rotation_comp.png
   :alt: A Rotation component built with CircuitBuilder
   :width: 200px
   :align: center

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_rotations(trainable=True, name="rotation")

.. image:: ../../_static/img/builder_layer/rotation_layer.png
   :alt: A Rotation layer built with CircuitBuilder
   :width: 200px
   :align: center

**add_angle_encoding**
----------------------------

Adds angle-based input encoding to the circuit.

Arguments:

- ``modes`` (``list[int]``): Modes to target for encoding.
- ``name`` (``str``): Prefix for generated input parameters.
- ``scale`` (``float``): Scaling factor for angle mapping.

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_angle_encoding(modes=list(range(6)), name="input", scale=np.pi)

This will show as a rotation layer as data is encoded in phase shifters.

**add_entangling_layer**
------------------------------

Adds an entangling layer spanning a range of modes using a generic interferometer.

Arguments:

- ``modes`` (``list[int]``): Modes to span.
- ``trainable`` (``bool``): Whether the entangling layer is trainable.
- ``model`` (``str``): Choose ``"mzi"`` (default) or ``"bell"`` to select the interferometer template.
- ``name`` (``str``): Optional prefix for parameter names.
- ``trainable_inner`` (``bool | None``): Override to control whether the internal phases of the MZIs remain trainable.
- ``trainable_outer`` (``bool | None``): Override to control the output phases at the end of the interferometer.

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_entangling_layer(trainable=True, name="U1")

.. image:: ../../_static/img/builder_layer/generic.png
   :alt: An entangling layer built with CircuitBuilder
   :width: 200px
   :align: center

To span it on different modes:

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_entangling_layer(trainable=True, name="U1", modes=[0, 3])

Switching to a Bell-style interferometer is as simple as:

.. code-block:: python

   builder.add_entangling_layer(model="bell", name="bell_block")

.. image:: ../../_static/img/builder_layer/generic_0_3.png
   :alt: An entangling layer built with CircuitBuilder
   :width: 200px
   :align: center


**add_superposition**
-----------------------

Adds one or more beam splitters with optional depth.

Arguments:

- ``targets`` (``tuple[int, int] | list[tuple[int, int]]``): Explicit mode pairs receiving beam splitters. When omitted, nearest neighbours across ``modes`` (or all modes) are used.
- ``depth`` (``int``): Number of successive passes to apply.
- ``theta`` (``float``): Mixing angle for fixed beam splitters.
- ``phi`` (``float``): Relative phase for fixed beam splitters.
- ``trainable`` (``bool``): Convenience flag marking both parameters trainable.
- ``trainable_theta`` (``bool``): Whether the mixing angle is trainable.
- ``trainable_phi`` (``bool``): Whether the relative phase is trainable.
- ``modes`` (``list[int]`` or ``ModuleGroup``): Mode span used when ``targets`` is omitted.
- ``name`` (``str``): Optional prefix for generated parameter names.

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs")

.. image:: ../../_static/img/builder_layer/supp_012.png
   :alt: A Superposition component (beam splitter)
   :width: 200px
   :align: center

.. code-block:: python

   builder = CircuitBuilder(n_modes=6)
   builder.add_superpositions(depth=2, name="mix")

.. image:: ../../_static/img/builder_layer/entangling_layer_depth2.png
   :alt: An entangling layer of depth 2
   :width: 200px
   :align: center


**add_memristive_ps**
--------------------------

Adds a memristive phase-shifter.

Arguments:

- ``mode`` (``int``): Circuit mode to target.
- ``update_rule`` (``callable``): Update rule to change the angle of the phase shifter after each forward pass of associated the :class:`~merlin.algorithms.layer.QuantumLayer` object. The function must take two
   positional arguments: update_rule(state,output). Here is the expected signature:
   .. code-block:: python

      def update_rule(state: torch.Tensor,output: torch.Tensor | StateVector | ProbabilityDistribution | PartialMeasurement)-> torch.Tensor

   The update rule must also handle batch inputs and return a tensor of size ``[batch_size]``, just like the state parameter. The output will be the same as the corresponding :class:`~merlin.algorithms.layer.QuantumLayer` forward.
- ``initial_state`` (``float``): The initial value of the phase shifter. This will be the value used after each :meth:`~merlin.algorithms.layer.QuantumLayer.reset` call.
- ``name`` (``str``): Prefix used for the generated memristive phase shifter parameter. Defaults to ``"mem"``.
- ``detach_at_each_forward`` (``bool``): Controls gradient flow through memristive state recurrence. When ``True`` (default), new states are detached after computation, blocking gradients through the state chain. When ``False``, gradients flow through the entire state history. Default value is ``True``.



Build
-----

Finalizes and returns the constructed circuit.
