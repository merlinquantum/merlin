merlin.core.components module
=============================

The `merlin.core.components` module defines the fundamental building blocks for photonic quantum circuits in Merlin. These components are platform-agnostic, descriptive objects that specify *what* should be present in a circuit (such as rotations, beam splitters, interferometers, and measurements), rather than *how* they are implemented on a specific backend.

**Why use components?**

- Components provide a clear, modular, and backend-independent way to describe quantum circuits.
- They enable high-level circuit construction, parameter management (fixed, trainable, or input-driven), and facilitate translation to various simulation or hardware platforms.
- By separating intent from implementation, components make it easy to prototype, optimize, and analyze quantum circuits in a flexible and extensible manner.

**Main components include:**

- **Rotation**: Describes a phase shifter (single-mode rotation), with configurable axis, value, and parameter role (fixed, trainable, or input).
- **BeamSplitter**: Represents a two-mode mixing operation, with tunable mixing angle (theta) and phase (phi), both of which can be fixed or trainable.
- **EntanglingBlock**: Specifies a block of entangling operations (e.g., nearest-neighbor beam splitters) to increase circuit expressivity.
- **GenericInterferometer**: Encodes a universal linear optical transformation over multiple modes, optionally trainable.

These components are used by higher-level tools (like `CircuitBuilder`) to assemble, manipulate, and compile quantum circuits for simulation or execution.

.. automodule:: merlin.core.components
   :members:
   :undoc-members:
   :show-inheritance:
