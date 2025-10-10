"""
Circuit is a simple container of components with metadata.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Circuit:
    """Simple circuit container."""

    n_modes: int
    components: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, component: Any) -> "Circuit":
        """Append a component and return the circuit for chained calls.

        Args:
            component: Circuit element (rotation, beam splitter, measurement, etc.).

        Returns:
            Circuit: ``self`` to support fluent-style chaining.
        """
        self.components.append(component)
        return self

    def clear(self):
        """Remove every component and metadata entry from the circuit."""
        self.components.clear()
        self.metadata.clear()

    @property
    def num_components(self) -> int:
        """Return the count of registered components."""
        return len(self.components)

    @property
    def depth(self) -> int:
        """Estimate logical depth by summing component depths when available."""
        depth = 0
        for comp in self.components:
            if hasattr(comp, "depth"):
                depth += comp.depth
            else:
                depth += 1
        return depth

    def get_parameters(self) -> dict[str, Any]:
        """Collect parameter placeholders exposed by each component.

        Returns:
            Dict[str, Any]: Mapping from parameter name to default value (or ``None``).
        """
        params = {}
        for comp in self.components:
            if hasattr(comp, "get_params"):
                params.update(comp.get_params())
        return params

    def __repr__(self) -> str:
        return f"Circuit(n_modes={self.n_modes}, components={self.num_components})"
