"""
Circuit is a simple container of components with metadata.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Circuit:
    """Simple circuit container."""
    n_modes: int
    components: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, component: Any) -> "Circuit":
        """Add a component."""
        self.components.append(component)
        return self

    def clear(self):
        """Clear all components."""
        self.components.clear()
        self.metadata.clear()

    @property
    def num_components(self) -> int:
        """Number of components in circuit."""
        return len(self.components)

    @property
    def depth(self) -> int:
        """Estimate circuit depth."""
        depth = 0
        for comp in self.components:
            if hasattr(comp, 'depth'):
                depth += comp.depth
            else:
                depth += 1
        return depth

    def get_parameters(self) -> Dict[str, Any]:
        """Extract all parameters from circuit."""
        params = {}
        for comp in self.components:
            if hasattr(comp, 'get_params'):
                params.update(comp.get_params())
        return params

    def __repr__(self) -> str:
        return f"Circuit(n_modes={self.n_modes}, components={self.num_components})"