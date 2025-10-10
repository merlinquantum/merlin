"""
Observable definitions for quantum measurements.
Backend-agnostic description of what to measure.
"""

from dataclasses import dataclass
from enum import Enum


class PauliBasis(Enum):
    """Pauli measurement bases."""

    X = "X"
    Y = "Y"
    Z = "Z"
    I = "I"  # noqa: E741


@dataclass
class PauliObservable:
    """
    Single Pauli string observable.

    For photonic (no_bunching=True): Z maps to (1-2n̂), giving ±1 eigenvalues
    For qubit: Standard Pauli operators
    """

    pauli_string: str  # e.g., "ZZI", "XYZ"
    coefficient: float = 1.0

    def __post_init__(self):
        """Validate that only Pauli symbols are used in the string representation."""
        if not all(c in "IXYZ" for c in self.pauli_string):
            raise ValueError(f"Invalid Pauli string: {self.pauli_string}")

    def __repr__(self):
        """Return a readable representation that keeps the coefficient explicit."""
        if self.coefficient == 1.0:
            return f"PauliObservable('{self.pauli_string}')"
        return f"PauliObservable({self.coefficient}*'{self.pauli_string}')"


@dataclass
class NumberOperator:
    """
    Number operator for photonic systems.

    Gives actual photon count expectation when no_bunching=False,
    or converts to Z measurement (±1) when no_bunching=True.
    """

    mode_index: int
    power: int = 1  # For ⟨n̂^k⟩
    operator_type: str = "number"
    coefficient: float = 1.0

    def __repr__(self):
        """Identify which mode is measured and, optionally, the exponent applied."""
        if self.power == 1:
            return f"NumberOperator(mode={self.mode_index})"
        return f"NumberOperator(mode={self.mode_index}, power={self.power})"


@dataclass
class CompositeObservable:
    """Sum of observables (e.g., for Hamiltonians)."""

    terms: list[PauliObservable | NumberOperator]

    def __iter__(self):
        """Yield individual observable terms when iterating over the composite."""
        return iter(self.terms)

    def __repr__(self):
        """Display the sum of terms using their symbolic form."""
        terms_str = " + ".join(
            f"{t.coefficient}*{getattr(t, 'pauli_string', f'n_{t.mode_index}')}"
            for t in self.terms
        )
        return f"CompositeObservable({terms_str})"


def parse_observable(
    expr: str, n_modes: int | None = None
) -> PauliObservable | CompositeObservable | NumberOperator:
    """Convert a string specification into the corresponding observable object.

    Args:
        expr: Expression such as ``"ZZI"`` or ``"0.5*ZZI + 0.5*IZZ"``.
        n_modes: Optional number of modes for validating Pauli string length.

    Returns:
        Union[PauliObservable, CompositeObservable, NumberOperator]: Parsed observable.
    """
    expr = expr.strip()

    # Check for number operator syntax
    if expr.startswith("n_"):
        try:
            mode_idx = int(expr.split("_")[1])
            return NumberOperator(mode_index=mode_idx)
        except (IndexError, ValueError):
            pass  # Fall through to regular parsing

    # Handle composite (sum of terms)
    if "+" in expr:
        terms = []
        for term in expr.split("+"):
            term = term.strip()
            obs = _parse_single_term(term, n_modes)
            terms.append(obs)
        return CompositeObservable(terms)

    return _parse_single_term(expr, n_modes)


def _parse_single_term(
    term: str, n_modes: int | None = None
) -> PauliObservable | NumberOperator:
    """Parse one summand appearing in a composite observable expression.

    Args:
        term: Sub-expression containing either a Pauli string or number operator.
        n_modes: Optional number of modes for validating Pauli string length.

    Returns:
        Union[PauliObservable, NumberOperator]: Parsed observable term.
    """
    term = term.strip()

    # Check for number operator in term
    if "n_" in term:
        parts = term.split("*")
        coefficient = 1.0
        if len(parts) == 2:
            coefficient = float(parts[0].strip())
            n_part = parts[1].strip()
        else:
            n_part = term

        if n_part.startswith("n_"):
            try:
                mode_idx = int(n_part.split("_")[1])
                return NumberOperator(mode_index=mode_idx, coefficient=coefficient)
            except (IndexError, ValueError):
                pass

    # Handle regular Pauli term
    coefficient = 1.0
    if "*" in term:
        parts = term.split("*")
        coefficient = float(parts[0].strip())
        pauli_str = parts[1].strip()
    else:
        pauli_str = term

    # Validate length if n_modes provided
    if n_modes is not None and len(pauli_str) != n_modes:
        raise ValueError(
            f"Pauli string '{pauli_str}' length {len(pauli_str)} "
            f"doesn't match n_modes={n_modes}"
        )

    return PauliObservable(pauli_str, coefficient)
