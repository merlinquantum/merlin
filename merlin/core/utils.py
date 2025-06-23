from typing import Generator
from itertools import combinations

def generate_all_fock_states(m, n, no_bunching = False) -> Generator:
    """Generates all possible Fock states for m modes and n photons."""
    if no_bunching:
        if n > m or n < 0:
            return
        for positions in combinations(range(m), n):
            fock_state = [0] * m

            for pos in positions:
                fock_state[pos] = 1
            yield tuple(fock_state)

    else:
        if n == 0:
            yield (0,) * m
            return
        if m == 1:
            yield (n,)
            return

        for i in reversed(range(n + 1)):
            for state in generate_all_fock_states(m-1, n-i):
                yield (i,) + state