from typing import Generator

def generate_all_fock_states(m, n) -> Generator:
    """Generates all possible Fock states for m modes and n photons."""
    if n == 0:
        yield (0,) * m
        return
    if m == 1:
        yield (n,)
        return

    for i in reversed(range(n + 1)):
        for state in generate_all_fock_states(m-1, n-i):
            yield (i,) + state