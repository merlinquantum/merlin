import numpy as np
from perceval.backends import SLOSBackend
from perceval.utils import BasicState

# n = 4  # photons
# m = 38 # modes


def _fast_all_unnormalized_probampli(self, input_state: BasicState) -> np.ndarray:
    self.set_input_state(input_state)
    c = self._state_mapping[input_state].coefs.reshape(
        self._fsas[input_state.n].count()
    )
    return c


SLOSBackend.all_unnormalized_probampli = _fast_all_unnormalized_probampli
