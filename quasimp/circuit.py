#!/usr/bin/env python3

import numpy as np
from typing import List, Union

from .gates import Gate
from .utils import probabilities_from_state


class Circuit:
    gates: List[Gate] = []
    qubit_num: int

    _state: np.ndarray = None
    _probabilities: np.ndarray = None

    def __init__(self, qubit_num: int) -> None:
        self.qubit_num = qubit_num

    def apply(self, gate: Gate) -> None:
        self._state, self._probabilities = None, None

        gate.set_qubit_num(self.qubit_num)
        self.gates.append(gate)

    @property
    def state(self) -> Union[np.ndarray, None]:
        return self._state

    def set_state(self, state: np.ndarray) -> None:
        self._state = state

    @property
    def probabilities(self) -> Union[np.ndarray, None]:
        if self._state is None:
            return None

        if self._probabilities is not None:
            return self._probabilities
        else:
            self._probabilities = probabilities_from_state(self.state)
            return self._probabilities
