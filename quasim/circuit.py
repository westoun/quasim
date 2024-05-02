#!/usr/bin/env python3

import numpy as np
from typing import List, Union, Dict

from .gates import IGate
from .utils import probabilities_from_state, probability_dict_from_state


class Circuit:
    gates: List[IGate]
    qubit_num: int

    _state: np.ndarray = None
    _probabilities: np.ndarray = None
    _probability_dict: Dict = None

    def __init__(self, qubit_num: int) -> None:
        self.gates = []

        self.qubit_num = qubit_num

    def apply(self, gate: IGate) -> None:
        self._state, self._probabilities = None, None

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

    @property
    def probability_dict(self) -> Union[Dict, None]:
        if self._state is None:
            return None

        if self._probability_dict is not None:
            return self._probability_dict
        else:
            self._probability_dict = probability_dict_from_state(self.state)
            return self._probability_dict

    def __repr__(self) -> str:
        return f"[{', '.join([str(gate) for gate in self.gates])}]"
