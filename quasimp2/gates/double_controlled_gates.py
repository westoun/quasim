#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate


class CCGate(IGate):
    target_qubit: int
    control_qubit1: int
    control_qubit2: int
    matrix: np.ndarray = None

    def __init__(
        self, control_qubit1: int, control_qubit2: int, target_qubit: int
    ) -> None:
        self.control_qubit1 = control_qubit1
        self.control_qubit2 = control_qubit2
        self.target_qubit = target_qubit

    @property
    def qubits(self) -> List[int]:
        return [self.target_qubit, self.control_qubit1, self.control_qubit2]
