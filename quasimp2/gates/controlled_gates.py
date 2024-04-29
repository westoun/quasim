#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate

X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)


class CGate(IGate):
    target_qubit: int
    control_qubit: int
    matrix: np.ndarray = None

    def __init__(self, control_qubit: int, target_qubit: int) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    @property
    def qubits(self) -> List[int]:
        return [self.target_qubit, self.control_qubit]


class CX(CGate):
    matrix: np.ndarray = X_MATRIX
