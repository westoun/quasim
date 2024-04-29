#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate

H_MATRIX = 1.0 / (2.0**0.5) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


class Gate(IGate):
    target_qubit: int
    matrix: np.ndarray = None

    def __init__(self, target_qubit: int) -> None:
        self.target_qubit = target_qubit

    @property
    def qubits(self) -> List[int]:
        return [self.target_qubit]


class H(Gate):
    matrix: np.ndarray = H_MATRIX
