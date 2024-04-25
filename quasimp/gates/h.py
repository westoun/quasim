#!/usr/bin/env python3

import numpy as np

from .gate import Gate
from .utils import create_identity, create_matrix

BASE_MATRIX = 1.0 / (2.0**0.5) * np.array([[1, 1], [1, -1]], dtype=complex)


class H(Gate):
    qubit_num: int
    target_qubit: int

    def __init__(self, target_qubit: int) -> None:
        self.target_qubit = target_qubit

    def _create_matrix(self) -> np.ndarray:
        return create_matrix(
            BASE_MATRIX, target_qubit=self.target_qubit, qubit_num=self.qubit_num
        )

    def _create_repr(self) -> str:
        return f"H{self.qubit_num}(target_qubit={self.target_qubit})"
