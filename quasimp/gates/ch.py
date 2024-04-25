#!/usr/bin/env python3

import numpy as np

from .gate import Gate
from .utils import create_controlled_matrix

BASE_MATRIX = 1.0 / (2.0**0.5) * np.array([[1, 1], [1, -1]], dtype=np.complex128)


class CH(Gate):
    qubit_num: int
    control_qubit: int
    target_qubit: int

    def __init__(self, control_qubit: int, target_qubit: int) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def _create_matrix(self) -> np.ndarray:
        return create_controlled_matrix(
            BASE_MATRIX,
            control_qubit=self.control_qubit,
            target_qubit=self.target_qubit,
            qubit_num=self.qubit_num,
        )

    def _create_repr(self) -> str:
        return f"CH{self.qubit_num}(control_qubit={self.control_qubit},target_qubit={self.target_qubit})"
