#!/usr/bin/env python3

import numpy as np
from typing import List

from .gate import Gate
from .utils import create_double_controlled_matrix

BASE_MATRIX = np.array([[1, 0], [0, -1]], dtype=np.complex128)


class CCZ(Gate):
    qubit_num: int
    control_qubits: List[int]
    target_qubit: int

    def __init__(self, control_qubits: List[int], target_qubit: int) -> None:
        control_qubits.sort()
        self.control_qubits = control_qubits
        self.target_qubit = target_qubit

    def _create_matrix(self) -> np.ndarray:
        return create_double_controlled_matrix(
            BASE_MATRIX,
            control_qubits=self.control_qubits,
            target_qubit=self.target_qubit,
            qubit_num=self.qubit_num,
        )

    def _create_repr(self) -> str:
        return f"CCZ{self.qubit_num}(control_qubits={self.control_qubits},target_qubit={self.target_qubit})"
