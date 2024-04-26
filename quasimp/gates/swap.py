#!/usr/bin/env python3

import numpy as np

from .gate import Gate
from .utils import create_controlled_matrix

CX_BASE_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)


class Swap(Gate):
    qubit_num: int
    qubit1: int
    qubit2: int

    def __init__(self, qubit1: int, qubit2: int) -> None:
        if qubit1 < qubit2:
            self.qubit1, self.qubit2 = qubit1, qubit2
        else:
            self.qubit1, self.qubit2 = qubit2, qubit1

    def _create_matrix(self) -> np.ndarray:
        # Inspired by https://quantumcomputing.stackexchange.com/a/24051
        cx1_matrix = create_controlled_matrix(
            base_matrix=CX_BASE_MATRIX,
            control_qubit=self.qubit1,
            target_qubit=self.qubit2,
            qubit_num=self.qubit_num,
        )
        cx2_matrix = create_controlled_matrix(
            base_matrix=CX_BASE_MATRIX,
            control_qubit=self.qubit2,
            target_qubit=self.qubit1,
            qubit_num=self.qubit_num,
        )

        swap_matrix = np.matmul(cx1_matrix, np.matmul(cx2_matrix, cx1_matrix))
        return swap_matrix

    def _create_repr(self) -> str:
        return f"Swap{self.qubit_num}(qubit1={self.qubit1},qubit2={self.qubit2})"
