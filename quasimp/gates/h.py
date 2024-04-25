#!/usr/bin/env python3

import numpy as np

from .gate import Gate
from ._utils import create_identity


class H(Gate):
    qubit_num: int
    target_qubit: int

    def __init__(self, target_qubit: int) -> None:
        self.target_qubit = target_qubit

    def _create_matrix(self) -> np.ndarray:
        matrix = 1.0 / (2.0**0.5) * np.array([[1, 1], [1, -1]], dtype=complex)

        if self.qubit_num == 1:
            return matrix

        qubits_before = self.target_qubit
        if qubits_before > 0:
            identity_before = create_identity(dim=2**qubits_before)
            matrix = np.kron(identity_before, matrix)

        qubits_after = self.qubit_num - (self.target_qubit + 1)
        if qubits_after > 0:
            identity_after = create_identity(dim=2**qubits_after)
            matrix = np.kron(matrix, identity_after)

        return matrix

    def _create_repr(self) -> str:
        return f"H{self.qubit_num}(target_qubit={self.target_qubit})"
