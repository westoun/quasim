#!/usr/bin/env python3

import numpy as np


def create_identity(dim: int = 2) -> np.ndarray:
    return np.eye(dim, dtype=complex)


def create_matrix(
    base_matrix: np.ndarray, target_qubit: int, qubit_num: int
) -> np.ndarray:
    matrix = base_matrix

    if qubit_num == 1:
        return matrix

    qubits_before = target_qubit
    if qubits_before > 0:
        identity_before = create_identity(dim=2**qubits_before)
        matrix = np.kron(identity_before, matrix)

    qubits_after = qubit_num - (target_qubit + 1)
    if qubits_after > 0:
        identity_after = create_identity(dim=2**qubits_after)
        matrix = np.kron(matrix, identity_after)

    return matrix
