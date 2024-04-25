#!/usr/bin/env python3

import numpy as np


def create_identity(dim: int = 2) -> np.ndarray:
    return np.eye(dim, dtype=np.complex128)


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


# Corresponds to |0> <0|
PROJECTOR_0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)

# Corresponds to |1> <1|
PROJECTOR_1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def create_controlled_matrix(
    base_matrix: np.ndarray, control_qubit: int, target_qubit: int, qubit_num: int
) -> np.ndarray:
    # Based on https://quantumcomputing.stackexchange.com/a/4255
    control_matrix = create_matrix(
        PROJECTOR_0, target_qubit=control_qubit, qubit_num=qubit_num
    )

    target_matrix = None
    for i in range(qubit_num):
        # first matrix to be added
        if target_matrix is None:
            if i == control_qubit:
                target_matrix = PROJECTOR_1
            elif i == target_qubit:
                target_matrix = base_matrix
            else:
                target_matrix = create_identity(2)

        else:
            if i == control_qubit:
                target_matrix = np.kron(target_matrix, PROJECTOR_1)
            elif i == target_qubit:
                target_matrix = np.kron(target_matrix, base_matrix)
            else:
                target_matrix = np.kron(target_matrix, create_identity(2))

    # projector matrix at position of control qubit
    # base matrix at position of target qubit

    matrix = control_matrix + target_matrix
    return matrix
