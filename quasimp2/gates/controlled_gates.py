#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate
from ._matrices import (
    H_MATRIX,
    X_MATRIX,
    Y_MATRIX,
    Z_MATRIX,
    RX_MATRIX,
    RY_MATRIX,
    RZ_MATRIX,
    PHASE_MATRIX,
)


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


class CH(CGate):
    matrix: np.ndarray = H_MATRIX


class CX(CGate):
    matrix: np.ndarray = X_MATRIX


class CY(CGate):
    matrix: np.ndarray = Y_MATRIX


class CZ(CGate):
    matrix: np.ndarray = Z_MATRIX


class CRX(CGate):
    matrix: np.ndarray

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.matrix = RX_MATRIX(theta)


class CRY(CGate):
    matrix: np.ndarray

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.matrix = RY_MATRIX(theta)


class CRZ(CGate):
    matrix: np.ndarray

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.matrix = RZ_MATRIX(theta)
