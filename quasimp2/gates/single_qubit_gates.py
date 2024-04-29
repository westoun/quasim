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


class X(Gate):
    matrix: np.ndarray = X_MATRIX


class Y(Gate):
    matrix: np.ndarray = Y_MATRIX


class Z(Gate):
    matrix: np.ndarray = Z_MATRIX


class RX(Gate):
    matrix: np.ndarray

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.matrix = RX_MATRIX(theta)


class RY(Gate):
    matrix: np.ndarray

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.matrix = RY_MATRIX(theta)


class RZ(Gate):
    matrix: np.ndarray

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.matrix = RZ_MATRIX(theta)


class Phase(Gate):
    matrix: np.ndarray

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.matrix = PHASE_MATRIX(theta)
