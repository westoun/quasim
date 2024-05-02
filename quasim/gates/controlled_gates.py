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

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit})"


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
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RX_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"


class CRY(CGate):
    matrix: np.ndarray
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RY_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"


class CRZ(CGate):
    matrix: np.ndarray
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RZ_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"
