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

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit})"


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
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RX_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"


class RY(Gate):
    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RY_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"


class RZ(Gate):
    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RZ_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"


class Phase(Gate):
    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = PHASE_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"
