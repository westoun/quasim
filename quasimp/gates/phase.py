#!/usr/bin/env python3

import cmath
import numpy as np

from .gate import Gate
from .utils import create_matrix


class Phase(Gate):
    qubit_num: int
    target_qubit: int
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta

    def _create_matrix(self) -> np.ndarray:
        base_matrix = np.array(
            [[1, 0], [0, cmath.exp(1j * self.theta)]],
            dtype=np.complex128,
        )

        return create_matrix(
            base_matrix, target_qubit=self.target_qubit, qubit_num=self.qubit_num
        )

    def _create_repr(self) -> str:
        return f"Phase{self.qubit_num}(target_qubit={self.target_qubit}, theta={round(self.theta, 4)})"
