#!/usr/bin/env python3

import math
import numpy as np

from .gate import Gate
from .utils import create_controlled_matrix


class CRX(Gate):
    qubit_num: int
    control_qubit: int
    target_qubit: int
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta

    def _create_matrix(self) -> np.ndarray:
        base_matrix = np.array(
            [
                [math.cos(self.theta / 2), -1j * math.sin(self.theta / 2)],
                [-1j * math.sin(self.theta / 2), math.cos(self.theta / 2)],
            ],
            dtype=np.complex128,
        )

        return create_controlled_matrix(
            base_matrix,
            control_qubit=self.control_qubit,
            target_qubit=self.target_qubit,
            qubit_num=self.qubit_num,
        )

    def _create_repr(self) -> str:
        return f"CRX{self.qubit_num}(control_qubit={self.control_qubit},target_qubit={self.target_qubit},theta={round(self.theta, 4)})"
