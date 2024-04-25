#!/usr/bin/env python3

import math
import numpy as np

from .gate import Gate
from .utils import create_matrix


class RY(Gate):
    qubit_num: int
    target_qubit: int
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta

    def _create_matrix(self) -> np.ndarray:
        base_matrix = np.array(
            [
                [math.cos(self.theta / 2), -1 * math.sin(self.theta / 2)],
                [math.sin(self.theta / 2), math.cos(self.theta / 2)],
            ],
            dtype=np.complex128,
        )

        return create_matrix(
            base_matrix, target_qubit=self.target_qubit, qubit_num=self.qubit_num
        )

    def _create_repr(self) -> str:
        return (
            f"RY{self.qubit_num}(target_qubit={self.target_qubit}, theta={round(self.theta, 4)})"
        )
