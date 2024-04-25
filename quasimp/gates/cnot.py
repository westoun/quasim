#!/usr/bin/env python3

import numpy as np

from .gate import Gate


class CNot(Gate):
    qubit_num: int
    control_qubit: int
    target_qubit: int

    def __init__(self, control_qubit: int, target_qubit: int) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    def _create_matrix(self) -> np.ndarray:
        raise NotImplementedError()

    def _create_repr(self) -> str:
        return f"CNot{self.qubit_num}(control_qubit={self.control_qubit},target_qubit={self.target_qubit})"
