#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from typing import List

from .interface import IGate


@dataclass
class SwapGate(IGate):
    qubit1: int
    qubit2: int
    matrix: np.ndarray = None

    @property
    def qubits(self) -> List[int]:
        return [self.qubit1, self.qubit2]
