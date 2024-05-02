#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class IGate(ABC):
    matrix: np.ndarray

    @property
    @abstractmethod
    def qubits(self) -> List[int]: ...
