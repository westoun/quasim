#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


class Gate(ABC):
    qubit_num: int

    _matrix: np.ndarray = None

    def set_qubit_num(self, qubit_num: int) -> None:
        # Chose not to provide qubit num in __init__,
        # to reduce amount of boilerplate parameter
        # and increase usability.
        self.qubit_num = qubit_num

    @property
    def matrix(self) -> np.ndarray:
        assert (
            self.qubit_num is not None
        ), "set_qubit_num has to be called before a matrix can be created."

        if self._matrix is not None:
            return self._matrix

        self._matrix = self._create_matrix()
        return self._matrix

    @abstractmethod
    def _create_matrix(self) -> np.ndarray: ...

    def __repr__(self) -> str:
        assert (
            self.qubit_num is not None
        ), "set_qubit_num has to be called before __repr__ can be called."

        return self._create_repr()

    @abstractmethod
    def _create_repr(self) -> str: ...

    def __str__(self) -> str:
        return self.__repr__()
