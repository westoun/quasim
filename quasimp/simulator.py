#!/usr/bin/env python3

import numpy as np
from typing import List, Dict

from .circuit import Circuit
from .gates import Gate


class QuaSimP:
    _gate_frequency_dict: Dict
    _ngram_frequency_dict: Dict

    def __init__(self) -> None:
        pass

    def build_cache(self, circuits: List[Circuit]) -> None:
        self._gate_frequency_dict = {}
        self._ngram_frequency_dict = {}

        raise NotImplementedError()

    def evaluate(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            self.evaluate_circuit(circuit)

    def evaluate_circuit(self, circuit: Circuit) -> None:
        if circuit.state is not None:
            return circuit.state

        state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
        state[0] = 1

        for gate in circuit.gates:
            state = np.matmul(gate.matrix, state)

        circuit.set_state(state)
