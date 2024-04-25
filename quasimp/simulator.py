#!/usr/bin/env python3

import numpy as np
from typing import List, Dict

from .circuit import Circuit
from .gates import Gate


class QuaSimP:
    _gate_frequency_cache: Dict
    _ngram_matrix_cache: Dict

    def __init__(self) -> None:
        pass

    def _build_gate_frequency_dict(self, circuits: List[Circuit]) -> Dict:
        gate_frequency_dict = {}

        for circuit in circuits:
            for gate in circuit.gates:
                if str(gate) in gate_frequency_dict:
                    gate_frequency_dict[str(gate)] += 1
                else:
                    gate_frequency_dict[str(gate)] = 1

        return gate_frequency_dict

    def _build_ngram_frequency_dict(
        self, circuits: List[Circuit], gate_frequency_dict: Dict, support_count: int
    ) -> Dict:
        ngram_frequency_dict = {}
        for circuit in circuits:
            gate_sequences = [[]]
            for gate in circuit.gates:
                if gate_frequency_dict[str(gate)] >= support_count:
                    gate_sequences[len(gate_sequences) - 1].append(gate)
                else:
                    gate_sequences.append([])

            gate_sequences = [
                gate_sequence
                for gate_sequence in gate_sequences
                if len(gate_sequence) > 0
            ]

            for gate_sequence in gate_sequences:
                for i in range(len(gate_sequence)):
                    for ngram_length in range(2, len(gate_sequence) - i + 1):
                        ngram = "_".join(
                            [str(gate) for gate in gate_sequence[i : i + ngram_length]]
                        )
                        ngram_gates = gate_sequence[i : i + ngram_length]

                        if ngram in ngram_frequency_dict:
                            ngram_frequency_dict[ngram]["count"] += 1
                        else:
                            ngram_frequency_dict[ngram] = {
                                "count": 1,
                                "gates": ngram_gates,
                            }

        return ngram_frequency_dict

    def build_cache(self, circuits: List[Circuit], support_count: int) -> None:
        self._gate_frequency_cache = {}
        self._ngram_matrix_cache = {}

        gate_frequency_dict = self._build_gate_frequency_dict(circuits)
        ngram_frequency_dict = self._build_ngram_frequency_dict(
            circuits, gate_frequency_dict, support_count=support_count
        )

        for ngram in ngram_frequency_dict:
            if ngram_frequency_dict[ngram]["count"] < support_count:
                continue

            ngram_matrix = None
            for gate in reversed(ngram_frequency_dict[ngram]["gates"]):
                if ngram_matrix is None:
                    ngram_matrix = gate.matrix

                else:
                    ngram_matrix = np.matmul(gate.matrix, ngram_matrix)

            self._ngram_matrix_cache[ngram] = ngram_matrix

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
