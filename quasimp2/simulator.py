#!/usr/bin/env python3

import numpy as np
from typing import List, Dict

from .circuit import Circuit
from .gates import IGate, SwapGate, Gate, CGate, CCGate
from .gates.utils import (
    create_double_controlled_matrix,
    create_controlled_matrix,
    create_matrix,
)

QUBIT_STARTING_STATE = np.zeros(2, dtype=np.complex128)
QUBIT_STARTING_STATE[0] = 1


class QuaSimP2:
    def __init__(self) -> None:
        pass

    def evaluate(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            self.evaluate_circuit(circuit)

    def evaluate_circuit(self, circuit: Circuit) -> None:
        if circuit.state is not None:
            return circuit.state

        qubit_map = {}
        for i in range(circuit.qubit_num):
            qubit_map[i] = i

        qubit_groups = []
        for i in range(circuit.qubit_num):
            qubit_group = {"qubits": [i], "state": QUBIT_STARTING_STATE}
            qubit_groups.append(qubit_group)

        for gate in circuit.gates:
            if type(gate) == SwapGate:
                qubit_target1 = qubit_map[gate.qubit1]
                qubit_target2 = qubit_map[gate.qubit2]

                qubit_map[gate.qubit1] = qubit_target2
                qubit_map[gate.qubit2] = qubit_target1

            qubits = [qubit_map[qubit] for qubit in gate.qubits]

            relevant_qubit_groups = []
            for qubit_group in qubit_groups:
                for qubit in qubit_group["qubits"]:
                    if qubit in qubits:
                        if qubit_group not in relevant_qubit_groups:
                            relevant_qubit_groups.append(qubit_group)

                        break

            relevant_qubit_group = relevant_qubit_groups[0]
            if len(relevant_qubit_groups) > 1:
                for qubit_group in relevant_qubit_groups[1:]:
                    relevant_qubit_group["qubits"].extend(qubit_group["qubits"])
                    relevant_qubit_group["state"] = np.kron(
                        relevant_qubit_group["state"], qubit_group["state"]
                    )

                    qubit_groups.remove(qubit_group)

            qubit_num = len(relevant_qubit_group["qubits"])
            if issubclass(gate.__class__, Gate):
                target_qubit = relevant_qubit_group["qubits"].index(
                    qubit_map[gate.target_qubit]
                )
                matrix = create_matrix(
                    gate.matrix, target_qubit=target_qubit, qubit_num=qubit_num
                )
                relevant_qubit_group["state"] = np.matmul(
                    matrix, relevant_qubit_group["state"]
                )

            elif issubclass(gate.__class__, CGate):
                target_qubit = relevant_qubit_group["qubits"].index(
                    qubit_map[gate.target_qubit]
                )
                control_qubit = relevant_qubit_group["qubits"].index(
                    qubit_map[gate.control_qubit]
                )
                matrix = create_controlled_matrix(
                    gate.matrix,
                    control_qubit=control_qubit,
                    target_qubit=target_qubit,
                    qubit_num=qubit_num,
                )
                relevant_qubit_group["state"] = np.matmul(
                    matrix, relevant_qubit_group["state"]
                )

            elif issubclass(gate.__class__, CCGate):
                target_qubit = relevant_qubit_group["qubits"].index(
                    qubit_map[gate.target_qubit]
                )
                control_qubit1 = relevant_qubit_group["qubits"].index(
                    qubit_map[gate.control_qubit1]
                )
                control_qubit2 = relevant_qubit_group["qubits"].index(
                    qubit_map[gate.control_qubit2]
                )
                matrix = create_double_controlled_matrix(
                    gate.matrix,
                    control_qubit1=control_qubit1,
                    control_qubit2=control_qubit2,
                    target_qubit=target_qubit,
                    qubit_num=qubit_num,
                )
                relevant_qubit_group["state"] = np.matmul(
                    matrix, relevant_qubit_group["state"]
                )

            else:
                raise NotImplementedError(
                    f"Unknown gate type for {gate} ({type(gate)})"
                )

        reverse_qubit_map = {}
        for i in qubit_map:
            reverse_qubit_map[qubit_map[i]] = i
