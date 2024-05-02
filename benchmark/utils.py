#!/usr/bin/env python3

import numpy as np
from random import choice, randint, sample, random
from typing import List, Tuple

from quasimp2 import QuaSimP2 as QuaSimP, Circuit
from quasimp2.gates import (
    CH,
    CRZ,
    CRX,
    CRY,
    CX,
    CY,
    CZ,
    H,
    RX,
    RY,
    RZ,
    X,
    Y,
    Z,
    CCX,
    CCZ,
    Phase,
    Swap,
)

from qiskit import QuantumCircuit, Aer


GATES = [
    "H",
    "X",
    "Y",
    "Z",
    "CY",
    "CX",
    "CY",
    "CZ",
    "CRX",
    "CRY",
    "CRZ",
    "RX",
    "RY",
    "RZ",
    "CCX",
    "CCZ",
    "PHASE",
    "SWAP",
]


def create_random_circuits(
    gate_count: int = 10, qubit_num: int = 4
) -> Tuple[QuantumCircuit, Circuit]:
    qiskit_circuit = QuantumCircuit(qubit_num)
    quasimp_circuit = Circuit(qubit_num)

    for _ in range(gate_count):
        gate_type = choice(GATES)

        if gate_type == "H":
            target_qubit = randint(0, qubit_num - 1)
            qiskit_circuit.h(target_qubit)
            quasimp_circuit.apply(H(target_qubit))

        elif gate_type == "X":
            target_qubit = randint(0, qubit_num - 1)
            qiskit_circuit.x(target_qubit)
            quasimp_circuit.apply(X(target_qubit))

        elif gate_type == "Y":
            target_qubit = randint(0, qubit_num - 1)
            qiskit_circuit.y(target_qubit)
            quasimp_circuit.apply(Y(target_qubit))

        elif gate_type == "Z":
            target_qubit = randint(0, qubit_num - 1)
            qiskit_circuit.z(target_qubit)
            quasimp_circuit.apply(Z(target_qubit))

        elif gate_type == "CH":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            qiskit_circuit.ch(control_qubit, target_qubit)
            quasimp_circuit.apply(CH(control_qubit, target_qubit))

        elif gate_type == "CX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            qiskit_circuit.cx(control_qubit, target_qubit)
            quasimp_circuit.apply(CX(control_qubit, target_qubit))

        elif gate_type == "CY":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            qiskit_circuit.cy(control_qubit, target_qubit)
            quasimp_circuit.apply(CY(control_qubit, target_qubit))

        elif gate_type == "CZ":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            qiskit_circuit.cz(control_qubit, target_qubit)
            quasimp_circuit.apply(CZ(control_qubit, target_qubit))

        elif gate_type == "RX":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.rx(theta, target_qubit)
            quasimp_circuit.apply(RX(target_qubit, theta))

        elif gate_type == "RY":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.ry(theta, target_qubit)
            quasimp_circuit.apply(RY(target_qubit, theta))

        elif gate_type == "RZ":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.rz(theta, target_qubit)
            quasimp_circuit.apply(RZ(target_qubit, theta))

        elif gate_type == "CRX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.crx(theta, control_qubit, target_qubit)
            quasimp_circuit.apply(CRX(control_qubit, target_qubit, theta))

        elif gate_type == "CRY":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.cry(theta, control_qubit, target_qubit)
            quasimp_circuit.apply(CRY(control_qubit, target_qubit, theta))

        elif gate_type == "CRZ":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.crz(theta, control_qubit, target_qubit)
            quasimp_circuit.apply(CRZ(control_qubit, target_qubit, theta))

        elif gate_type == "CCX":
            target_qubit, control_qubit1, control_qubit2 = sample(
                range(0, qubit_num), 3
            )
            qiskit_circuit.ccx(control_qubit1, control_qubit2, target_qubit)
            quasimp_circuit.apply(CCX(control_qubit1, control_qubit2, target_qubit))

        elif gate_type == "CCZ":
            target_qubit, control_qubit1, control_qubit2 = sample(
                range(0, qubit_num), 3
            )
            qiskit_circuit.ccz(control_qubit1, control_qubit2, target_qubit)
            quasimp_circuit.apply(CCZ(control_qubit1, control_qubit2, target_qubit))

        elif gate_type == "PHASE":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            qiskit_circuit.p(theta, target_qubit)
            quasimp_circuit.apply(Phase(target_qubit, theta))

        elif gate_type == "SWAP":
            qubit1, qubit2 = sample(range(0, qubit_num), 2)

            qiskit_circuit.swap(qubit1=qubit1, qubit2=qubit2)
            quasimp_circuit.apply(Swap(qubit1, qubit2))

        else:
            raise NotImplementedError()

    return (qiskit_circuit, quasimp_circuit)


def evaluate_qiskit_circuits(
    circuits: List[QuantumCircuit], backend
) -> List[List[float]]:
    probabilities = []

    for circuit in circuits:
        circuit = circuit.decompose(reps=5)

        job = backend.run(circuit)

        result = job.result()

        output_state = result.get_statevector(circuit, decimals=3)
        output_state = output_state.reverse_qargs()

        circuit_probabilities = output_state.probabilities().tolist()
        probabilities.append(circuit_probabilities)

    return probabilities


def evaluate_quasimp_circuits(
    circuits: List[Circuit], simulator: QuaSimP
) -> List[List[float]]:
    simulator.evaluate(circuits)

    probabilities = []
    for circuit in circuits:
        circuit_probabilities = circuit.probabilities.tolist()
        probabilities.append(circuit_probabilities)

    return probabilities
