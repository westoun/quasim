#!/usr/bin/env python3

from datetime import datetime
import numpy as np
from pprint import pprint
from random import choice, randint, sample, random
from typing import List, Tuple

from quasimp import QuaSimP, Circuit
from quasimp.gates import CH, CRZ, CRX, CRY, CX, CY, CZ, H, RX, RY, RZ, X, Y, Z

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

        else:
            raise NotImplementedError()

    return (qiskit_circuit, quasimp_circuit)


def evaluate_qiskit_circuits(
    circuits: List[QuantumCircuit], backend
) -> List[List[float]]:
    probabilities = []

    for circuit in circuits:
        circuit = circuit.decompose(reps=1)

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


if __name__ == "__main__":
    CIRCUIT_COUNT = 1000
    GATE_COUNT = 10
    QUBIT_NUM = 5

    qiskit_backend = Aer.get_backend("statevector_simulator")
    quasimp_simulator = QuaSimP()

    qiskit_circuits = []
    quasimp_circuits = []
    for _ in range(CIRCUIT_COUNT):
        qiskit_circuit, quasimp_circuit = create_random_circuits(
            gate_count=GATE_COUNT, qubit_num=QUBIT_NUM
        )
        qiskit_circuits.append(qiskit_circuit)
        quasimp_circuits.append(quasimp_circuit)

    start = datetime.now()
    qiskit_probabilities = evaluate_qiskit_circuits(
        qiskit_circuits, backend=qiskit_backend
    )
    end = datetime.now()
    qiskit_duration = end - start

    start = datetime.now()
    quasimp_probabilities = evaluate_quasimp_circuits(
        quasimp_circuits, simulator=quasimp_simulator
    )
    end = datetime.now()
    quasimp_duration = end - start

    print(
        f"Finished evaluation benchmarking on {CIRCUIT_COUNT} circuits with {GATE_COUNT} gates and {QUBIT_NUM} qubits each."
    )
    print("Qiskit duration: ", qiskit_duration)
    print("Quasimp duration: ", quasimp_duration)
