#!/usr/bin/env python3

from datetime import datetime
import numpy as np
from pprint import pprint
from random import choice, randint, sample, random
from typing import List, Tuple

from quasimp2 import QuaSimP2 as QuaSimP, Circuit

from qiskit import QuantumCircuit, Aer

from .utils import (
    create_random_circuits,
    evaluate_qiskit_circuits,
    evaluate_quasimp_circuits,
)


def run_time_benchmark():
    CIRCUIT_COUNT = 1000
    GATE_COUNT = 40
    QUBIT_NUM = 3

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
