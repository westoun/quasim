#!/usr/bin/env python3

from datetime import datetime
import numpy as np
from pprint import pprint
from random import choice, randint, sample, random
from typing import List, Tuple

from quasim import QuaSim, Circuit

from qiskit import QuantumCircuit, Aer

from .utils import (
    create_random_circuits,
    evaluate_qiskit_circuits,
    evaluate_quasim_circuits,
)


def run_time_benchmark(circuit_count=1000, gate_count=40, qubit_num=3):

    qiskit_backend = Aer.get_backend("statevector_simulator")
    quasim_simulator = QuaSim()

    qiskit_circuits = []
    quasim_circuits = []
    for _ in range(circuit_count):
        qiskit_circuit, quasim_circuit = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )
        qiskit_circuits.append(qiskit_circuit)
        quasim_circuits.append(quasim_circuit)

    start = datetime.now()
    qiskit_probabilities = evaluate_qiskit_circuits(
        qiskit_circuits, backend=qiskit_backend
    )
    end = datetime.now()
    qiskit_duration = end - start

    start = datetime.now()
    quasim_probabilities = evaluate_quasim_circuits(
        quasim_circuits, simulator=quasim_simulator
    )
    end = datetime.now()
    quasim_duration = end - start

    print(
        f"Finished evaluation benchmarking on {circuit_count} circuits with {gate_count} gates and {qubit_num} qubits each."
    )
    print("Qiskit duration: ", qiskit_duration)
    print("quasim duration: ", quasim_duration)
