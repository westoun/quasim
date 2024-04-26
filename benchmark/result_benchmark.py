#!/usr/bin/env python3

from datetime import datetime
import numpy as np
from pprint import pprint
from scipy.spatial import distance
from typing import List, Tuple

from quasimp import QuaSimP, Circuit

from qiskit import QuantumCircuit, Aer

from .utils import (
    create_random_circuits,
    evaluate_qiskit_circuits,
    evaluate_quasimp_circuits,
)


def run_result_benchmark():
    CIRCUIT_COUNT = 100
    GATE_COUNT = 40
    QUBIT_NUM = 4

    qiskit_backend = Aer.get_backend("statevector_simulator")
    quasimp_simulator = QuaSimP()

    for _ in range(CIRCUIT_COUNT):
        qiskit_circuit, quasimp_circuit = create_random_circuits(
            gate_count=GATE_COUNT, qubit_num=QUBIT_NUM
        )

        qiskit_probabilities = evaluate_qiskit_circuits(
            [qiskit_circuit], backend=qiskit_backend
        )[0]
        quasimp_probabilities = evaluate_quasimp_circuits(
            [quasimp_circuit], simulator=quasimp_simulator
        )[0]

        error = distance.jensenshannon(qiskit_probabilities, quasimp_probabilities)

        if error > 0.01:
            print(f"Encountered strong divergence ({error}) on")
            print(f"\t{quasimp_circuit}")

    else:
        print(
            f"Finished result benchmarking. No significant divergences between qiskit and quasimp encountered."
        )
