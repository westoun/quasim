#!/usr/bin/env python3

from datetime import datetime
import numpy as np
from pprint import pprint
from scipy.spatial import distance
from typing import List, Tuple

from quasim import QuaSim, Circuit

from qiskit import QuantumCircuit, Aer

from .utils import (
    create_random_circuits,
    evaluate_qiskit_circuits,
    evaluate_quasim_circuits,
)


def run_result_benchmark(circuit_count=100, gate_count=40, qubit_num=4):

    qiskit_backend = Aer.get_backend("statevector_simulator")
    quasim_simulator = QuaSim()

    for _ in range(circuit_count):
        qiskit_circuit, quasim_circuit = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )

        qiskit_probabilities = evaluate_qiskit_circuits(
            [qiskit_circuit], backend=qiskit_backend
        )[0]
        quasim_probabilities = evaluate_quasim_circuits(
            [quasim_circuit], simulator=quasim_simulator
        )[0]

        error = distance.jensenshannon(qiskit_probabilities, quasim_probabilities)

        if error > 0.01:
            print(f"\nEncountered strong divergence ({error}) on")
            print(f"\t{quasim_circuit}")
            print(qiskit_probabilities)
            print(quasim_probabilities)
            break

    else:
        print(
            f"Finished result benchmarking. No significant divergences between qiskit and quasim encountered."
        )
