#!/usr/bin/env python3


from quasim import QuaSim, Circuit
from quasim.gates import H, CX, X


if __name__ == "__main__":
    circuit = Circuit(3)
    circuit.apply(H(0))
    circuit.apply(CX(0, 1))

    simulator = QuaSim()
    simulator.evaluate_circuit(circuit)

    print(circuit.state)
    print(circuit.probabilities)
    print(circuit.probability_dict)
