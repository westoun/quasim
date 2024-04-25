#!/usr/bin/env python3


from quasimp import QuaSimP, Circuit
from quasimp.gates import H, CNot, X


if __name__ == "__main__":
    circuit = Circuit(3)
    circuit.apply(H(0))
    circuit.apply(CNot(0, 1))

    simulator = QuaSimP()
    simulator.evaluate_circuit(circuit)

    print(circuit.state)
    print(circuit.probabilities)
    print(circuit.probability_dict)
