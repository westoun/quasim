#!/usr/bin/env python3


from quasimp2 import QuaSimP2 as QuaSimP, Circuit
from quasimp2.gates import H, CX, X


if __name__ == "__main__":
    circuit = Circuit(3)
    circuit.apply(H(0))
    circuit.apply(CX(0, 1))

    simulator = QuaSimP()
    simulator.evaluate_circuit(circuit)

    print(circuit.state)
    print(circuit.probabilities)
    print(circuit.probability_dict)
