# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-07-14

- add S, T, Controlled-S, and Controlled-Phase gate.

## [0.2.1] - 2024-07-14

- remove unneeded qiskit-terra dependency.

## [0.2.0] - 2024-05-14

- adjust simulation code to simplify controlled gates if control qubits are either inactive (|s> = |0>) or active, but not in a superposition state (|s> = |1>).
- refactor function structure and move some functions to utils.
- adjust benchmarking setup and add speed comparison to readme.

## [0.1.0] - 2024-05-02

- initial version.
