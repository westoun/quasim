#!/usr/bin/env python3

import numpy as np
from typing import List, Dict

from .circuit import Circuit
from .gates import IGate, Swap, Gate, CGate, CCGate
from .gates.utils import (
    create_double_controlled_matrix,
    create_controlled_matrix,
    create_matrix,
)
from .utils import QubitGroup

QUBIT_STARTING_STATE = np.zeros(2, dtype=np.complex128)
QUBIT_STARTING_STATE[0] = 1


def apply_swap_gate(qubit_groups: List[QubitGroup], gate: Swap) -> None:
    """Swap the indices of the qubits targetted by the swap gate
    in place.
    """
    for qubit_group in qubit_groups:
        for i in range(len(qubit_group.qubits)):
            if qubit_group.qubits[i] == gate.qubit1:
                qubit_group.qubits[i] = gate.qubit2

            elif qubit_group.qubits[i] == gate.qubit2:
                qubit_group.qubits[i] = gate.qubit1


def initialize_qubit_groups(qubit_num: int) -> List[QubitGroup]:
    """Create a list of qubit groups where each qubit group contains
    exactly one qubit id.
    """
    qubit_groups: List[QubitGroup] = []
    for i in range(qubit_num):
        qubit_groups.append(QubitGroup(qubits=[i], state=QUBIT_STARTING_STATE))
    return qubit_groups


def select_affected_qubit_group(
    qubit_groups: List[QubitGroup], qubit_id: int
) -> QubitGroup:
    """Select the qubit group whose qubit id
    is affected by the specified gate.
    """

    for qubit_group in qubit_groups:
        if qubit_id in qubit_group.qubits:
            return qubit_group

    raise ValueError(f"No matching qubit group found for qubit '{qubit_id}'")


def select_affected_qubit_groups(
    qubit_groups: List[QubitGroup], qubit_ids: List[int]
) -> List[QubitGroup]:
    """Select the subset of qubit groups whose qubit ids
    are affected by the specified gate.
    """

    relevant_qubit_groups: List[QubitGroup] = []
    for qubit_group in qubit_groups:
        for qubit in qubit_group.qubits:
            if qubit in qubit_ids:
                if qubit_group not in relevant_qubit_groups:
                    relevant_qubit_groups.append(qubit_group)

                break

    return relevant_qubit_groups


def merge_qubit_groups(
    relevant_groups: List[QubitGroup], total_groups: List[QubitGroup]
) -> QubitGroup:
    """Merge selected (relevant) qubit groups into a single qubit group
    and remove the previous qubit groups from the list of all qubit groups.
    The removal happens in place, while the merged qubit group is returned.
    """

    # remove duplicates
    relevant_groups = list(set(relevant_groups))

    merged_qubit_group = relevant_groups[0]
    for qubit_group in relevant_groups[1:]:
        merged_qubit_group.qubits.extend(qubit_group.qubits)
        merged_qubit_group.state = np.kron(merged_qubit_group.state, qubit_group.state)

        total_groups.remove(qubit_group)

    return merged_qubit_group


def apply_gate(relevant_qubit_group: QubitGroup, gate: Gate) -> None:
    """Apply the action of the specified gate onto the selected
    qubit group in place.
    """
    target_qubit = relevant_qubit_group.qubits.index(gate.target_qubit)
    matrix = create_matrix(
        gate.matrix,
        target_qubit=target_qubit,
        qubit_num=len(relevant_qubit_group.qubits),
    )
    relevant_qubit_group.state = np.matmul(matrix, relevant_qubit_group.state)


def apply_cgate(relevant_qubit_group: QubitGroup, gate: CGate) -> None:
    """Apply the action of the specified controlled gate onto the
    selected qubit group in place.
    """
    target_qubit = relevant_qubit_group.qubits.index(gate.target_qubit)
    control_qubit = relevant_qubit_group.qubits.index(gate.control_qubit)
    matrix = create_controlled_matrix(
        gate.matrix,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        qubit_num=len(relevant_qubit_group.qubits),
    )
    relevant_qubit_group.state = np.matmul(matrix, relevant_qubit_group.state)


def apply_ccgate(relevant_qubit_group: QubitGroup, gate: CCGate) -> None:
    """Apply the action of the specified double controlled gate onto the
    selected qubit group in place.
    """
    target_qubit = relevant_qubit_group.qubits.index(gate.target_qubit)
    control_qubit1 = relevant_qubit_group.qubits.index(gate.control_qubit1)
    control_qubit2 = relevant_qubit_group.qubits.index(gate.control_qubit2)
    matrix = create_double_controlled_matrix(
        gate.matrix,
        control_qubit1=control_qubit1,
        control_qubit2=control_qubit2,
        target_qubit=target_qubit,
        qubit_num=len(relevant_qubit_group.qubits),
    )
    relevant_qubit_group.state = np.matmul(matrix, relevant_qubit_group.state)


def aggregate_qubit_groups(qubit_groups: List[QubitGroup]) -> QubitGroup:
    """Combine a list of qubit groups into a new joined qubit group."""
    aggregated_qubit_group = qubit_groups[0]
    for qubit_group in qubit_groups[1:]:
        aggregated_qubit_group.qubits.extend(qubit_group.qubits)
        aggregated_qubit_group.state = np.kron(
            aggregated_qubit_group.state, qubit_group.state
        )
    return aggregated_qubit_group


def get_sorted_state(qubit_group: QubitGroup) -> np.ndarray:
    """Return a sorted version of the state of a qubit group
    based on the order of the group's qubit ids.
    """
    qubit_num = len(qubit_group.qubits)

    sorting_order = [0] * (2**qubit_num)
    for i in range(2**qubit_num):

        remainder = i
        for j in reversed(range(qubit_num)):

            if remainder >= 2**j:
                sorting_order[i] += 2 ** (
                    qubit_num - qubit_group.qubits[qubit_num - j - 1] - 1
                )
                remainder -= 2**j

    sorted_state = [0] * 2**qubit_num
    for i in range(2**qubit_num):
        sorted_state[sorting_order[i]] = qubit_group.state[i]

    sorted_state = np.asarray(sorted_state, dtype=np.complex128)
    return sorted_state


def is_in_ket0(qubit_group: QubitGroup) -> bool:
    if len(qubit_group.qubits) != 1:
        raise NotImplementedError()

    return qubit_group.state.tolist()[0] == 1


def is_in_ket1(qubit_group: QubitGroup) -> bool:
    if len(qubit_group.qubits) != 1:
        raise NotImplementedError()

    return qubit_group.state.tolist()[1] == 1


class QuaSim:
    """Quantum circuit simulator used to evaluate quantum
    circuits.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, circuits: List[Circuit]) -> None:
        """Evaluates a list of quantum circuits and stores the
        state at the end of each circuit in circuit.state."""

        for circuit in circuits:
            self.evaluate_circuit(circuit)

    def evaluate_circuit(self, circuit: Circuit) -> None:
        """Evaluates a quantum circuit and stores the
        state at the end of the circuit in circuit.state."""

        if circuit.state is not None:
            return circuit.state

        qubit_groups = initialize_qubit_groups(circuit.qubit_num)

        for gate in circuit.gates:
            if type(gate) == Swap:
                apply_swap_gate(qubit_groups, gate)
                continue

            if issubclass(gate.__class__, Gate):
                target_qubit_group = select_affected_qubit_group(
                    qubit_groups, qubit_id=gate.target_qubit
                )

                apply_gate(target_qubit_group, gate)

            elif issubclass(gate.__class__, CGate):
                control_qubit_group = select_affected_qubit_group(
                    qubit_groups, qubit_id=gate.control_qubit
                )
                target_qubit_group = select_affected_qubit_group(
                    qubit_groups, qubit_id=gate.target_qubit
                )

                if len(control_qubit_group.qubits) == 1:
                    # Conrol qubit is inactive
                    if is_in_ket0(control_qubit_group):
                        pass
                    # Control qubit is active
                    elif is_in_ket1(control_qubit_group):
                        apply_gate(target_qubit_group, gate)
                    # Control qubit is in superposition state
                    else:
                        merged_qubit_group = merge_qubit_groups(
                            relevant_groups=[control_qubit_group, target_qubit_group],
                            total_groups=qubit_groups,
                        )
                        apply_cgate(merged_qubit_group, gate)

                else:
                    merged_qubit_group = merge_qubit_groups(
                        relevant_groups=[control_qubit_group, target_qubit_group],
                        total_groups=qubit_groups,
                    )
                    apply_cgate(merged_qubit_group, gate)

            elif issubclass(gate.__class__, CCGate):
                control_qubit1_group = select_affected_qubit_group(
                    qubit_groups, qubit_id=gate.control_qubit1
                )
                control_qubit2_group = select_affected_qubit_group(
                    qubit_groups, qubit_id=gate.control_qubit2
                )
                target_qubit_group = select_affected_qubit_group(
                    qubit_groups, qubit_id=gate.target_qubit
                )

                if (
                    len(control_qubit1_group.qubits) == 1
                    and len(control_qubit2_group.qubits) == 1
                ):

                    # Control qubit 1 is inactive
                    if is_in_ket0(control_qubit1_group):
                        pass

                    # Control qubit 1 is active
                    elif is_in_ket1(control_qubit1_group):

                        # Control qubit 2 is inactive
                        if is_in_ket0(control_qubit2_group):
                            pass

                        # Control qubit 2 is active
                        elif is_in_ket1(control_qubit2_group):
                            apply_gate(target_qubit_group, gate)

                        # Control qubit 2 is in superposition
                        else:
                            equivalent_cgate = CGate(
                                control_qubit=gate.control_qubit2,
                                target_qubit=gate.target_qubit,
                            )
                            equivalent_cgate.matrix = gate.matrix

                            merged_qubit_group = merge_qubit_groups(
                                relevant_groups=[
                                    control_qubit2_group,
                                    target_qubit_group,
                                ],
                                total_groups=qubit_groups,
                            )
                            apply_cgate(merged_qubit_group, equivalent_cgate)

                    # Control qubit 1 is in superposition
                    else:

                        # Control qubit 2 is inactive
                        if is_in_ket0(control_qubit2_group):
                            pass

                        # Control qubit 2 is active
                        elif is_in_ket1(control_qubit2_group):
                            equivalent_cgate = CGate(
                                control_qubit=gate.control_qubit1,
                                target_qubit=gate.target_qubit,
                            )
                            equivalent_cgate.matrix = gate.matrix

                            merged_qubit_group = merge_qubit_groups(
                                relevant_groups=[
                                    control_qubit1_group,
                                    target_qubit_group,
                                ],
                                total_groups=qubit_groups,
                            )
                            apply_cgate(merged_qubit_group, equivalent_cgate)

                        # Control qubit 2 is in superposition
                        else:
                            merged_qubit_group = merge_qubit_groups(
                                relevant_groups=[
                                    control_qubit1_group,
                                    control_qubit2_group,
                                    target_qubit_group,
                                ],
                                total_groups=qubit_groups,
                            )
                            apply_ccgate(merged_qubit_group, gate)

                elif (
                    len(control_qubit1_group.qubits) == 1
                ):  # and control qubit 2 group contains multiple qubits

                    # Control qubit 1 is inactive
                    if is_in_ket0(control_qubit1_group):
                        pass

                    # Control qubit 1 is active
                    elif is_in_ket1(control_qubit1_group):
                        equivalent_cgate = CGate(
                            control_qubit=gate.control_qubit2,
                            target_qubit=gate.target_qubit,
                        )
                        equivalent_cgate.matrix = gate.matrix

                        merged_qubit_group = merge_qubit_groups(
                            relevant_groups=[
                                control_qubit2_group,
                                target_qubit_group,
                            ],
                            total_groups=qubit_groups,
                        )
                        apply_cgate(merged_qubit_group, equivalent_cgate)

                    # Control qubit 1 is in superposition
                    else:
                        merged_qubit_group = merge_qubit_groups(
                            relevant_groups=[
                                control_qubit1_group,
                                control_qubit2_group,
                                target_qubit_group,
                            ],
                            total_groups=qubit_groups,
                        )
                        apply_ccgate(merged_qubit_group, gate)

                elif (
                    len(control_qubit2_group.qubits) == 1
                ):  # and control qubit 1 group contains multiple qubits

                    # Control qubit 2 is inactive
                    if is_in_ket0(control_qubit2_group):
                        pass

                    # Control qubit 2 is active
                    elif is_in_ket1(control_qubit2_group):
                        equivalent_cgate = CGate(
                            control_qubit=gate.control_qubit1,
                            target_qubit=gate.target_qubit,
                        )
                        equivalent_cgate.matrix = gate.matrix

                        merged_qubit_group = merge_qubit_groups(
                            relevant_groups=[
                                control_qubit1_group,
                                target_qubit_group,
                            ],
                            total_groups=qubit_groups,
                        )
                        apply_cgate(merged_qubit_group, equivalent_cgate)

                    # Control qubit 2 is in superposition
                    else:
                        merged_qubit_group = merge_qubit_groups(
                            relevant_groups=[
                                control_qubit1_group,
                                control_qubit2_group,
                                target_qubit_group,
                            ],
                            total_groups=qubit_groups,
                        )
                        apply_ccgate(merged_qubit_group, gate)

                # both qubit groups contain more than 1 qubit
                else:
                    merged_qubit_group = merge_qubit_groups(
                        relevant_groups=[
                            control_qubit1_group,
                            control_qubit2_group,
                            target_qubit_group,
                        ],
                        total_groups=qubit_groups,
                    )
                    apply_ccgate(merged_qubit_group, gate)

            else:
                raise NotImplementedError(
                    f"Unknown gate type for {gate} ({type(gate)})"
                )

        aggregated_qubit_group = aggregate_qubit_groups(qubit_groups)

        sorted_state = get_sorted_state(aggregated_qubit_group)

        circuit.set_state(sorted_state)