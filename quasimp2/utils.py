#!/usr/bin/env python3

import math
import numpy as np
import warnings


def probabilities_from_state(state: np.ndarray) -> np.ndarray:
    conjugate = state.conjugate()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Ignore "np.complex128Warning: Casting np.complex128 values to real discards the imaginary part"
        # since that is precisely what we want.
        probabilities = np.multiply(state, conjugate).astype(float)

    return probabilities


def probability_dict_from_state(state: np.ndarray) -> np.ndarray:
    qubit_num = int(math.log2(len(state)))

    probabilities = probabilities_from_state(state)

    probability_dict = {}
    for i, probability in enumerate(probabilities):

        if probability == 0:
            continue

        state: str = ""

        for j in reversed(range(qubit_num)):
            if i % (2 ** (j + 1)) == 0:
                state += "0"
            else:
                state += "1"

        probability_dict[state] = probability

    return probability_dict
