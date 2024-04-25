#!/usr/bin/env python3

import numpy as np


def create_identity(dim: int = 2) -> np.ndarray:
    return np.eye(dim, dtype=complex)
