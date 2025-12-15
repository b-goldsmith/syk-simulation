"""This file contains utilities used for implementing the QDrift algorithm"""

import numpy as np


def sample_distribution(coefficients: list[float]) -> int:
    """Takes a list of coefficients and takes a sample from the corresponding probability distribution."""
    normalized_coefficients = np.array(np.abs(coefficients)) / sum(np.abs(coefficients))
    return np.random.choice(len(normalized_coefficients), p=normalized_coefficients)
