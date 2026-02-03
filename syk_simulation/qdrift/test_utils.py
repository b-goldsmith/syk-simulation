from syk_simulation.qdrift.utils import sample_distribution
from collections import Counter
from pytest import approx, mark

import numpy as np


def run_sample_distribution(coefficients: list[float], num_samples: int) -> list[int]:
    """Runs the sample_distribution function multiple times to gather statistics."""
    samples = []
    for _ in range(num_samples):
        sample = sample_distribution(coefficients)
        samples.append(sample)
    return samples


@mark.parametrize(
    "coefficients",
    [
        [1.0, 1.0],
        [1.0, 3.0],
        [-0.25, 0.42, 2.1, 1.18, -0.0032],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 2.0, 3.0],
        [10.0, 1.0],
        [0.1, 0.9],
    ],
)
def test_static_sample_distribution(coefficients, num_samples=10000, rel_tolerance=1e-1, abs_tolerance=1e-2):
    samples = run_sample_distribution(coefficients, num_samples)
    counts = Counter(samples)

    lambda_val = np.sum(np.abs(np.array(coefficients)))

    # Check results again the expected distribution
    for key in counts:
        ideal_probability = abs(coefficients[int(key)]) / lambda_val
        probability = counts[key] / num_samples
        assert ideal_probability == approx(probability, rel=rel_tolerance, abs=abs_tolerance)


def test_sample_distribution():
    for _ in range(10):
        num_coefficients = np.random.randint(2, 10)
        coefficients = np.random.uniform(-5.0, 5.0, size=num_coefficients).tolist()
        test_static_sample_distribution(coefficients)
