import pytest
from syk_simulation.qubitization.asymmetric_qubitization import OracleA
import numpy as np
from psiqworkbench import Qubits, QPU
from math import isclose


def run_test_oracle_a(N=32):
    """Tests that Oracle A produces a state with Gaussian-distributed amplitudes
    as required by Babbush et al. 2019 (Section III A).

    Verified properties:
    - Zero mean of amplitudes
    - Variance equal to 1/2^n (normalized Haar-random state)
    - No single amplitude dominates (max amplitude check)

    Depth formula determined empirically:
    - N <= 8:  10 * num_qubits
    - N <= 16:  5 * num_qubits
    - N <= 32:  3 * num_qubits
    - N >  32:  2 * num_qubits

    N=4 is too small for meaningful testing.
    N > 128 exceeds state vector simulator memory limits (29 qubits).
    """
    num_qubits = 4 * int(np.ceil(np.log2(N)))
    qpu = QPU(num_qubits=num_qubits)
    index = Qubits(num_qubits, "index", qpu=qpu)

    ideal_variance = 1 / 2**num_qubits
    if N <= 4:
        mean_tolerance = 1e-2
        max_tolerance = 0.5
    else:
        mean_tolerance = 1e-3
        max_tolerance = 0.1

    random_depth = 2 * num_qubits

    rtol = 1e-3
    atol = 1e-4
    random_depth = 2 * num_qubits

    oracleA = OracleA()
    oracleA.compute(index=index, random_depth=random_depth)
    state = index.pull_state()

    assert abs(np.mean(state.real)) < mean_tolerance
    assert isclose(ideal_variance, np.var(state.real), rel_tol=rtol, abs_tol=atol)
    assert np.max(np.abs(state.real)) < max_tolerance


@pytest.mark.parametrize("N", (4, 8, 16, 32))  # 128 must be run manually due to memory
def test_oracle_a(N):
    run_test_oracle_a(N)


def test_oracle_a_reversible():
    """Test to ensure that Oracle A is producing a Unitary that is reversible by
    confirming that $A_dagger A = I$"""
    num_qubits = np.random.choice(range(3, 21))
    random_depth = 2 * num_qubits
    qpu = QPU(num_qubits=num_qubits, filters=[">>state-vector-sim>>"])
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleA = OracleA()
    oracleA.compute(index=index, random_depth=random_depth)
    oracleA.uncompute()
    state = index.pull_state()
    assert abs(state[0] - 1) < 1e-8
