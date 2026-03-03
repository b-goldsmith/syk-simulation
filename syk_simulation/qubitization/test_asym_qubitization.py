from syk_simulation.qubitization.asymmetric_qubitization import OracleB, OracleA, Select
from psiqworkbench import QPU, Qubits
import numpy as np
from math import isclose, factorial
import pytest
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask, pauli_sum_to_numpy
from scipy.stats import normaltest, skewtest
from scipy import stats


def run_test_oracleb():
    """This test is to confirm that Oracle B is correctly resulting in
        $B_l = 1 / \sqrt{L}$
    which is in accordance with 2019 Babbush et. al. paper in the first
    paragraph of section III A.
    """
    num_qubits = np.random.choice(range(1, 21))
    qpu = QPU(num_qubits=num_qubits, filters=[">>state-vector-sim>>"])
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    state = index.pull_state()
    for coeff in state:
        assert np.isclose(np.abs(coeff), 1 / np.sqrt(2**num_qubits))


def test_oracleb():
    for _ in range(10):
        run_test_oracleb()


def test_oracleb_reversible():
    """Test to ensure that Oracle B is producing a Unitary that is reversible."""
    num_qubits = np.random.choice(range(1, 21))
    qpu = QPU(num_qubits=num_qubits)
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    oracleB.uncompute()
    state = index.pull_state()
    assert abs(state[0] - 1) < 1e-8


def run_test_oraclea(N=32):
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
    if N <= 8:
        mean_tolerance = 1e-2
        max_tolerance = 0.2
        random_depth = 10 * num_qubits
    elif N <= 16:
        mean_tolerance = 1e-2
        max_tolerance = 0.2
        random_depth = 5 * num_qubits
    elif N <= 32:
        random_depth = 3 * num_qubits
        mean_tolerance = 1e-3
        max_tolerance = 0.1
    else:
        random_depth = 2 * num_qubits
        mean_tolerance = 1e-3
        max_tolerance = 0.1
    rtol = 1e-3
    atol = 1e-4

    oracleA = OracleA()
    oracleA.compute(index=index, random_depth=random_depth)
    state = index.pull_state()

    assert abs(np.mean(state.real)) < mean_tolerance
    assert isclose(ideal_variance, np.var(state.real), rel_tol=rtol, abs_tol=atol)
    assert np.max(np.abs(state.real)) < max_tolerance


@pytest.mark.parametrize("N", (8, 16, 32, 64))  # 128 must be run manually due to memory
def test_oraclea(N):
    run_test_oraclea(N)


def extract_syk_coefficients(statevector, n_qubits, N_majoranas):
    bits_per_idx = n_qubits // 4
    hamiltonian_terms = {}

    for i, amplitude in enumerate(statevector):
        if abs(amplitude) < 1e-10:
            continue  # Skip "garbage" states

        # Slicing the index i into p, q, r, s
        # This assumes Big-Endian: [p][q][r][s]
        p = (i >> (bits_per_idx * 3)) & (2**bits_per_idx - 1)
        q = (i >> (bits_per_idx * 2)) & (2**bits_per_idx - 1)
        r = (i >> (bits_per_idx * 1)) & (2**bits_per_idx - 1)
        s = (i >> (bits_per_idx * 0)) & (2**bits_per_idx - 1)

        # SYK Constraint: terms only exist for p < q < r < s
        if p < q < r < s < N_majoranas:
            hamiltonian_terms[(p, q, r, s)] = amplitude

    return hamiltonian_terms


def test_oraclea_reversible():
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


@pytest.mark.parametrize("N", (4, 5, 6, 7, 8))
def test_select(N):
    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size))) + 1  # aux_index + range_flag
    total_qubits = system_size + index_size + aux_size
    for _ in range(3):
        run_select_test(system_size, index_chunk_size, index_size, total_qubits)


def run_select_test(system_size, index_chunk_size, index_size, total_qubits):
    """This is testing the select to ensure that
    |l>\gamma_l|\psi> = |l>X_l \cdotp Z_(l-1) \cdotp Z_(l-2) \cdotp\cdotp\cdotp Z_0|\psi>
    - from 2019 Babbush et. al. paper equation 18.
    """

    qpu = QPU(num_qubits=total_qubits)
    qpu2 = QPU(num_qubits=system_size)

    # test simple case where index is a single int
    index_value_p = np.random.choice(range(system_size))
    index_value_q = np.random.choice(range(system_size))
    index_value_r = np.random.choice(range(system_size))
    index_value_s = np.random.choice(range(system_size))

    index = Qubits(index_size, "index", qpu=qpu)
    index[0:index_chunk_size].write(index_value_p)
    index[index_chunk_size : 2 * index_chunk_size].write(index_value_q)
    index[2 * index_chunk_size : 3 * index_chunk_size].write(index_value_r)
    index[3 * index_chunk_size : 4 * index_chunk_size].write(index_value_s)

    system = Qubits(system_size, "system", qpu=qpu)
    select = Select()
    select.compute(index=index, system=system)

    system_state = system.pull_state()

    hardcoded_system = Qubits(system_size, qpu=qpu2)

    for index_pqrs in [index_value_p, index_value_q, index_value_r, index_value_s]:
        hardcoded_system[index_pqrs].x()
        for z_chain_index in range(index_pqrs):
            hardcoded_system[z_chain_index].z()

    hardcoded_state = hardcoded_system.pull_state()

    fidelity = np.abs(np.vdot(hardcoded_state, system_state)) ** 2

    assert np.isclose(fidelity, 1.0, atol=1e-7)
    assert index[0:index_chunk_size].read() == index_value_p
    assert index[index_chunk_size : 2 * index_chunk_size].read() == index_value_q
    assert index[2 * index_chunk_size : 3 * index_chunk_size].read() == index_value_r
    assert index[3 * index_chunk_size : 4 * index_chunk_size].read() == index_value_s


def test_select_reversible():
    """Test to ensure that SELECT is producing a Unitary that is reversible by
    confirming that $SELECT_dagger SELECT = I$"""
    N = np.random.choice(range(4, 9))
    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size))) + 1  # aux_index + range_flag
    total_qubits = system_size + index_size + aux_size
    qpu = QPU(num_qubits=total_qubits)

    index_value_p = np.random.choice(range(system_size))
    index_value_q = np.random.choice(range(system_size))
    index_value_r = np.random.choice(range(system_size))
    index_value_s = np.random.choice(range(system_size))

    index = Qubits(index_size, "index", qpu=qpu)

    index[0:index_chunk_size].write(index_value_p)
    index[index_chunk_size : 2 * index_chunk_size].write(index_value_q)
    index[2 * index_chunk_size : 3 * index_chunk_size].write(index_value_r)
    index[3 * index_chunk_size : 4 * index_chunk_size].write(index_value_s)

    system = Qubits(system_size, "system", qpu=qpu)
    select = Select()
    original_state = qpu.pull_state()
    select.compute(index, system)
    select.uncompute()
    final_state = qpu.pull_state()
    assert np.allclose(original_state, final_state, atol=1e-7)
