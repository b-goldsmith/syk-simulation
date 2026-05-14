from syk_simulation.qubitization.asymmetric_qubitization import OracleA, OracleB
from psiqworkbench import QPU, Qubits
import numpy as np

from syk_simulation.qubitization.utils import get_oraclea_coefficients, get_syk_coefficients


def test_oraclea_coeff(N=4, random_seed=2):
    """Extract the coefficients from Oracle A's state vector.
    Returns a dictionary mapping (p, q, r, s) tuples to coefficients."""
    # Qubit register sizes
    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size

    # Setup QPU and Qubit registers
    # (except for aux_unary and range_flag as they are auxiliary Qubrick qubits)
    qpu = QPU(num_qubits=num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")

    # PREPROCESSING
    # 1) set branch to |+>
    # 2) push a state to system if one was passed in
    branch.had()
    walk_state = walk.pull_state()

    # PROCESSING
    # Since this is not the full AsymmetricQubitization since it doesn't
    # include the reflection, we must manually perform the steps from
    # AsymmetricQubitization.compute()
    #
    # NOTE: This does not include ctrl if this works but AQ.compute() fails
    # check ctrl
    oracleA = OracleA(random_seed=random_seed)

    # Set random_depth
    n = index_size
    random_depth = 2 * n

    # Run PREPARE for qubitization
    oracleA.compute(index=index, random_depth=random_depth, ctrl=(~branch))

    walk_state = walk.pull_state()
    state = walk_state[0 : 2 ** (index_size + 1) : 2] * np.sqrt(2)

    chunk = n // 4
    circuit_coefficients = {}
    for basis_state, amplitude in enumerate(state):
        p = basis_state & (2**chunk - 1)
        q = (basis_state >> chunk) & (2**chunk - 1)
        r = (basis_state >> (2 * chunk)) & (2**chunk - 1)
        s = (basis_state >> (3 * chunk)) & (2**chunk - 1)
        circuit_coefficients[(p, q, r, s)] = amplitude

    oraclea_coefficients = get_oraclea_coefficients(N, random_seed)

    for basis in circuit_coefficients:
        assert np.isclose(circuit_coefficients[basis], oraclea_coefficients[basis])


def test_oracle_a_coefficient_extraction(N=4, random_seed=5):
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk

    # Run Oracle A in isolation and get its state vector
    n = index_size
    random_depth = 2 * n

    qpu = QPU(num_qubits=n)
    index = Qubits(n, "index", qpu=qpu)
    oracle_a = OracleA(random_seed=random_seed)
    oracle_a.compute(index=index, random_depth=random_depth)
    raw_state = index.pull_state()

    # Now get coefficients via get_oraclea_coefficients
    coefficients = get_oraclea_coefficients(N, random_seed)

    # Reconstruct state vector from coefficients and compare
    reconstructed = np.zeros(2**index_size, dtype=complex)
    chunk = index_size // 4
    for (p, q, r, s), amp in coefficients.items():
        basis_idx = p | (q << chunk) | (r << 2 * chunk) | (s << 3 * chunk)
        reconstructed[basis_idx] = amp

    assert np.allclose(raw_state, reconstructed)


def test_syk_coefficients(N=4, random_seed=2):
    """Extract the coefficients from Oracle A's state vector.
    Returns a dictionary mapping (p, q, r, s) tuples to coefficients."""
    # Qubit register sizes
    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size

    # Setup QPU and Qubit registers
    # (except for aux_unary and range_flag as they are auxiliary Qubrick qubits)
    qpu = QPU(num_qubits=num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")

    # PREPROCESSING
    # 1) set branch to |+>
    # 2) push a state to system if one was passed in
    branch.had()

    # PROCESSING
    # Since this is not the full AsymmetricQubitization since it doesn't
    # include the reflection, we must manually perform the steps from
    # AsymmetricQubitization.compute()

    oracleA = OracleA(random_seed=random_seed)
    oracleB = OracleB()

    # Set random_depth
    n = index_size
    random_depth = 2 * n

    # Run PREPARE for qubitization
    oracleA.compute(index=index, random_depth=random_depth, ctrl=(~branch))
    oracleB.compute(index=index, ctrl=(branch))
    walk_state = walk.pull_state()

    even_index = walk_state[0 : 2 ** (index_size + 1) : 2] * np.sqrt(2)
    odd_index = walk_state[0 : 2 ** (index_size + 1) : 2] * np.sqrt(2)

    state = even_index + odd_index

    chunk = n // 4
    circuit_coefficients = {}
    for basis_state, amplitude in enumerate(state):
        p = basis_state & (2**chunk - 1)
        q = (basis_state >> chunk) & (2**chunk - 1)
        r = (basis_state >> (2 * chunk)) & (2**chunk - 1)
        s = (basis_state >> (3 * chunk)) & (2**chunk - 1)
        circuit_coefficients[(p, q, r, s)] = amplitude

    syk_coefficients = get_syk_coefficients(N, random_seed)

    for basis in circuit_coefficients:
        assert np.isclose(circuit_coefficients[basis], syk_coefficients[basis])
