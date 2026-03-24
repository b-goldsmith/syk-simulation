from psiqworkbench import QPU, Qubits
from syk_simulation.qubitization.asymmetric_qubitization import OracleB, OracleA, Select, AsymmetricQubitization

# from syk_simulation.qubitization.jw_utils import numpy_unitary
import numpy as np


def generate_h_circuit():
    pass


def equal_up_to_global_phase(A, B, tol=1e-9):
    # Compute the phase that best aligns B to A
    inner = np.vdot(B, A)
    phase = inner / np.abs(inner)

    # Compare A to the phase-adjusted B
    return np.allclose(A, phase * B, atol=tol, rtol=0)


def extract_system_state(full_state, n_system, n_index, n_branch=1):

    tensor = full_state.reshape(2**n_system, 2**n_index, 2**n_branch)
    system_plus_branch = tensor[:, 0, :]
    extracted_system = (system_plus_branch[:, 0] + system_plus_branch[:, 1]) / np.sqrt(2)

    return extracted_system


def project_onto_G(walk_state, N):
    index_and_branch = 4 * int(np.ceil(np.log2(N))) + 1  # index + branch
    system_space = 2**N
    result = np.zeros(system_space, dtype=complex)

    for s in range(system_space):
        base = s << index_and_branch
        branch0 = walk_state[base]
        branch1 = walk_state[base + 1]
        result[s] = (branch0 + branch1) / np.sqrt(2)
    return result


def get_oraclea_coefficients(N, random_seed):
    """Extract the coefficients from Oracle A's state vector.
    Returns a dictionary mapping (p, q, r, s) tuples to coefficients."""
    n = 4 * int(np.ceil(np.log2(N)))
    random_depth = 2 * n

    oraclea_qpu = QPU(num_qubits=n)
    index = Qubits(n, "index", qpu=oraclea_qpu)
    oracle_a = OracleA(random_seed=random_seed)
    oracle_a.compute(index=index, random_depth=random_depth)
    state = index.pull_state()
    chunk = n // 4
    coefficients = {}
    for basis_state, amplitude in enumerate(state):
        p = basis_state & (2**chunk - 1)
        q = (basis_state >> chunk) & (2**chunk - 1)
        r = (basis_state >> (2 * chunk)) & (2**chunk - 1)
        s = (basis_state >> (3 * chunk)) & (2**chunk - 1)
        coefficients[(p, q, r, s)] = amplitude
    return coefficients


def get_syk_coefficients(N, random_seed):
    """This fucntiosn returns the coefficients after PREPARE.
    These can be obtained from Oracle A coefficients * 2.
    The reason for the * 2 is because in the circuit the
    oracle call is controlled on branch in |+>. To adjust
    for this the branch = 0 portion should be multiplied by
    sqrt(2) and the branch = 1 portion should be multiplied by
    sqrt(2). Since this is just using Oracle A only, you combine
    the oracleA*sqrt(2) + Identity*sqrt(2) = oracleA * 2
    """
    syk_coefficients = get_oraclea_coefficients(N, random_seed)
    syk_coefficients = {k: v * 2 for k, v in syk_coefficients.items()}
    return syk_coefficients


def get_real_syk_coefficients(N, random_seed):
    """This fucntiosn returns the coefficients after PREPARE.
    These can be obtained from Oracle A coefficients * 2.
    The reason for the * 2 is because in the circuit the
    oracle call is controlled on branch in |+>. To adjust
    for this the branch = 0 portion should be multiplied by
    sqrt(2) and the branch = 1 portion should be multiplied by
    sqrt(2). Since this is just using Oracle A only, you combine
    the oracleA*sqrt(2) + Identity*sqrt(2) = oracleA * 2
    """
    syk_coefficients = get_oraclea_coefficients(N, random_seed)
    syk_coefficients = {k: v.real * 2 for k, v in syk_coefficients.items()}
    return syk_coefficients


def generate_walk_state_for_u(N, random_seed, init_state: np.ndarray = None):
    """This function returns the walk state as a state vector after comleting
    the steps of Asymmetric Qubitization, without the reflection."""

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
    qpu.reset(num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")

    # PREPROCESSING
    # 1) set branch to |+>
    # 2) push a state to system if one was passed in
    branch.had()
    if init_state is not None:
        system.push_state(init_state)

    # PROCESSING
    # Since this is not the full AsymmetricQubitization since it doesn't
    # include the reflection, we must manually perform the steps from
    # AsymmetricQubitization.compute()
    oracleA = OracleA(random_seed=random_seed)
    oracleB = OracleB()
    select = Select()

    # Set random_depth
    n = index_size
    random_depth = 2 * n

    # Run PREPARE for qubitization
    oracleA.compute(index=index, random_depth=random_depth, ctrl=(~branch))
    oracleB.compute(index=index, ctrl=(branch))

    # Run SELECT for qubitization
    select.compute(index=index, system=system)

    # NOT on branch
    branch.x()

    # Run UNPREPARE for qubitization
    oracleB.uncompute()
    oracleA.uncompute()

    return walk.pull_state()


def generate_walk_state_from_walk(N, random_seed, init_state: np.ndarray = None):
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
    qpu.reset(num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")

    # PREPROCESSING
    # 1) set branch to |+>
    # 2) push a state to system if one was passed in
    branch.had()
    if init_state is not None:
        system.push_state(init_state)

    AQ = AsymmetricQubitization(random_seed=random_seed)
    AQ.compute(branch=branch, index=index, system=system)
    return walk.pull_state()


def generate_u_from_circuit(N: int, random_seed: int):
    """This function builds a matrix U by running the basis
    states as the starting system state through the circuit.
    Note: The"""
    H_circuit = np.zeros((2**N, 2**N), dtype=complex)
    for basis in range(2**N):
        basis_vec = np.zeros(2**N)
        basis_vec[basis] = 1.0
        state = generate_walk_state_for_u(N, random_seed, basis_vec)
        circuit_col = state[1::512]
        H_circuit[:, basis] = circuit_col
    return H_circuit


def build_circuit_matrix(N, random_seed):
    branch_index_size = 1 + 4 * int(np.ceil(np.log2(N)))

    H_matrix = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        psi = np.zeros(2**N, dtype=complex)
        psi[i] = 1.0
        walk_state = generate_walk_state_from_walk(N, random_seed, psi)
        system_state = walk_state[1 :: 2**branch_index_size]
        H_matrix[:, i] = system_state
    return H_matrix.T


def save_circuit_matrix(N, random_seed):
    branch_index_size = 1 + 4 * int(np.ceil(np.log2(N)))

    chunk = 2**N // 2
    H_matrix = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(chunk * 0, chunk * 1):
        psi = np.zeros(2**N, dtype=complex)
        psi[i] = 1.0
        walk_state = generate_walk_state_from_walk(N, random_seed, psi)
        system_state = walk_state[1 :: 2**branch_index_size]
        H_matrix[:, i] = system_state
    np.save(f"full_walk_N_{N}-Seed_{random_seed}-chunk_1-2.npy", H_matrix)
