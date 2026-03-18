from syk_simulation.qubitization.asymmetric_qubitization import OracleB, OracleA, Select, AsymmetricQubitization
from psiqworkbench import QPU, Qubits
import numpy as np


# extract the coefficients only from Oracle A
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


# These coefficients are from after PREPARE. These are the
# weights used for the controls of SELECT
def get_syk_coefficients(N, random_seed):
    # These can be obtained from Oracle A coefficients * 2
    # The reason for the * 2 is because in the circuit the
    # oracle call is controlled on branch in |+>. To adjust
    # for this the branch = 0 portion should be multiplied by
    # sqrt(2) and the branch = 1 portion should be multiplied by
    # sqrt(2). Since this is just using Oracle A only, you combine
    # the oracleA*sqrt(2) + Identity*sqrt(2) = oracleA * 2
    syk_coefficients = get_oraclea_coefficients(N, random_seed)
    syk_coefficients = {k: v * 2 for k, v in syk_coefficients.items()}
    return syk_coefficients


def get_prepare_coefficients(N, random_seed):
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
    #
    # NOTE: This does not include ctrl if this works but AQ.compute() fails
    # check ctrl
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
    coefficients = {}
    for basis_state, amplitude in enumerate(state):
        p = basis_state & (2**chunk - 1)
        q = (basis_state >> chunk) & (2**chunk - 1)
        r = (basis_state >> (2 * chunk)) & (2**chunk - 1)
        s = (basis_state >> (3 * chunk)) & (2**chunk - 1)
        coefficients[(p, q, r, s)] = amplitude

    return coefficients


# This method goes through each basis state of the system and performs a full quantum walk.
# The resulting state vectors are then combined to produce a matrix H
def generate_h_circuit(N: int, eigenvector: np.ndarray, random_seed: int):
    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size)))
    total_aux_size = aux_size + 1 + 1  # aux_index + range_flag + branch
    total_qubits = system_size + index_size + total_aux_size

    qpu = QPU(num_qubits=total_qubits)

    H_circuit = np.zeros((2**N, 2**N), dtype=complex)

    for basis in range(2**N):
        # fresh QPU per basis state
        qpu.reset(total_qubits)
        quantum_walk = Qubits(system_size + index_size + 1, "walk", qpu=qpu)
        branch = Qubits(quantum_walk[0], "branch")
        system = Qubits(quantum_walk[1 : system_size + 1], "system")
        index = Qubits(quantum_walk[system_size + 1 :], "index")
        branch.had()
        basis_vec = np.zeros(2**N)
        basis_vec[basis] = 1.0
        system.push_state(basis_vec)
        init = quantum_walk.pull_state()

        # Check projection before walk
        T_0 = project_onto_G(init, N)
        if not np.allclose(T_0, basis_vec, atol=1e-10):
            print(
                f"basis {format(basis, f'0{N}b')}: project_onto_G wrong BEFORE walk, "
                f"max diff {np.max(np.abs(T_0 - basis_vec)):.2e}"
            )

        AS = AsymmetricQubitization(random_seed=random_seed)
        AS.compute(branch=branch, system=system, index=index)
        H_circuit[:, basis] = project_onto_G(quantum_walk.pull_state(), N)

    np.save(f"h_circuit_N{N}_seed{random_seed}.npy", H_circuit)
    return H_circuit


# Only the branch is used in the projection because the index register is back to |0>
def project_onto_G(walk_state, N):
    system_space = 2**N
    result = np.zeros(system_space, dtype=complex)
    for s in range(system_space):
        branch0 = walk_state[(s << 1)]
        branch1 = walk_state[(s << 1) + 1]
        result[s] = (branch0 + branch1) / np.sqrt(2)
    return result


# walks through the same beahvior as Asymmetric Qubitization without the
# reflection at the end
def generate_walk_for_u(N, random_seed, init_state: np.ndarray = None):

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
    if init_state is not None:
        system.push_state(init_state)

    # PROCESSING
    # Since this is not the full AsymmetricQubitization since it doesn't
    # include the reflection, we must manually perform the steps from
    # AsymmetricQubitization.compute()
    #
    # NOTE: This does not include ctrl if this works but AQ.compute() fails
    # check ctrl
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


def equal_up_to_global_phase(A, B, tol=1e-9):
    # Compute the phase that best aligns B to A
    inner = np.vdot(B, A)
    phase = inner / np.abs(inner)

    # Compare A to the phase-adjusted B
    return np.allclose(A, phase * B, atol=tol, rtol=0)
