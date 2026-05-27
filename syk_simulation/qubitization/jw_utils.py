import numpy as np
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask, pauli_sum_to_numpy
from psiqworkbench.utils.numpy_utils import reverse_numpy_op
from syk_simulation.qubitization.utils import get_syk_coefficients, generate_walk_state_for_u


def create_unitary(N, random_seed):
    """Method of creating the matrix representing U that matches the circuit's approach of applying P, then Q,
    then R, then S. This method therefore needs to pull the individual state vectors for P, Q, R, S. Then 
    Single Majorana operators can be run for P, Q, R, then S matching the SELECT from the circuit's:

        majoranaOperator.compute(system, p, ctrl=ctrl)
        majoranaOperator.compute(system, q, ctrl=ctrl)
        majoranaOperator.compute(system, r, ctrl=ctrl)
        majoranaOperator.compute(system, s, ctrl=ctrl)
    """
    coefficients = get_syk_coefficients(N, random_seed)

    p_alpha = np.zeros(N, dtype=complex)
    q_alpha = np.zeros(N, dtype=complex)
    r_alpha = np.zeros(N, dtype=complex)
    s_alpha = np.zeros(N, dtype=complex)

    for (p, q, r, s), alpha in coefficients.items():
        p_alpha[p] += alpha
        q_alpha[q] += alpha
        r_alpha[r] += alpha
        s_alpha[s] += alpha

    P_op = sum(p_alpha[p] * single_maj_op(p, N) for p in range(N))
    Q_op = sum(q_alpha[q] * single_maj_op(q, N) for q in range(N))
    R_op = sum(r_alpha[r] * single_maj_op(r, N) for r in range(N))
    S_op = sum(s_alpha[s] * single_maj_op(s, N) for s in range(N))

    return S_op @ R_op @ Q_op @ P_op


def matrix_from_paulisum(N, random_seed):
    terms = []
    coefficients = get_syk_coefficients(N, random_seed)
    for (p, q, r, s), weight in coefficients.items():

        x_mask, z_mask = 2**p, 2**p - 1
        q_x_mask, q_z_mask = 2**q, 2**q - 1
        r_x_mask, r_z_mask = 2**r, 2**r - 1
        s_x_mask, s_z_mask = 2**s, 2**s - 1

        x_mask = x_mask ^ q_x_mask ^ r_x_mask ^ s_x_mask
        z_mask = z_mask ^ q_z_mask ^ r_z_mask ^ s_z_mask
        terms.append([weight, PauliMask(x_mask, z_mask)])

    return reverse_numpy_op(pauli_sum_to_numpy(PauliSum(*terms)))

def single_maj_op(p, N):
    """Functions used for classical generation of the Hamiltonian.
    This uses np.kron for Majorana operation adn attempts to rely on the
    matrix multiplication to handle signs. """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    operators = [I] * N

    # Apply X to the l-th qubit
    operators[p] = X

    # Apply Z to all qubits before l
    for i in range(p):
        operators[i] = Z

    # Combine using Kronecker product
    res = operators[0]
    for i in range(1, N):
        res = np.kron(res, operators[i])
    return res


def numpy_unitary(N=4, random_seed=2):
    """This method returns a numpy unitary matrix by combing each single Majorana operator for a term, 
    then adding the results of the combined Majoraona operators into a matrix."""
    L = N**4
    coefficients = get_syk_coefficients(N, random_seed)
    H_matrix = np.zeros((2**N, 2**N), dtype=complex)
    for (p, q, r, s), weight in coefficients.items():
        term_ops = single_maj_op(p, N) @ single_maj_op(q, N) @ single_maj_op(r, N) @ single_maj_op(s, N)
        H_matrix += weight * term_ops
    return reverse_numpy_op(H_matrix)


def adjusted_classical_h(N=4, random_seed=2):
    """This method multiplies the numpy unitary matrix the _lambda scalar (ratio) from running the circuit 
    in order to achieve the scaled unitary matrix used with the rotation to make up the quantum wakl."""
    H = numpy_unitary(N, random_seed)

    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0

    walk_state = generate_walk_state_for_u(N, random_seed, psi)
    branch_index = 1 + 4 * int(np.ceil(np.log2(N)))
    system_state = walk_state[1 :: 2**branch_index]
    nonzero = np.abs(system_state) > 1e-6
    H_col = H @ psi
    ratio = np.mean(np.abs(system_state[nonzero] / H_col[nonzero]))

    return H * ratio