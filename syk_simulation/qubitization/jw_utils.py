import numpy as np
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask, pauli_sum_to_numpy
from psiqworkbench.utils.numpy_utils import reverse_numpy_op
from syk_simulation.qubitization.utils import get_oraclea_coefficients, get_syk_coefficients, generate_walk_state_for_u


# Method of creating the matrix representing U that matches hte circuit's approach of applying P, then Q,
# then R, then S. This method therefore needs to pull the individual state vectors for P, Q, R, S. Then
# Single Majorana operators can be run for P, Q, R, then S matching the SELECT from the circuit's:
#
# majoranaOperator.compute(system, p, ctrl=ctrl)
# majoranaOperator.compute(system, q, ctrl=ctrl)
# majoranaOperator.compute(system, r, ctrl=ctrl)
# majoranaOperator.compute(system, s, ctrl=ctrl)

# create method to not


def create_unitary(N, random_seed):
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


##############################################################################################################
##############################################################################################################


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


##############################################################################################################
##############################################################################################################


# Functions used for classical generation of the Hamiltonian
# This uses np.kron for the majorana operation and attempts to rely on
# the matrix multiplication to handle signs
def single_maj_op(p, N):
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
    L = N**4
    coefficients = get_syk_coefficients(N, random_seed)
    H_matrix = np.zeros((2**N, 2**N), dtype=complex)
    for (p, q, r, s), weight in coefficients.items():
        term_ops = single_maj_op(p, N) @ single_maj_op(q, N) @ single_maj_op(r, N) @ single_maj_op(s, N)
        H_matrix += weight * term_ops
    return reverse_numpy_op(H_matrix)


def adjusted_classical_h(N=4, random_seed=2):
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


##############################################################################################################
# another single majorana operator attempt including trying to handle signs manually
def single_majorana_matrix(p, N):
    x_mask = 2**p
    z_mask = 0
    if p > 0:
        z_mask = 2**p - 1
    ps = PauliSum([1.0, PauliMask(x_mask, z_mask)])
    operator_matrix = reverse_numpy_op(pauli_sum_to_numpy(ps))
    current_size = operator_matrix.shape[0]
    if current_size == 2**N:
        return operator_matrix
    identity_padding = np.eye(2**N // current_size)
    return np.kron(identity_padding, operator_matrix)


def jw_sign(p, q, r, s):
    pairs = [(p, q), (p, r), (q, r), (p, s), (q, s), (r, s)]
    return (-1) ** sum(1 for (a, b) in pairs if a < b)


def get_classical_matrix(N=4, random_seed=2):
    coefficients = get_oraclea_coefficients(N, random_seed)

    L = N**4
    terms = np.zeros((2**N, 2**N), dtype=complex)

    for (p, q, r, s), alpha in coefficients.items():
        if len({p, q, r, s}) < 4:
            continue
        gp = single_majorana_matrix(p, N)
        gq = single_majorana_matrix(q, N)
        gr = single_majorana_matrix(r, N)
        gs = single_majorana_matrix(s, N)
        alpha = alpha * jw_sign(p, q, r, s)
        terms += (alpha / np.sqrt(L)) * (gs @ gr @ gq @ gp)

    return terms


##############################################################################################################
# another attempt using PauliSums and attempting to determine the sign
def multiply_two_maj_ops(x1, z1, x2, z2):
    zx_count = bin(z1 & x2).count("1")
    xz_count = bin(x1 & z2).count("1")
    phase = (1j**xz_count) * ((-1j) ** zx_count)
    return x1 ^ x2, z1 ^ z2, phase


def classical_numpy_unitary(N, random_seed, inti_state: np.ndarray = None):
    L = 4**N
    coefficients = get_oraclea_coefficients(N, random_seed)
    pauli_mask_tuple = ()
    beta = 1.0 / np.sqrt(L) * np.sqrt(2)  # sqrt(2) is for the branch Hadamard
    for (p, q, r, s), alpha in coefficients.items():
        p_x_mask = 1 << p
        p_z_mask = (1 << p) - 1

        q_x_mask = 1 << q
        q_z_mask = (1 << q) - 1

        total_sign = 1

        x_mask, z_mask, sign = multiply_two_maj_ops(p_x_mask, p_z_mask, q_x_mask, q_z_mask)
        total_sign *= sign

        r_x_mask = 1 << r
        r_z_mask = (1 << r) - 1
        x_mask, z_mask, sign = multiply_two_maj_ops(x_mask, z_mask, r_x_mask, r_z_mask)
        total_sign *= sign

        s_x_mask = 1 << s
        s_z_mask = (1 << s) - 1
        x_mask, z_mask, sign = multiply_two_maj_ops(x_mask, z_mask, s_x_mask, s_z_mask)
        total_sign *= sign

        y_count = bin(x_mask & z_mask).count("1")
        pauli_mask_correction = (1j) ** y_count
        total_sign *= pauli_mask_correction

        assert np.abs(alpha.imag) < 1e-6, "Oracle A's coefficients are complex."
        pauli_mask_tuple += ([complex(total_sign * alpha * beta), PauliMask(x_mask, z_mask)],)

    psum = PauliSum(*pauli_mask_tuple)

    return reverse_numpy_op(pauli_sum_to_numpy(psum))


def pauli_sum_without_sign(N, random_seed):
    L = 4**N
    coefficients = get_oraclea_coefficients(N, random_seed)
    pauli_mask_tuple = ()
    beta = 1.0 / np.sqrt(L) * np.sqrt(2)  # sqrt(2) is for the branch Hadamard
    for (p, q, r, s), alpha in coefficients.items():
        p_x_mask = 1 << p
        p_z_mask = (1 << p) - 1

        q_x_mask = 1 << q
        q_z_mask = (1 << q) - 1

        total_sign = 1

        x_mask, z_mask, sign = multiply_two_maj_ops(p_x_mask, p_z_mask, q_x_mask, q_z_mask)
        total_sign *= sign

        r_x_mask = 1 << r
        r_z_mask = (1 << r) - 1
        x_mask, z_mask, sign = multiply_two_maj_ops(x_mask, z_mask, r_x_mask, r_z_mask)
        total_sign *= sign

        s_x_mask = 1 << s
        s_z_mask = (1 << s) - 1
        x_mask, z_mask, sign = multiply_two_maj_ops(x_mask, z_mask, s_x_mask, s_z_mask)
        total_sign *= sign

        y_count = bin(x_mask & z_mask).count("1")
        pauli_mask_correction = (1j) ** y_count
        total_sign *= pauli_mask_correction

        x_mask = p_x_mask ^ q_x_mask ^ r_x_mask ^ s_x_mask
        z_mask = p_z_mask ^ q_z_mask ^ r_z_mask ^ s_z_mask

        assert np.abs(alpha.imag) < 1e-6, "Oracle A's coefficients are complex."
        pauli_mask_tuple += ([complex(alpha * beta), PauliMask(x_mask, z_mask)],)

    psum = PauliSum(*pauli_mask_tuple)

    return reverse_numpy_op(pauli_sum_to_numpy(psum))


# This implementation generates the PauliSum which can be used with pauli_sum_to_numpy()
# This tries to handle the signs manually


def pauli_multiply(x1, z1, x2, z2):
    phase = 1.0 + 0j

    n_bits = max(x1, z1, x2, z2, 1).bit_length()
    for bit in range(n_bits):
        a = ((x1 >> bit) & 1, (z1 >> bit) & 1)
        b = ((x2 >> bit) & 1, (z2 >> bit) & 1)
        # (x,z): 00=I, 10=X, 01=Z, 11=Y
        if a == (1, 0) and b == (0, 1):
            phase *= -1j  # X*Z = -iY
        elif a == (0, 1) and b == (1, 0):
            phase *= 1j  # Z*X = iY
        elif a == (1, 0) and b == (1, 1):
            phase *= 1j  # X*Y = iZ
        elif a == (1, 1) and b == (1, 0):
            phase *= -1j  # Y*X = -iZ
        elif a == (0, 1) and b == (1, 1):
            phase *= -1j  # Z*Y = -iX
        elif a == (1, 1) and b == (0, 1):
            phase *= 1j  # Y*Z = iX

    return phase, x1 ^ x2, z1 ^ z2


def four_majorana_mask(p, q, r, s, N):
    """Phase and PauliMask for γ_p γ_q γ_r γ_s."""
    total_phase = 1.0 + 0j
    total_x, total_z = 0, 0
    for k in [p, q, r, s]:
        x_k = 2**k
        z_k = 2**k - 1
        ph, total_x, total_z = pauli_multiply(total_x, total_z, x_k, z_k)
        total_phase *= ph
    return total_phase, PauliMask(total_x, total_z)


def build_pauli_sum_from_oraclea(N, random_seed):
    coefficients = get_oraclea_coefficients(N, random_seed)

    L = N**4
    terms = []

    for (p, q, r, s), alpha in coefficients.items():
        coeff = alpha / np.sqrt(N**4)
        phase, mask = four_majorana_mask(p, q, r, s, N)
        terms.append([coeff * phase, mask])

    return reverse_numpy_op(pauli_sum_to_numpy(PauliSum(*terms)))


# The following are utilities for generating the gamma matrix and checkign the sign
# taken from Qubits pull_state() documentation
def correct_phase(state, expected_state):
    inner_product = np.vdot(expected_state, state)  # np.vdot does complex conjugate of first arg
    global_phase = np.angle(inner_product)
    phase_correction = np.exp(-1j * global_phase)
    return state * phase_correction


def gamma_matrix(l, N):
    return reverse_numpy_op(
        pauli_sum_to_numpy(PauliSum([1.0, PauliMask(1 << l, (1 << l) - 1)], [0.0, PauliMask(1 << (N - 1), 0)]))
    )
