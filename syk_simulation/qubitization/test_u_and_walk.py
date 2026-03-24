import numpy as np
from syk_simulation.qubitization.utils import (
    generate_walk_state_for_u,
)
from syk_simulation.qubitization.jw_utils import (
    numpy_unitary,
    correct_phase,
    adjusted_classical_h,
)
import pytest


@pytest.mark.skip(reason="Each run takes over 30 seconds")
@pytest.mark.parametrize("random_seed", [1, 2])
def manual_test_unitary_n8(random_seed):
    """This function tests the unitary matrix that results from AQ without
    the reflection. This is testing with N=8. Due to the size and time it takes,
    this test is labeled as manual."""
    N = 8
    H = adjusted_classical_h(N, random_seed)

    imag_sign_h = np.sign(H[0, 0].imag)

    # The simulator requires that the first non-zero amplitude must have positive real components. This means the first column of
    # the unitary matrix is all postive. Therefore, adjust the mathematically calculated H so that all elements of the first column
    # are postive. Note if the imaginary sign is 1.0 or -1.0, if the first column's element was 0 or negative, change the sign of
    # the entire row.
    if imag_sign_h == 0:
        mask = H[:, 0] < 0
    else:
        mask = H[:, 0] <= 0
    H[mask] *= -1

    filename = f"projects/syk-simulation/N_8-Seed_{random_seed}-chunk_1-2.npy"
    filename2 = f"projects/syk-simulation/N_8-Seed_{random_seed}-chunk_2-2.npy"
    circuit_matrix = np.zeros((2**N, 2**N), dtype=complex)
    circuit_matrix += np.load(filename)
    circuit_matrix += np.load(filename2)
    assert np.allclose(H, circuit_matrix.T)


@pytest.mark.parametrize("random_seed", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_unitary_n4(random_seed):
    """This function tests the unitary matrix that results from AQ without
    the reflection. This is testing with N=4 and different random seeds."""
    N = 4
    H = numpy_unitary(N, random_seed)

    real_sign_h = np.sign(H[0, 0].real)
    imag_sign_h = np.sign(H[0, 0].imag)

    # handle phases
    # simulator requires first non-zero amplitude must have positive real components. This means |0> is all positive.
    # Therefore, adjust mathematically calculated H so that all |0> is postive
    if imag_sign_h == 0:
        mask = H[:, 0] < 0
    else:
        mask = H[:, 0] <= 0
    H[mask] *= -1

    branch_index_size = 1 + 4 * int(np.ceil(np.log2(N)))

    ratio = 0
    H_matrix = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(2**N):
        psi = np.zeros(2**N, dtype=complex)
        psi[i] = 1.0

        walk_state = generate_walk_state_for_u(N, random_seed, psi)

        system_state = walk_state[1 :: 2**branch_index_size]
        nonzero = np.abs(system_state) > 1e-6
        H_col = H @ psi
        ratio = np.mean(np.abs(H_col[nonzero] / system_state[nonzero]))
        H_matrix[:, i] = system_state

    H_matrix = H_matrix.T

    for idx in range(len(H_matrix)):
        print(H_matrix[idx].real.tolist())
    print("space circuit above")
    for idx in range(len(H_matrix)):
        print(H[idx].real.tolist())
    print("H above")
    print(f"seed:{random_seed} {real_sign_h} - {imag_sign_h} - {H[0, 0]} - ratio: {ratio}")
    assert np.allclose(H, H_matrix * ratio)

    eigenvalues, eigenvectors = np.linalg.eig(H)
    psi = eigenvectors[:, 0]

    # psi = np.zeros(2**N, dtype=complex)
    # psi[4] = 1.0

    H_matrix *= ratio
    walk_state = generate_walk_state_for_u(N, random_seed, psi)
    system_state = walk_state[1 :: 2**branch_index_size]

    right_side = eigenvalues[0] * psi
    print(right_side.real.tolist())
    print(system_state.real.tolist())

    # Test an eigenvector
    np.allclose(H_matrix @ psi, H @ psi)
    np.allclose(system_state, H_matrix @ psi)
    np.allclose(system_state, H @ psi)


# similar test to test_full_reconstruction but using `correct_phase`
def man_test_u_block_encoding():
    # TODO once this is working set N and random_seed to be parameters
    N = 4
    random_seed = 5
    init_state = None

    H = adjusted_classical_h(N, random_seed)

    imag_sign_h = np.sign(H[0, 0].imag)

    # The simulator requires that the first non-zero amplitude must have positive real components. This means the first column of
    # the unitary matrix is all postive. Therefore, adjust the mathematically calculated H so that all elements of the first column
    # are postive. Note if the imaginary sign is 1.0 or -1.0, if the first column's element was 0 or negative, change the sign of
    # the entire row.
    if imag_sign_h == 0:
        mask = H[:, 0] < 0
    else:
        mask = H[:, 0] <= 0
    H[mask] *= -1

    # H_matrix = classical_numpy_unitary(N, random_seed, init_state)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    most_real_idx = np.argmin(np.abs(np.imag(eigenvalues)))
    evec = np.real(eigenvectors[:, most_real_idx])
    evec /= np.linalg.norm(evec)
    init_state = evec

    walk_state = generate_walk_state_for_u(N, random_seed, init_state)

    # project onto |B_dagger>
    branch_index_size = 1 + 4 * np.log2(N)  # branch_size + index_size

    branch_1_index_0_amplitudes = walk_state[1 :: int(2 ** (branch_index_size))]

    norm_branch_1_index_0_amplitudes = np.linalg.norm(branch_1_index_0_amplitudes)
    conditional_state_branch_1_index_0 = branch_1_index_0_amplitudes / norm_branch_1_index_0_amplitudes

    if init_state is None:
        init_state = np.zeros(2**N)
        init_state[0] = 1.0

    # comparisons and asserts check H/lambda from classical against circuit
    expected = H @ init_state
    circuit_col = walk_state[1 :: int(2 ** (branch_index_size))]
    print("circuit col:", np.round(circuit_col, 4))
    print("classical col:", np.round(expected, 4))
    print("ratio:", np.round(circuit_col / expected, 4))

    print(f"norm expected sqrd: {np.linalg.norm(expected) ** 2}")

    print(f"2*norm branch 1 index 0 sqrd: {(np.linalg.norm(branch_1_index_0_amplitudes)**2)*2}")

    expected_normalized = expected / np.linalg.norm(expected)
    corrected_state_branch_1_index_0 = correct_phase(conditional_state_branch_1_index_0, expected_normalized)

    for i in range(10):
        print(
            f"cond: {conditional_state_branch_1_index_0[i]} - corrected: {corrected_state_branch_1_index_0[i]} exp: {expected_normalized[i]}"
        )

    assert np.allclose((np.linalg.norm(branch_1_index_0_amplitudes) ** 2) * 2, np.linalg.norm(expected) ** 2, atol=1e-6)

    assert np.allclose(
        corrected_state_branch_1_index_0, expected_normalized
    ), "Conditional state branch 1 index 0 does not match H_matrix|init_state>"
