from syk_simulation.qubitization.asymmetric_qubitization import AsymmetricQubitization
from psiqworkbench import QPU, Qubits
import numpy as np
from workbench_algorithms.utils.paulimask import pauli_sum_to_numpy
from psiqworkbench.utils.numpy_utils import reverse_numpy_op
import os
from syk_simulation.qubitization.utils import get_oraclea_coefficients, project_onto_G
from syk_simulation.qubitization.jw_utils import single_majorana_matrix, build_pauli_sum_from_oraclea


def man_test_two_walks():
    N = 4
    random_seed = 5
    np.random.seed(random_seed)
    filename = f"h_circuit_N{N}_seed{random_seed}.npy"
    if os.path.exists(filename):
        H_circuit = np.load(filename)
    else:
        H_circuit = np.load(N, random_seed)

    eigenvalues, eigenvectors = np.linalg.eig(H_circuit)
    most_real_idx = np.argmin(np.abs(np.imag(eigenvalues)))
    evec = np.real(eigenvectors[:, most_real_idx])
    evec /= np.linalg.norm(evec)

    # psum = build_pauli_sum_from_oraclea(N, random_seed)
    # psum_numpy = reverse_numpy_op(pauli_sum_to_numpy(psum))
    # eigs = np.linalg.eig(psum_numpy)
    # evegsc, eigval = eigs[1][:, 0], eigs[0][0]

    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size)))
    total_aux_size = aux_size + 1 + 1  # aux_index + range_flag + branch
    total_qubits = system_size + index_size + total_aux_size
    qpu = QPU(num_qubits=total_qubits)

    quantum_walk = Qubits(system_size + index_size + 1, "walk", qpu=qpu)  # 1 is the branch qubit

    branch = Qubits(quantum_walk[0], "branch")
    system = Qubits(quantum_walk[1 : system_size + 1], "system")
    index = Qubits(quantum_walk[system_size + 1 :], "index")
    branch.had()

    # rng = np.random.default_rng(42)
    # some_state = rng.standard_normal(2**N)
    # some_state /= np.linalg.norm(some_state)

    some_state = evec

    system.push_state(some_state)
    init_state = quantum_walk.pull_state()

    AS = AsymmetricQubitization(random_seed=random_seed)
    AS.compute(branch=branch, index=index, system=system)

    after_one_w = quantum_walk.pull_state()

    AS.compute(branch=branch, index=index, system=system)
    after_two_w = quantum_walk.pull_state()

    T_0 = project_onto_G(init_state, N)
    T_1 = project_onto_G(after_one_w, N)
    T_2 = project_onto_G(after_two_w, N)

    nonzero = np.abs(T_0) > 1e-10

    T2_val = np.mean(np.real(T_2[nonzero] / T_0[nonzero]))
    h_over_lambda = np.sqrt((T2_val + 1) / 2)

    print(f"T2(h/λ) from circuit: {T2_val:.6f}")
    print(f"h/λ from circuit: ±{h_over_lambda:.6f}")
    print(f"2*(h/λ)²-1 (should match T2(h/λ)): {2*h_over_lambda**2 - 1}")

    # Now verify T_1 is consistent: v1/v0 should all equal h/λ (or -h/λ)
    # for an eigenstate — but for random state it won't be uniform.
    # Instead verify the Chebyshev recurrence holds:
    # T_2 = 2*(h/λ)*T_1 - T_0
    # => v2 = 2*(h/λ)*v1 - v0
    residual = T_2[nonzero] - (2 * h_over_lambda * T_1[nonzero] - T_0[nonzero])
    print(f"Recurrence residual max: {np.max(np.abs(residual)):.2e}")

    # Also check with negative root
    residual_neg = T_2[nonzero] - (2 * (-h_over_lambda) * T_1[nonzero] - T_0[nonzero])
    print(f"Recurrence residual max (negative root): {np.max(np.abs(residual_neg)):.2e}")

    coefficients = get_oraclea_coefficients(N, random_seed)

    L = N**4
    terms = np.zeros((2**N, 2**N), dtype=complex)

    for (p, q, r, s), alpha in coefficients.items():
        gp = single_majorana_matrix(p, N)
        gq = single_majorana_matrix(q, N)
        gr = single_majorana_matrix(r, N)
        gs = single_majorana_matrix(s, N)
        terms += (alpha / np.sqrt(L)) * (gp @ gq @ gr @ gs)

    # terms = H_circuit

    # Apply H/lambda classically to some_state
    # H_applied = terms @ evec
    H_applied = terms @ some_state
    # H_circuit_applied = H_circuit @ some_state

    # These should match: T_1 = (H/lambda) @ some_state
    print("Classical H/λ|ψ⟩ vs circuit T_1:")
    print(f"Max diff: {np.max(np.abs(H_applied - T_1)):.2e}")
    # print("Classical H_circuit H/λ|ψ⟩ vs classical:")
    # print(f"Max diff: {np.max(np.abs(H_applied - H_circuit_applied)):.2e}")

    # Also check norms
    print(f"||H/λ|ψ⟩|| = {np.linalg.norm(H_applied):.6f}")
    print(f"||T_1||     = {np.linalg.norm(T_1):.6f}")

    # Check if T_1 and H_applied are proportional
    nonzero = np.abs(H_applied) > 1e-10
    ratios = T_1[nonzero] / H_applied[nonzero]
    print("T_1 / (H/λ|ψ⟩) ratios (should all be equal if proportional):")
    # print(np.round(ratios, 6))
    print(f"Are they proportional: {np.allclose(ratios, ratios[0], atol=1e-4)}")
    print(f"Proportionality constant: {np.mean(ratios):.6f}")
    print(f"Expected from norms: {np.linalg.norm(T_1)/np.linalg.norm(H_applied):.6f}")
    # Before applying W, project initial state - should recover some_state
    T_0_check = project_onto_G(init_state, N)
    # print(f"T_0 vs some_state max diff: {np.max(np.abs(T_0_check - evec)):.2e}")
    print(f"T_0 vs some_state max diff: {np.max(np.abs(T_0_check - some_state)):.2e}")
    print(f"||T_0|| = {np.linalg.norm(T_0_check):.6f}")
    # print(f"||evec|| = {np.linalg.norm(evec):.6f}")
    print(f"||some_state|| = {np.linalg.norm(some_state):.6f}")
    print(f"Max diff H_circuit vs H_classical: {np.max(np.abs(H_circuit - terms)):.2e}")
    print(f"Max diff H_circuit vs -H_classical: {np.max(np.abs(H_circuit - (-terms))):.2e}")
    print(f"Are they equal: {np.allclose(H_circuit, terms, atol=1e-6)}")

    print(f"h/λ from circuit: ±{h_over_lambda:.6f}")
    print(f"T2(h/λ): {T2_val:.6f}  (self-check: {2*h_over_lambda**2 - 1:.6f})")
    print(f"||H/λ|ψ>|| = {np.linalg.norm(terms @ some_state):.6f}, ||T_1|| = {np.linalg.norm(T_1):.6f}")

    # OPEN ISSUE: H_circuit and terms disagree by a sign pattern on certain
    # basis states (e.g. |0000> and |1111>). Root cause not yet identified —
    # likely a sign convention in PREPARE/UNPREPARE interaction with boundary
    # states. Does not affect T2 uniformity or norm correctness.
    print(f"[OPEN] Max diff H_circuit vs terms: {np.max(np.abs(H_circuit - terms)):.2e}")

    assert False


def man_test_walk_eigs():
    N = 4
    random_seed = 10
    psum = build_pauli_sum_from_oraclea(N, random_seed)
    psum_numpy = reverse_numpy_op(pauli_sum_to_numpy(psum))
    eigs = np.linalg.eig(psum_numpy)
    h_over_lambda = eigs[0][0]  # this is already h/λ, no division needed - eigenvalue

    expected_phase = np.exp(1j * np.arccos(h_over_lambda))
    print(f"h/λ = {h_over_lambda}")
    print(f"Expected phase: {expected_phase}")

    # print(psum_numpy)
    # # eigs = np.linalg.eigh(psum_numpy)
    evec, eigval = eigs[1][:, 0], eigs[0][0]
    # # print(f"first evec: {evec}")
    # print(f"first eigval (h): {eigval}")

    # Verify evec is a true eigenvector of the classical H/lambda
    Hv = psum_numpy @ evec
    nonzero = np.abs(evec) > 1e-10
    ratios = Hv[nonzero] / evec[nonzero]
    print("Classical check - H/λ @ evec / evec (should all be h/λ):")
    print(ratios)
    print(f"Is evec a true eigenvector: {np.allclose(Hv, h_over_lambda * evec, atol=1e-10)}")
    print(f"Is psum_numpy Hermitian: {np.allclose(psum_numpy, psum_numpy.conj().T, atol=1e-10)}")
    print(f"psum_numpy shape: {psum_numpy.shape}")
    print(f"Eigenvalues: {np.linalg.eig(psum_numpy)[0]}")
    print(f"evec norm: {np.linalg.norm(evec)}")

    # Check if psum_numpy is actually symmetric (real)
    print(f"Is psum_numpy symmetric (real): {np.allclose(psum_numpy, psum_numpy.T, atol=1e-10)}")

    # Print the actual matrix
    print("psum_numpy diagonal:")
    print(np.diag(psum_numpy))

    # Most importantly - print the full matrix to see its structure
    np.set_printoptions(precision=3, suppress=True)
    print("psum_numpy:")
    print(psum_numpy)

    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size)))
    total_aux_size = aux_size + 1 + 1  # aux_index + range_flag + branch
    total_qubits = system_size + index_size + total_aux_size
    qpu = QPU(num_qubits=total_qubits)

    quantum_walk = Qubits(system_size + index_size + 1, "walk", qpu=qpu)  # 1 is the branch qubit

    branch = Qubits(quantum_walk[0], "branch")
    system = Qubits(quantum_walk[1 : system_size + 1], "system")
    index = Qubits(quantum_walk[system_size + 1 :], "index")
    print(f"sizes- total:{len(quantum_walk)} -- branch:{len(branch)} - system:{len(system)} - index:{len(index)}")
    branch.had()

    og_walk = quantum_walk.pull_state()

    indices = np.nonzero(og_walk)[0]
    test_array = [(int(idx), val) for idx, val in zip(indices, og_walk[indices])]
    print(f"Original:\n{test_array}")

    system.push_state(evec)

    push_walk = quantum_walk.pull_state()

    indices = np.nonzero(push_walk)[0]
    test_array = [(int(idx), val) for idx, val in zip(indices, push_walk[indices])]
    print(f"After push:\n{test_array}")
    print(push_walk[::512])

    # oracleA = OracleA(random_seed=random_seed)
    # oracleB = OracleB()
    # random_depth = 2 * index_size

    # # Test PREPARE → UNPREPARE only, no SELECT
    # oracleA.compute(index=index, random_depth=random_depth, ctrl=(~branch))
    # oracleB.compute(index=index, ctrl=(branch))
    # # NO select
    # oracleB.uncompute()
    # oracleA.uncompute()

    # walk_state_no_select = quantum_walk.pull_state()
    # branch0_index0 = walk_state_no_select[0:32:2]
    # branch1_index0 = walk_state_no_select[1:32:2]
    # projected = (branch0_index0 + branch1_index0) / np.sqrt(2)

    # initial_system = (push_walk[0:32:2] + push_walk[1:32:2]) / np.sqrt(2)
    # nonzero = np.abs(initial_system) > 1e-10
    # print("PREPARE→UNPREPARE only ratios (should all be 1.0):")
    # print(projected[nonzero] / initial_system[nonzero])

    # assert False

    AS = AsymmetricQubitization(random_seed=random_seed)
    AS.compute(branch=branch, index=index, system=system)

    walk_state_after_U = quantum_walk.pull_state()

    # For each system state s (0..15), index=0, branch=0 and branch=1:
    projected_system = np.zeros(2**N, dtype=complex)
    for s in range(2**N):
        pos_b0 = s << 1  # branch=0, system=s, index=0
        pos_b1 = (s << 1) + 1  # branch=1, system=s, index=0
        projected_system[s] = (walk_state_after_U[pos_b0] + walk_state_after_U[pos_b1]) / np.sqrt(2)

    initial_system = np.zeros(2**N, dtype=complex)
    for s in range(2**N):
        pos_b0 = s << 1
        pos_b1 = (s << 1) + 1
        initial_system[s] = (push_walk[pos_b0] + push_walk[pos_b1]) / np.sqrt(2)

    nonzero = np.abs(initial_system) > 1e-10
    print(f"Nonzero components: {np.sum(nonzero)}")
    print("Ratios -- should all be h/λ = -0.115:")
    print(projected_system[nonzero] / initial_system[nonzero])

    # # Comment out reflect, then after UNPREPARE:
    # walk_state_after_U = quantum_walk.pull_state()
    # branch0 = walk_state_after_U[::2]  # branch=0
    # branch1 = walk_state_after_U[1::2]  # branch=1
    # projected = (branch0 + branch1) / np.sqrt(2)

    # # Compare to h/λ * evec * (1/sqrt(2)) since branch is in |+>
    # nonzero = np.abs(evec) > 1e-10
    # print("Ratios projected / (evec/sqrt(2)) -- should all be h/λ = -0.115:")
    # print(projected[nonzero] / (evec[nonzero] / np.sqrt(2)))

    # final_walk = quantum_walk.pull_state()
    # test_array = [(int(idx), val) for idx, val in zip(indices, final_walk[indices])]
    # print(f"Final:\n{test_array}")

    # # Project final state onto |G> = |+> on branch
    # branch0_final = final_walk[::2]
    # branch1_final = final_walk[1::2]
    # projected_final = (branch0_final + branch1_final) / np.sqrt(2)

    # # Project initial state onto |G> = |+> on branch
    # branch0_init = push_walk[::2]
    # branch1_init = push_walk[1::2]
    # projected_init = (branch0_init + branch1_init) / np.sqrt(2)

    # # Ratio should be h/λ for eigenvector components
    # nonzero = np.abs(projected_init) > 1e-10
    # ratios = projected_final[nonzero] / projected_init[nonzero]
    # print(f"Ratios (should all be h/λ = {h_over_lambda:.6f}):")
    # print(ratios)
    # print(f"Mean ratio: {np.mean(ratios):.6f}")

    # h = eigval
    # lambda_val = sum(abs(amp) for amp in get_oraclea_coefficients(N, random_seed).values())
    # expected_phase = np.exp(1j * np.arccos(h / lambda_val))
    # print(f"Expected vdot: {expected_phase}")
    # print(f"Actual  vdot: {np.vdot(push_walk, final_walk)}")
    # print(f"Conjugate match: {np.allclose(np.vdot(push_walk, final_walk), np.conj(expected_phase), atol=1e-3)}")
    # print(f"Direct match:    {np.allclose(np.vdot(push_walk, final_walk), expected_phase, atol=1e-3)}")
    # print(f"norm after push and final: {np.linalg.norm(final_walk)}")
    # print(f"shape after push and final: {final_walk.shape}")

    # print(f"vdot after push and final: {np.vdot(push_walk, final_walk)}")

    assert False
