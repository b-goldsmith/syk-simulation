from syk_simulation.qubitization.qsp import qsp_evolution, QSP, get_qsp_phases
from syk_simulation.qubitization.utils import get_oraclea_coefficients, get_syk_coefficients, generate_walk_state_for_u
from syk_simulation.qubitization.jw_utils import numpy_unitary
from syk_simulation.qubitization.asymmetric_qubitization import AsymmetricQubitization
from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask
import numpy as np
import scipy


def test_qsp():

    random_seed = 3
    N = 8
    time = 1 
    J = 1
    epsilon = 1e-3
    random_seed=3
    extra_qubits = 0

    H = numpy_unitary(N, random_seed)
    H = H/96  # 96 comes from 4 * 4!

    real_sign_h = np.sign(H[0, 0].real)
    imag_sign_h = np.sign(H[0, 0].imag)

    # handle global phase from separate simulator runs
    if imag_sign_h == 0:
        mask = H[:, 0] < 0
    else:
        mask = H[:, 0] <= 0
    H[mask] *= -1

    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0

    walk_state = generate_walk_state_for_u(N, random_seed, psi)
    branch_index = 1 + 4 * int(np.ceil(np.log2(N)))
    system_state = walk_state[1 :: 2**branch_index]
    nonzero = np.abs(system_state) > 1e-6
    H_col = H @ psi
    lambda_ = np.mean(np.abs(system_state[nonzero] / H_col[nonzero]))

    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1
    mode_size = 1
    selection_size = 1
    control_size = 1
    extra_qubits = 0

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size + mode_size + selection_size + control_size + extra_qubits

    print(f"num_qubits: {num_qubits} - without extra: {num_qubits - extra_qubits}")
    qpu = QPU(num_qubits=num_qubits)
    qpu.reset(num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")
    mode = Qubits(mode_size, "mode", qpu=qpu)
    selection = Qubits(selection_size, "selection", qpu=qpu)

    branch.had()
    qsp_evolution(
        N=N,
        J=J,
        branch=branch,
        index=index,
        system=system,
        mode=mode,
        selection=selection,
        time=time,
        epsilon=epsilon,
        random_seed=random_seed,
        lambda_=lambda_,
    )
    branch.had()

    # postselection for QSP
    selection.postselect(0)
    system_state = system.pull_state()


    psi_ground_truth = scipy.linalg.expm(1j*H) @ psi

    psi_ground_truth /= np.linalg.norm(psi_ground_truth)
    system_state /= np.linalg.norm(system_state)
    fidelity = np.abs(np.vdot(psi_ground_truth, system_state))**2

    assert fidelity >= 1-epsilon

def test_r_ratio():
    N = 8 
    random_seed=25

    H = numpy_unitary(N, random_seed)
    H = H/96  # 96 comes from 4 * 4!

    real_sign_h = np.sign(H[0, 0].real)
    imag_sign_h = np.sign(H[0, 0].imag)

    # handle global phase from separate simulator runs
    if imag_sign_h == 0:
        mask = H[:, 0] < 0
    else:
        mask = H[:, 0] <= 0
    H[mask] *= -1

    # The Physics Validation
    energies = np.linalg.eigvalsh(H)
    spacings = np.diff(energies)

    s_i = spacings[:-1]
    s_next = spacings[1:]
    
    # 4. Compute ratios with stability handling
    with np.errstate(divide='ignore', invalid='ignore'):
        r_i = np.minimum(s_i, s_next) / np.maximum(s_i, s_next)
        r_n = r_i[np.isfinite(r_i)]

    print(f"Mean r-ratio: {np.mean(r_n)}")
    assert False

    # Table 1 of https://arxiv.org/pdf/1212.5611 
    # should be either 0.53590 (GOE), 0.60266 (GUE), or 0.67617 (GSE)
    # per https://arxiv.org/pdf/1602.06964 (2nd page end of top paragraph)
    # assert 0.53950 <= np.mean(r_n) <= 0.67617

    # This test always fails due to using all indices in the SYK Hamiltonian.
