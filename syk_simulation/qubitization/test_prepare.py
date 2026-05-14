import pytest
from psiqworkbench import QPU, Qubits
from syk_simulation.qubitization.asymmetric_qubitization import OracleA, OracleB
from syk_simulation.qubitization.utils import get_oraclea_coefficients
import numpy as np


@pytest.mark.parametrize("random_seed", (4, 5, 6, 7, 8))
def test_prepare(random_seed):
    N = 4
    # random_seed = 6

    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size)))
    total_aux_size = aux_size + 1 + 1  # aux_index + range_flag + branch
    total_qubits = system_size + index_size + total_aux_size
    qpu = QPU(num_qubits=total_qubits)

    quantum_walk = Qubits(system_size + index_size + 1, "walk", qpu=qpu)  # 1 is the branch qubit

    branch = Qubits(quantum_walk[0], "branch")
    index = Qubits(quantum_walk[1 : index_size + 1 :], "index")
    system = Qubits(quantum_walk[index_size + 1 :], "system")

    # Instead of callling AsymmetricQubitization, just do the steps for PREPARE

    oracleA = OracleA(random_seed=random_seed)
    oracleB = OracleB()

    # Set random_depth based on empircal testing of test_oraclea(N)
    N = len(system)
    n = len(index)
    random_depth = 2 * n

    branch.had()
    # Run PREPARE for qubitization
    oracleA.compute(index=index, random_depth=random_depth, ctrl=(~branch))
    oracleB.compute(index=index, ctrl=(branch))

    prepare_index_state = quantum_walk.pull_state()
    branch_0_system_0_amplitudes = prepare_index_state[0 : 2 ** (1 + index_size) : 2]
    branch_0_system_0_normalized = branch_0_system_0_amplitudes / np.linalg.norm(branch_0_system_0_amplitudes)

    ideal_index_state = get_oraclea_coefficients(N, random_seed)
    ideal_real_index_state = np.array([v.real for v in ideal_index_state.values()])

    assert np.allclose(branch_0_system_0_normalized, ideal_real_index_state, atol=1e-6)
