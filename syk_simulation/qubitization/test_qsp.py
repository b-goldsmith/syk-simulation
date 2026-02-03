from syk_simulation.qubitization.qsp import qsp_evolution, QSP, get_qsp_phases

from syk_simulation.qubitization.asymmetric_qubitization import AsymmetricQubitization
from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask
import numpy as np


def test_qsp_simple():

    N = 8
    time = 0.5
    epsilon = 1e-3
    random_depth = 2

    num_system_qubits = N
    L = N**4
    num_index_qubits = int(np.ceil(np.log2(L)))
    total_num_qubits = num_system_qubits + num_index_qubits + 1

    qpu = QPU(num_qubits=total_num_qubits)
    branch = Qubits(1, "branch", qpu=qpu)
    index_register = Qubits(num_index_qubits, "index", qpu=qpu)
    system_register = Qubits(num_system_qubits, "system", qpu=qpu)

    # qsp_evolution(N, branch, index_register, system_register, time, epsilon, random_depth)
    # qpu.serialize("syk_aq_N8.basq.dataset.json", dialect="basquiat")
    assert True
