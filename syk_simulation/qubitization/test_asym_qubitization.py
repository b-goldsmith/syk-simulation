# from pytest import mark
from syk_simulation.qubitization.asymmetric_qubitization import AsymmetricQubitization, OracleB, OracleA
from psiqworkbench import QPU, Qubits
import numpy as np
from scipy.stats import normaltest


# TODO do more than 1 test with num_terms = 12
def test_oracleb():
    num_terms = 12
    num_index_qubits = int(np.ceil(np.log2(num_terms)))
    qpu = QPU(num_qubits=num_index_qubits, filters=[">>state-vector-sim>>", ">>buffer>>"])
    index = Qubits(num_index_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    state = qpu.pull_state_specific(index)
    for coeff in state:
        assert np.isclose(np.abs(coeff), 1 / np.sqrt(2**num_index_qubits))


def test_oraclea():
    num_terms = 12
    num_index_qubits = int(np.ceil(np.log2(num_terms)))
    qpu = QPU(num_qubits=num_index_qubits, filters=[">>state-vector-sim>>", ">>buffer>>"])
    index = Qubits(num_index_qubits, "index", qpu=qpu)
    oracleA = OracleA()
    oracleA.compute(index=index, random_depth=2)
    state = qpu.pull_state_specific(index)
    mean = np.mean(state)
    var = np.var(state)

    assert abs(mean) < 5 / np.sqrt(len(state))

    assert abs(var - 1 / len(state)) < 0.3 / len(state)

    stat, p_value = normaltest(np.real(state))

    # assert p_value > 0.01


# @mark.parametrize(
#     "N,num_terms,depth",
#     [
#         # (6, 12, 2),
#         # (8, 70, 5),
#     ],
# )  # (10, 100, 7)
def manual_test_asymmetric_qubitization(N, depth):
    # N = 8  # number of Majorana fermions, 100 is interesting
    # num_terms = 70  # number of Pauli terms in the Hamiltonian
    # depth = 5

    num_system_qubits = N
    num_terms = N**4
    num_index_qubits = int(np.ceil(np.log2(num_terms)))

    total_qubits = int(num_system_qubits + num_index_qubits + 1)
    qpu = QPU(num_qubits=total_qubits, filters=[">>unitary>>", ">>buffer>>"])

    branch = Qubits(1, "branch", qpu=qpu)
    index = Qubits(num_index_qubits, "index", qpu=qpu)
    system = Qubits(N, "system", qpu=qpu)

    AQ = AsymmetricQubitization()

    AQ.compute(branch=branch, index=index, system=system, depth=depth)

    assert True  # Replace with actual checks


def generate_pauli_strings(num_terms, num_system_qubits):
    pauli_strings = []
    paulis = ["I", "X", "Y", "Z"]
    for _ in range(num_terms):
        term = "".join(np.random.choice(paulis, size=num_system_qubits))
        pauli_strings.append(" ".join(f"{p}{i}" for i, p in enumerate(term)))
    return pauli_strings


# def run_aq():
#     manual_test_asymmetric_qubitization(6, 12, 2)


# run_aq()
