from syk_simulation.qubitization.asymmetric_qubitization import OracleB
import numpy as np
from psiqworkbench import Qubits, QPU


def run_test_oracle_b():
    """This test is to confirm that Oracle B is correctly resulting in
        $B_l = 1 / \sqrt{L}$
    which is in accordance with 2019 Babbush et. al. paper in the first
    paragraph of section III A.
    """
    num_qubits = np.random.choice(range(1, 10))
    qpu = QPU(num_qubits=num_qubits, filters=[">>state-vector-sim>>"])
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    state = index.pull_state()
    for coeff in state:
        assert np.isclose(np.abs(coeff), 1 / np.sqrt(2**num_qubits))


def test_oracle_b():
    for _ in range(5):
        run_test_oracle_b()


def test_oracle_b_reversible():
    """Test to ensure that Oracle B is producing a Unitary that is reversible."""
    num_qubits = np.random.choice(range(1, 21))
    qpu = QPU(num_qubits=num_qubits)
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    oracleB.uncompute()
    state = index.pull_state()
    assert abs(state[0] - 1) < 1e-8
