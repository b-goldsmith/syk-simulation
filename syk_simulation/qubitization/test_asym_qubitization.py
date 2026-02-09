from syk_simulation.qubitization.asymmetric_qubitization import AsymmetricQubitization, OracleB, OracleA, Select
from psiqworkbench import QPU, Qubits
import numpy as np


def run_test_oracleb():
    """This test is to confirm that Oracle B is correctly resulting in
        $B_\ell = 1 / \sqrt{L}$
    which is in accordance with 2019 Babbush et. al. paper in the first
    paragraph of section III A.
    """
    num_qubits = np.random.choice(range(1, 21))
    qpu = QPU(num_qubits=num_qubits, filters=[">>state-vector-sim>>"])
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    state = index.pull_state()
    for coeff in state:
        assert np.isclose(np.abs(coeff), 1 / np.sqrt(2**num_qubits))


def test_oracleb():
    for _ in range(1, 10):
        run_test_oracleb()


def test_oracleb_reversible():
    """Test to ensure that Oracle B is producing a Unitary that is reversible."""
    num_qubits = np.random.choice(range(1, 21))
    qpu = QPU(num_qubits=num_qubits)
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleB = OracleB()
    oracleB.compute(index=index)
    oracleB.uncompute()
    state = index.pull_state()
    assert state[0] == 1


def run_test_oraclea():
    """Orthogonal random quantum evolutions [14] hold that the $a_\ell$ are Gaussian
    distributed with zero mean and variance equal to the Hilbert space dimension.
    - from 2019 Babbush et. al. paper in first paragraph of section III A.

    The random depth is set to $2*num_qubits$ to ensure the Gaussian distribution.

    Zero mean tolerance was chosen as $5.0$ as that is $~ 5 * \sigma$ since $\sigma_mean = 1$
    Variance tolerance is $sqrt{2/L}$
    """
    num_qubits = np.random.choice(range(3, 21))
    random_depth = 2 * num_qubits
    qpu = QPU(num_qubits=num_qubits)
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleA = OracleA()
    oracleA.compute(index=index, random_depth=random_depth)
    state = index.pull_state()
    unnormalized_state = state * 2**num_qubits
    mean = np.mean(unnormalized_state)
    var = np.var(unnormalized_state)
    print(f"n:{num_qubits}\nmean: {mean}\nvar: {var}")

    assert abs(mean) < 5.0
    assert np.isclose(2**num_qubits, var, rtol=np.sqrt(2 / (2**num_qubits)))


def test_oraclea():
    for _ in range(10):
        run_test_oraclea()


def test_oraclea_reversible():
    """Test to ensure that Oracle A is producing a Unitary that is reversible."""
    num_qubits = np.random.choice(range(3, 21))
    random_depth = 2 * num_qubits
    qpu = QPU(num_qubits=num_qubits, filters=[">>state-vector-sim>>"])
    index = Qubits(num_qubits, "index", qpu=qpu)
    oracleA = OracleA()
    oracleA.compute(index=index, random_depth=random_depth)
    oracleA.uncompute()
    state = index.pull_state()
    assert state[0] == 1


def test_select():
    for _ in range(3):

        N = np.random.choice(range(4, 9))
        system_size = N
        index_chunk_size = int(np.ceil(np.log2(system_size)))
        index_size = 4 * index_chunk_size
        aux_size = int(np.ceil(np.log2(system_size))) + 1  # aux_index + range_flag
        total_qubits = system_size + index_size + aux_size

        for _ in range(3):
            run_select_test(system_size, index_chunk_size, index_size, total_qubits)


def run_select_test(system_size, index_chunk_size, index_size, total_qubits):
    """This is testing the select to ensure that
    |l>\gamma_l|\psi> = |l>X_l \cdotp Z_(l-1) \cdotp Z_(l-2) \cdotp\cdotp\cdotp Z_0|\psi>
    - from 2019 Babbush et. al. paper equation 18.

    To do this test we'll compare the state vector results to make sure they match
    limited to N of 8 due to the number of required qubits and  RAM limitations
    """

    qpu = QPU(num_qubits=total_qubits)
    qpu2 = QPU(num_qubits=system_size)

    # test simple case where index is a single int
    index_value_p = np.random.choice(range(system_size))
    index_value_q = np.random.choice(range(system_size))
    index_value_r = np.random.choice(range(system_size))
    index_value_s = np.random.choice(range(system_size))

    index = Qubits(index_size, "index", qpu=qpu)
    index[0:index_chunk_size].write(index_value_p)
    index[index_chunk_size : 2 * index_chunk_size].write(index_value_q)
    index[2 * index_chunk_size : 3 * index_chunk_size].write(index_value_r)
    index[3 * index_chunk_size : 4 * index_chunk_size].write(index_value_s)

    system = Qubits(system_size, "system", qpu=qpu)
    select = Select()
    select.compute(index=index, system=system)
    system_state = system.pull_state()

    hardcoded_system = Qubits(system_size, qpu=qpu2)

    for index_pqrs in [index_value_p, index_value_q, index_value_r, index_value_s]:
        hardcoded_system[index_pqrs].x()
        for z_chain_index in range(index_pqrs):
            hardcoded_system[z_chain_index].z()

    hardcoded_state = hardcoded_system.pull_state()

    qpu.draw(filename="syk-select.svg")
    qpu2.draw(filename="syk-hardcoded.svg")

    print(f"hardcoded:{np.nonzero(hardcoded_state)[0]} - select:{np.nonzero(system_state)[0]}")
    fidelity = np.abs(np.vdot(hardcoded_state, system_state)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-07)
    assert index[0:index_chunk_size].read() == index_value_p
    assert index[index_chunk_size : 2 * index_chunk_size].read() == index_value_q
    assert index[2 * index_chunk_size : 3 * index_chunk_size].read() == index_value_r
    assert index[3 * index_chunk_size : 4 * index_chunk_size].read() == index_value_s
