from syk_simulation.qubitization.asymmetric_qubitization import Select
from syk_simulation.qubitization.jw_utils import single_maj_op
import numpy as np
from psiqworkbench import Qubits, QPU
from psiqworkbench.utils.numpy_utils import reverse_numpy_op
import pytest


@pytest.mark.parametrize("N", (4, 5, 6, 7, 8))
def test_select(N):
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
    """

    qpu = QPU(num_qubits=total_qubits)
    qpu2 = QPU(num_qubits=system_size)

    index_value_p = np.random.choice(range(system_size))
    index_value_q = np.random.choice(range(system_size))
    index_value_r = np.random.choice(range(system_size))
    index_value_s = np.random.choice(range(system_size))

    index = Qubits(index_size, "index", qpu=qpu)
    index[0:index_chunk_size].write(index_value_p)
    index[index_chunk_size : index_chunk_size * 2].write(index_value_q)
    index[2 * index_chunk_size : 3 * index_chunk_size].write(index_value_r)
    index[3 * index_chunk_size :].write(index_value_s)

    hardcoded_system = Qubits(system_size, qpu=qpu2)

    for index_pqrs in [index_value_p, index_value_q, index_value_r, index_value_s]:
        hardcoded_system[index_pqrs].x()
        for z_chain_index in range(index_pqrs - 1, -1, -1):
            hardcoded_system[z_chain_index].z()

    hardcoded_state = hardcoded_system.pull_state()

    system = Qubits(system_size, "system", qpu=qpu)
    select = Select()
    select.compute(index=index, system=system)

    system_state = system.pull_state()

    fidelity = np.abs(np.vdot(hardcoded_state, system_state)) ** 2

    assert np.isclose(fidelity, 1.0, atol=1e-7)
    assert index[0:index_chunk_size].read() == index_value_p
    assert index[index_chunk_size : 2 * index_chunk_size].read() == index_value_q
    assert index[2 * index_chunk_size : 3 * index_chunk_size].read() == index_value_r
    assert index[3 * index_chunk_size : 4 * index_chunk_size].read() == index_value_s


def test_select_reversible():
    """Test to ensure that SELECT is producing a Unitary that is reversible by
    confirming that $SELECT_dagger SELECT = I$"""
    N = np.random.choice(range(4, 9))
    system_size = N
    index_chunk_size = int(np.ceil(np.log2(system_size)))
    index_size = 4 * index_chunk_size
    aux_size = int(np.ceil(np.log2(system_size))) + 1  # aux_index + range_flag
    total_qubits = system_size + index_size + aux_size
    qpu = QPU(num_qubits=total_qubits)

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
    original_state = qpu.pull_state()
    select.compute(index, system)
    select.uncompute()
    final_state = qpu.pull_state()
    assert np.allclose(original_state, final_state, atol=1e-7)


# Test select only with an initial index state against manual creation of the single term
def test_select_with_init_state(N=4, random_seed=5):
    # TODO right now this just has a hardcoded init index state, set this to random and run it multiple times

    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size

    init = np.zeros(2**index_size, dtype=complex)
    init[228] = 1.0

    qpu = QPU(num_qubits=num_qubits)

    for init_idx in range(2**N):
        sys_init = np.zeros(2**N, dtype=complex)
        sys_init[init_idx] = 1.0
        qpu.reset(num_qubits)

        walk = Qubits(walk_size, "walk", qpu)
        branch = Qubits(walk[0:branch_size], "branch")
        index = Qubits(walk[branch_size : index_size + branch_size], "index")
        system = Qubits(walk[index_size + branch_size :], "system")

        branch.had()
        system.push_state(sys_init)
        index.push_state(init)

        zero_init = np.zeros(2**system_size, dtype=complex)
        zero_init[0] = 1.0

        select = Select()
        select.compute(index=index, system=system)
        system_state = system.pull_state()

        temp_matrix = np.zeros((2**N, 2**N), dtype=complex)
        terms_op = single_maj_op(0, N) @ single_maj_op(1, N) @ single_maj_op(2, N) @ single_maj_op(3, N)
        temp_matrix += 1 * terms_op
        temp_matrix = reverse_numpy_op(temp_matrix)

        expected = temp_matrix @ sys_init
        nonzero_classical = np.where(np.abs(expected) > 1e-6)[0]
        nonzero_circuit = np.where(np.abs(system_state) > 1e-6)[0]

        nonzero = np.abs(expected) > 1e-6
        if np.any(nonzero):
            ratios = expected[nonzero] / system_state[nonzero]
            # All ratios must be real
            assert np.allclose(ratios.imag, 0, atol=1e-6), f"Complex ratios for init_idx={init_idx}: {ratios}"
            # All ratios must be equal (consistent global phase/scale)
            assert np.allclose(
                ratios, ratios[0], atol=1e-4
            ), f"Inconsistent ratios for init_idx={init_idx}: {np.round(ratios, 4)}"
            # Ratio must be exactly +1 (same sign, no scale difference)
            assert np.isclose(
                ratios[0].real, 1.0, atol=1e-4
            ), f"Wrong ratio for init_idx={init_idx}: {ratios[0]:.4f} (expected 1.0)"
