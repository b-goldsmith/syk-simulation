from psiqworkbench import QPU, Qubits, Units
from random import randint, uniform
import numpy as np

from pytest import mark

from .ppr import PPR


@mark.parametrize(
    "num_qubits,x_mask,z_mask,theta",
    [
        # (5, 0b00000, 0b00000, 90.0),  # no non-I qubits in the Pauli operation WAITING ON BUGFIX
        (5, 0b00001, 0b00000, 90.0),  # only 1 non-I qubit in the Pauli operation the rest are identity
        (5, 0b00111, 0b00000, 90.0),
        (5, 0b00000, 0b00111, 90.0),
        (5, 0b00111, 0b00111, 90.0),
        (5, 0b10101, 0b00000, 45.0),
        (5, 0b00000, 0b10101, 45.0),
        (5, 0b10101, 0b10101, 45.0),
        (5, 0b00111, 0b00000, 135.0),
        (5, 0b00000, 0b00111, 135.0),
        (5, 0b00111, 0b00111, 135.0),
        (5, 0b11111, 0b00000, 180.0 * Units.rad),
        (5, 0b00000, 0b11111, 180.0 * Units.rad),
        (5, 0b10111, 0b11101, 180.0 * Units.rad),
        (4, 0b1111, 0b1111, (1, 4)),  # pi/4 radians
    ],
)
def test_simple_ppr(
    num_qubits: int,
    x_mask: int,
    z_mask: int,
    theta: float | Units.RotationAngle | tuple[int, int],
):
    # num_qubits = 4
    # x_mask = 0b0000
    # z_mask = 0b1111
    # theta = 90.0
    run_test_ppr(num_qubits, x_mask, z_mask, theta)


def test_manual_ppr():
    num_qubits = 5
    x_mask = 0b00001
    z_mask = 0b00001
    theta = 90.0
    run_test_ppr(num_qubits, x_mask, z_mask, theta)


def test_ppr():
    num_qubits = randint(2, 4)
    x_mask = randint(0, 2**num_qubits - 1)
    z_mask = randint(0, 2**num_qubits - 1)
    theta = uniform(0.0, 360.0)
    run_test_ppr(num_qubits, x_mask, z_mask, theta)

def test_large_ppr():
    num_qubits = randint(6, 25)
    x_mask = randint(0, 2**num_qubits - 1)
    z_mask = randint(0, 2**num_qubits - 1)
    theta = uniform(0.0, 360.0)
    state = np.random.rand(2**num_qubits)
    run_test_ppr_statevec(num_qubits, x_mask, z_mask, theta, state)

def test_ppr_statevec_manual():
    num_qubits = 15
    x_mask = 0
    z_mask = 0
    theta = 90.0
    state = state = np.random.rand(2**num_qubits)
    run_test_ppr_statevec(num_qubits, x_mask, z_mask, theta, state)

def run_test_ppr(
    num_qubits: int,
    x_mask: int,
    z_mask: int,
    theta: float | Units.RotationAngle | tuple[int, int],
):
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)

    ppr = PPR()
    ppr.compute(qubits, theta=theta, x_mask=x_mask, z_mask=z_mask)

    ufilter = qpu.get_filter_by_name(">>unitary>>")
    matrix = ufilter.get()

    qpu.reset(num_qubits)

    pq_qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    pq_qubits = Qubits(qpu=pq_qpu, num_qubits=num_qubits)

    pq_qubits.ppr(theta=theta, x_mask=x_mask, z_mask=z_mask)
    pq_ufilter = pq_qpu.get_filter_by_name(">>unitary>>")
    matrix2 = pq_ufilter.get()
    assert np.allclose(matrix, matrix2)

def run_test_ppr_statevec(num_qubits: int,
    x_mask: int,
    z_mask: int,
    theta: float | Units.RotationAngle | tuple[int, int],
    state: list | np.ndarray,
):
    qpu1 = QPU(num_qubits = num_qubits)
    qubits1 = Qubits(num_qubits = num_qubits, qpu=qpu1)
    qubits1.push_state(state)
    
    ppr = PPR()
    ppr.compute(qubits1, theta = theta, x_mask = x_mask, z_mask = z_mask)
    vec1 = qpu1.pull_state()
    qpu1.reset(num_qubits)

    qpu2 = QPU(num_qubits = num_qubits)
    qubits2 = Qubits(num_qubits = num_qubits, qpu=qpu2)
    qubits2.push_state(state)
    
    qubits2.ppr(theta = theta, x_mask = x_mask, z_mask = z_mask)
    vec2 = qpu2.pull_state()
    qpu2.reset(num_qubits)

    assert np.allclose(vec1, vec2)
    

    
    