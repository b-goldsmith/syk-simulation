from psiqworkbench import QPU, Qubits, Units
from random import randint
import numpy as np

from .ppr import PPR

def test_ppr():
    num_qubits = randint(1, 4)
    x_mask = randint(0, 2**num_qubits - 1)
    z_mask = randint(0, 2**num_qubits - 1)
    theta = float(randint(0, 360))  # degrees
    run_test_ppr(num_qubits, x_mask, z_mask, theta)

def run_test_ppr(num_qubits: int, x_mask: int, z_mask: int, theta: float | Units.RotationAngle | tuple[int, int]):
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)

    ppr = PPR()
    ppr.compute(qubits, theta=theta, x_mask=x_mask, z_mask=z_mask)
    
    ufilter = qpu.get_filter_by_name('>>unitary>>')
    matrix = ufilter.get()

    qpu.reset(num_qubits)

    qubits.ppr(theta=theta, x_mask=x_mask, z_mask=z_mask)
    ufilter = qpu.get_filter_by_name('>>unitary>>')
    matrix2 = ufilter.get()
    # qpu.print_instructions()
    assert np.allclose(matrix, matrix2)