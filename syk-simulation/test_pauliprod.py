from psiqworkbench import QPU, Qubits, Units
from workbench_algorithms import TrotterQuery
from workbench_algorithms.utils import PauliMask, PauliSum 
import numpy as np

from pytest import mark

from ppr_utils import apply_hamiltonian_as_pprs
#from ppr import PPR

#@mark.parametrize(
#    "num_qubits, hamiltonian",
#    [
#        (5, PauliSum([np.pi/3, PauliMask(x_mask = 0b00001, z_mask = 0b00010)], [np.pi/4, PauliMask(x_mask = 0b00111, z_mask = 0b10001)])), 
#        (5, PauliSum([np.pi/4, PauliMask(x_mask = 0b00111, z_mask = 0b10001)], [np.pi/3, PauliMask(x_mask = 0b00001, z_mask = 0b00010)])), 
#        (3, PauliSum([np.pi/3, PauliMask(x_mask = 0b001, z_mask = 0b010)], [np.pi/4, PauliMask(x_mask = 0b111, z_mask = 0b001)], [np.pi/5, PauliMask(x_mask = 0b101, z_mask = 0b111)])), 
#    ],
#)

#def test_pauliprod(num_qubits: int, hamiltonian: PauliSum):
#    run_pauliprod_test(num_qubits = num_qubits, hamiltonian = hamiltonian)

def quantum_ising(num_qubits: int, J: float=1.0, g: float =2.0):
    """
    Returns hamiltonian for transverse-field 1D Ising model as PauliSum
    """
    hamil = PauliSum([-J, PauliMask(0,2**(num_qubits-1)+1)])
    for i in range(1, num_qubits):
        hamil.append([-J, PauliMask(0,2**i + 2**(i-1)) ])
    for i in range(num_qubits):
        hamil.append([-J*g, PauliMask(2**i, 0) ])
    return hamil
    

def test_pauliprod_commute():
    hamil1 = PauliSum([1, PauliMask(x_mask = 0b10, z_mask=0b00)], [1, PauliMask(x_mask = 0b00, z_mask=0b01)])
    hamil2 = PauliSum([1, PauliMask(x_mask = 0b00, z_mask=0b01)], [1, PauliMask(x_mask = 0b10, z_mask=0b00)])

    qpu1 = QPU(num_qubits = 2, filters = ">>unitary>>")
    qubits1 = Qubits(num_qubits = 2, qpu=qpu1)

    apply_hamiltonian_as_pprs(hamiltonian = hamil1, qubits= qubits1, ppr_instance= PPR)

    ufilter = qpu1.get_filter_by_name(">>unitary>>")
    matrix1 = ufilter.get()

    qpu1.reset(2)

    qpu2 = QPU(num_qubits = 2, filters = ">>unitary>>")
    qubits2 = Qubits(num_qubits = 2, qpu=qpu2)

    apply_hamiltonian_as_pprs(hamiltonian = hamil2, qubits= qubits2, ppr_instance= PPR)

    ufilter = qpu2.get_filter_by_name(">>unitary>>")
    matrix1 = ufilter.get()

    assert np.allclose(matrix1, matrix2)

def test_pauliprod_manual():
    num_qubits = 3
    hamil = quantum_ising(num_qubits)
    run_pauliprod_test(num_qubits = num_qubits, hamiltonian = hamil)


def run_pauliprod_test(num_qubits: int,
    hamiltonian: PauliSum,
):
    qpu1 = QPU(num_qubits = num_qubits, filters = ">>unitary>>")
    qubits1 = Qubits(num_qubits = num_qubits, qpu=qpu1)
    
    apply_hamiltonian_as_pprs(hamiltonian = hamiltonian, qubits= qubits1, ppr_instance= PPR)

    ufilter = qpu1.get_filter_by_name(">>unitary>>")
    matrix1 = ufilter.get()

    qpu1.reset(num_qubits)

    qpu2 = QPU(num_qubits = num_qubits, filters = ">>unitary>>")
    qubits2 = Qubits(num_qubits = num_qubits, qpu=qpu2)
    
    trotter = TrotterQuery(hamiltonian, trotter_order = 1)
    trotter.compute(qubits2, steps= 1, evo_time = 1)
    
    ufilter = qpu2.get_filter_by_name(">>unitary>>")
    matrix2 = ufilter.get()

    assert np.allclose(matrix1, matrix2)