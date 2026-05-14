"""
Resource estimates for qdrift
"""

import numpy as np
import csv
from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils.paulimask import PauliSum, pauli_sum_to_numpy 

from psiqworkbench.resource_estimation.qre import resource_estimator
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.linalg import expm

from .SYK_qdrift import SYK_qdrift, SYK_qdrift_sample
from ..jw_transform import SYK_hamil
from .qdrift import qdrift 
from ..ppr import PPR


def SYK_qdrift_fetch_res(number_qubits: int, 
                         time: float,
                         epsilon: float,
                         J: float=1, 
                         coefs: list | None = None,
                         random_seed: int | None = None,
                         break_rot=True):
    '''
    Function returning QREs for hamiltonian simulation using
    SYK model and qdrift for given number of qubits

    Parameters:
    number_qubits (int): number of qubits for the SYK hamiltonian (half the number of majoranas)
    time (float): simulation time
    epsilon (float): error tolerance
    J: scaling constant for SYK model
    coefs: Given list of coefficients for the Hamiltonian
    break_rot: Boolean variable indicating if rotations should be broken up using rs-synth-filter
    '''
    
    if break_rot:
        qpu = QPU(num_qubits = number_qubits, filters = [">>clean-ladder-filter>>", ">>single-control-filter>>", ">>rs-synth-filter>>"])
        qpu.reset(num_qubits = number_qubits)
    else:
        qpu = QPU(num_qubits = number_qubits, filters =[">>witness>>"])
        qpu.reset(num_qubits = number_qubits)
    
    qubits = Qubits(num_qubits=number_qubits,qpu=qpu)

    SYK_qdrift(qubits = qubits, time = time, epsilon = epsilon, J=J, random_seed=random_seed)

    res = resource_estimator(qpu).resources()
    return res["rotations"]

#Resource estimate for fixed error and growing number of qubits
#For lower counts we also use the workbench resource estimator

# number_rot = []
# nums_qubits = np.linspace(4,100,30, dtype = int)
# worst_case =[]
# number_rot_wb=[]
# time = 1
# J = 24
# eps = [5e-5 for _ in range(len(nums_qubits))]

# for num in nums_qubits:
#     np.random.seed(42)
#     scale = np.sqrt(6/(2*num)**3)*J
#     number_coef = int(comb(2*num,4))
#     coef = np.random.normal(loc=0, scale= scale, size = number_coef)
#     exp_norm = (1/96)*number_coef*scale*np.sqrt(2/np.pi)
#     norm = (1/96)*sum(np.abs(coef))
#     analytical_res = np.ceil(((2*time)**2*norm)/eps[0])
#     exp_res = np.ceil(((2*time)**2*exp_norm)/eps[0])
#     number_rot.append(analytical_res)
#     worst_case.append(exp_res)

number_rot = []
nums_qubits = []
# worst_case =[]
# number_rot_wb=[]
time = 1
J = 24
eps2 = [0.5*1/10**x for x in np.arange(3,14)]
eps = np.tile(eps2, 4)
qubits = [8, 16, 32, 50]
for i in qubits:
    nums_qubits=np.append(nums_qubits, [i for _ in range(len(eps2))])

np.random.seed(42)
for num in qubits:
    np.random.seed(42)
    scale = np.sqrt(6/(2*num)**3)*J
    number_coef = int(comb(2*num,4))
    coef = np.random.normal(loc=0, scale= scale, size = number_coef)
    norm = (1/96)*sum(np.abs(coef))

    for error in eps2:
        analytical_res = np.ceil(((2*time)**2*norm)/error)
        number_rot.append(analytical_res)

#J_row = [J for _ in range(len(eps))]
majorana = [2*n for n in nums_qubits]
#times = [ time for _ in range(len(eps))]
#name = ['qdrift' for _ in range(len(eps))]

with open("syk_simulation/RE_data/qdrift_eps_reloaded.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # header
    writer.writerow(["type", "Majorana", "J", "time", "Algorithmic error", "Rotations", "Number of qubits"])
    
    # rows
    for majorana, eps,rot, num in zip(majorana, eps, number_rot, nums_qubits):
        writer.writerow(['qdrift', majorana, J, time, eps,rot, num])

# #Compare with workbench
# for num in [8,12]:
#     for error in np.logspace(-2,-6,20):
#         hamil = SYK_hamil(n=num, random_seed = 42)
#         coefs = hamil.get_coefficients()
#         number_rot_wb.append(SYK_qdrift_fetch_res(number_qubits = num, time = time, epsilon = error, coefs = coefs, random_seed = 42, break_rot = False))
#         print(num, error, np.where(np.logspace(-2,-6,20)==error))

# with open("syk_simulation/RE_data/qdrift_wb.csv", "w", newline="") as f:
#     writer = csv.writer(f)
    
#     # header
#     writer.writerow(["Number of qubits", "Algorithmic error", "Rotations"])
    
#     # rows
#     for num,eps,rot in zip([8 for _ in range(20)] + [12 for _ in range(20)], np.tile(np.logspace(-2,-6,20),2), number_rot_wb):
#         writer.writerow([num,eps,rot])




