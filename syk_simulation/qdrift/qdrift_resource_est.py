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

def SYK_qdrift_samplesize(number_qubits: int, 
                         time: float,
                         epsilon: float,
                         J: float=1, 
                         coefs: list | None = None,
                         random_seed: int | None = None):
    '''
    Function returning the necessary sample size to achieve given error for the
    SYK model and qdrift for given number of qubits

    Parameters:
    number_qubits (int): number of qubits for the SYK hamiltonian (half the number of majoranas)
    time (float): simulation time
    epsilon (float): error tolerance
    J: scaling constant for SYK model
    coefs: Given list of coefficients for the Hamiltonian
    break_rot: Boolean variable indicating if rotations should be broken up using rs-synth-filter
    '''

    hamil = SYK_hamil(number_qubits, J, coefs = coefs, random_seed = random_seed)
    exact_mat = expm(-1j * pauli_sum_to_numpy(hamil) * time)

    worst_case = int(np.ceil(4 * ((hamil.norm()* (time ** 2)) / epsilon)))
    sample_size = worst_case//2+1 #1/2 of the upper bound to start (rounded up)

    while sample_size<worst_case:

        qpu = QPU(num_qubits = number_qubits, filter = ">>unitary>>")
        qubits = Qubits(num_qubits=number_qubits,qpu=qpu)
            
        SYK_qdrift_sample(qubits = qubits, time = time, num_samples = sample_size, J=J, random_seed=random_seed)
        ufilter = qpu.get_filter_by_name(">>unitary>>")
        matrix = ufilter.get()
        qpu.reset(number_qubits)

        if np.linalg.norm(exact_mat-matrix, ord=2)< epsilon:
            return sample_size
        
        sample_size += 25

    return worst_case

# #Sample size test
# eps = np.logspace(-3,-6,20)
# sample = []
# upper_bound = []
# ratio = []

# for num in range(4,5):
#     hamil=SYK_hamil(n=num, J=1, random_seed = 42)
#     norm = hamil.norm()
#     coefs = hamil.get_coefficients()
#     print("Qubits",num)

#     for err in eps:
#         actual = SYK_qdrift_samplesize(number_qubits = num, time =1, epsilon = err, J=1, coefs = coefs, random_seed = 42)
#         worst = int(np.ceil(4 * (norm/ err)))
#         sample.append(actual)
#         upper_bound.append(worst)
#         ratio.append(actual/worst)
#         print("Eps", err)

# error = np.tile(eps, len(range(4,5)))
# num_qubits = np.repeat( np.arange(4,5,1), len(range(4,5)))

# with open("syk_simulation/RE_data/qdrift_samplesize.csv", "w", newline="") as f:
#     writer = csv.writer(f)
    
#     # header
#     writer.writerow(["Number of qubits", "Algorithmic error", "Sample Size", "Upper Bound", "Ratio"])
    
#     # rows
#     for num,eps,sam,worst,rat in zip(num_qubits, error, sample, upper_bound, ratio):
#         writer.writerow([num,eps,sam,worst,rat])


#Resource estimate for fixed error and growing number of qubits
#For lower counts we also use the workbench resource estimator

number_rot = []
nums_qubits = np.linspace(4,100,45, dtype = int)
worst_case =[]
number_rot_wb=[]
time = 1
eps = [0.01 for _ in range(len(nums_qubits))]

for num in nums_qubits:
    np.random.seed(42)
    scale = np.sqrt(6/(2*num)**3)
    number_coef = int(comb(2*num,4))
    coef = np.random.normal(loc=0, scale= scale, size = number_coef)
    exp_norm = (1/96)*number_coef*scale*np.sqrt(2/np.pi)
    norm = (1/96)*sum(np.abs(coef))
    analytical_res = np.ceil(((2*time)**2*norm)/eps[0])
    exp_res = np.ceil(((2*time)**2*exp_norm)/eps[0])
    number_rot.append(analytical_res)
    worst_case.append(exp_res)

eps = np.concatenate((eps, np.logspace(-2,-6,20)), axis = 0)
time = 1
nums_qubits=np.append(nums_qubits,[12 for _ in range(len(np.logspace(-2,-6,20)))])

np.random.seed(42)
scale = np.sqrt(6/(2*nums_qubits[-1])**3)
number_coef = int(comb(2*nums_qubits[-1],4))
coef = np.random.normal(loc=0, scale= scale, size = number_coef)
norm = (1/96)*sum(np.abs(coef))

for error in np.logspace(-2,-6,20):
    analytical_res = np.ceil(((2*time)**2*norm)/error)
    number_rot.append(analytical_res)

with open("syk_simulation/RE_data/qdrift.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # header
    writer.writerow(["Number of qubits", "Algorithmic error", "Rotations"])
    
    # rows
    for num,eps,rot in zip(nums_qubits, eps, number_rot):
        writer.writerow([num,eps,rot])

#Compare with workbench
for num in [8,12]:
    for error in np.logspace(-2,-6,20):
        hamil = SYK_hamil(n=num, random_seed = 42)
        coefs = hamil.get_coefficients()
        number_rot_wb.append(SYK_qdrift_fetch_res(number_qubits = num, time = time, epsilon = error, coefs = coefs, random_seed = 42, break_rot = False))
        print(num, error, np.where(np.logspace(-2,-6,20)==error))

with open("syk_simulation/RE_data/qdrift_wb.csv", "w", newline="") as f:
    writer = csv.writer(f)
    
    # header
    writer.writerow(["Number of qubits", "Algorithmic error", "Rotations"])
    
    # rows
    for num,eps,rot in zip([8 for _ in range(20)] + [12 for _ in range(20)], np.tile(np.logspace(-2,-6,20),2), number_rot_wb):
        writer.writerow([num,eps,rot])




