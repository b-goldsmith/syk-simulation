"""
Algorithm for hamiltonian simulation for the SYK model using qdrift
"""
from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils import PauliMask, PauliSum 
from ..jw_transform import SYK_hamil
from ..ppr import PPR
from .qdrift import qdrift

import numpy as np



def SYK_qdrift(
    qubits: Qubits, 
    time: float,
    epsilon: float,
    J: float=1, 
    coefs: list | None = None, 
    random_seed: int | None = None):
    """
    Function applying time evolution according to SYK model using qdrift to given qubits

    :param qubits: Register time evolution is supposed to act on
    :type qubits: workbench Qubits

    :param time: Register time evolution is supposed to act on
    :type time: float

    :param epsilon: Error threshold
    :type epsilon: float

    :param J: coupling constant for SYK model (for the model with arbitrary coefficients)
    :type J: float

    :param coefs: array of coefficients for SYK model drawn from an appropriate distribution
    :type coefs: list

    :param random_seed(optional, default = none): Ability to set random_seed for testing/reproducability purposes
    :type n: int

    Output: None (time evolution applied to qubits)
    """
    num_qubits = len(qubits)

    #Generate SYK hamiltonian
    hamil = SYK_hamil(n = num_qubits, J= J, coefs = coefs, random_seed = random_seed)

    #Determine required sample size for qdrift
    num_samples = int(np.ceil(4 * ((hamil.norm()* (time ** 2)) / epsilon)))
    
    ppr = PPR()
    #Perform qdrift
    qdrift(hamil, qubits, ppr, time, num_samples, random_seed)

def SYK_qdrift_sample(qubits: Qubits, 
    time: float,
    num_samples: int,
    J: float=1, 
    coefs: list | None = None, 
    random_seed: int | None = None):

    num_qubits = len(qubits)

    #Generate SYK hamiltonian
    hamil = SYK_hamil(n = num_qubits, J= J, coefs = coefs, random_seed = random_seed)
    
    ppr = PPR()
    #Perform qdrift
    qdrift(hamil, qubits, ppr, time, num_samples, random_seed)