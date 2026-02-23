from workbench_algorithms.utils import PauliMask, PauliSum 
from math import floor 
from itertools import combinations
from .hamiltonian import SYK_pair_to_mask, SYK_hamil
from scipy.special import comb

import numpy as np
from math import isclose
from scipy.stats import normaltest
from pytest import mark


def sign_coefficient(array: tuple | list):
    #For checking coefficients in product of four majorana modes with increasing indices
    parity = 1
    coefs = [floor(i/2) for i in array]
    par = [i%2 for i in array]
    ln = len(set(coefs))

    if (ln == 4) and (par[0] == par[2]):
        parity*=-1
    elif ln==3:
        if (coefs[0]==coefs[1]) and (par[2]==1):
            parity*=-1
        elif par[0]==1:
            parity*=-1
    elif ln==2:
        parity *=-1
    return parity

def mult_SYK_pairs(array):
    pauli_op1 = SYK_pair_to_mask(array[0], array[1])
    pauli_op2 = SYK_pair_to_mask(array[2], array[3])

    return [np.real(pauli_op1[0]*pauli_op2[0]), pauli_op1[1]*pauli_op2[1]]

@mark.parametrize(
    "coefficient, indices, pauli_string",
    [
        (-1, (0,1,2,3), 'Z0 Z1'),
        (-1, (4,5,6,7), 'Z2 Z3'),
        (-1, (0,1,12,13), 'Z0 Z6'),
        (-1, (1,2,3,4), 'X0 X2'),
        (-1, (7,8,9,10), 'X3 X5'),
        (-1, (0,1,7,8), 'Z0 X3 X4'),
        (1, (2,3,6,8), 'Z1 Y3 X4'),
        (1, (0,1,2,4), 'Z0 Y1 X2'),
        (-1, (2,4,6,8), 'Y1 X2 Y3 X4'),
        (-1, (0,6,12,14), 'Y0 Z1 Z2 X3 Y6 X7'),
        (1, (0,3,8,9), 'Y0 Y1 Z4'),
    ],
)

def test_jw_transform(coefficient, indices, pauli_string):
    pauli_op1 = SYK_pair_to_mask(indices[0],indices[1])
    pauli_op2 = SYK_pair_to_mask(indices[2],indices[3])

    pauli_op = pauli_op1[1]*pauli_op2[1]
    assert [pauli_op1[0]*pauli_op2[0], pauli_op.get_pauli_string()] == [coefficient, pauli_string]

def test_SYK_pair_phase():
    for i in range(50):
        p = np.random.randint(0,200)
        q = np.random.randint(0,200)
        if p != q:
            summand1 = SYK_pair_to_mask(p,q)
            summand2 = SYK_pair_to_mask(q,p)
        assert summand1 == [-1*summand2[0], summand2[1]]

def test_SYK_coefficient():
    for i in range(100):
        array = np.random.randint(0,200, size = 4)
        while len(set(array))<4:
            array = np.random.randint(0,200, size = 4)
        array.sort()
        summand = mult_SYK_pairs(array)
        assert sign_coefficient(array) == summand[0]


def test_majorana_prod_string():
    for i in range(100):
        array = np.random.randint(0,200, size = 4)
        while len(set(array))<4:
            array = np.random.randint(0,200, size = 4)
        array.sort()
        array_perm = np.random.permutation(array)

        assert mult_SYK_pairs(array_perm)[1] == mult_SYK_pairs(array)[1]


def test_syk_summands():
    for i in range(2,12):
        theory = comb(2*i, 4)
        hamil = SYK_hamil(i)
        assert theory == len(hamil)

def compare_syk_distribution(n: int, J: float, rtol=1e-3, atol=1e-4):
    hamil = SYK_hamil(n, J= J)
    coefs = hamil.get_coefficients()

    #Test that the coefficients are normally distributed
    res1 = normaltest(coefs)

    #Scaling of 4!J is taken into account as well as 1/(4*4!)
    variance = (6/(2*n)**3)*(J/4)**2

    assert res1[1]>0.05
    assert np.abs(np.mean(coefs))<1e-3
    assert isclose(variance, np.var(coefs), rel_tol=rtol, abs_tol=atol)

def test_syk_statistics():
     for _ in range(5):
        n=np.random.randint(15,30)
        J=10
        compare_syk_distribution(n=n, J=J)
     

    
