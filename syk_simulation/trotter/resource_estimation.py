import numpy as np
import matplotlib.pyplot as plt
from psiqworkbench import QPU, Qubits, resource_estimator
from ..ppr.ppr import PPR
from ..jw_transform.hamiltonian import SYK_hamil
from .trotter import second_order_trotter
from scipy.special import comb

from decimal import Decimal, getcontext
getcontext().prec = 60

def calculate_t_gate_costs(rotations, epsilon_total):
    """
    Calculates T-gates using high-precision decimals to avoid 'inf'
    and divide-by-zero errors.
    """
    if rotations <= 0:
        return 0
    
    # Convert inputs to high-precision Decimals
    rot = Decimal(str(rotations))
    eps_tot = Decimal(str(epsilon_total))
    one = Decimal('1')
    
    # Equation (1): eps_single = 1 - (1 - eps_total)**(1/rotations)
    # Using Decimal power for extreme precision
    eps_single = one - (one - eps_tot)**(one / rot)
    
    # If eps_single is effectively 0, we avoid log error
    if eps_single <= 0:
        return float('inf') 

    # Mean T-count cost formula: 0.53 * log2(1/eps_single) + 4.86
    # log2(x) = ln(x) / ln(2)
    inv_eps = one / eps_single
    log2_inv_eps = inv_eps.ln() / Decimal('2').ln()
    
    t_per_rotation = Decimal('0.53') * log2_inv_eps + Decimal('4.86')
    
    return float(rot * t_per_rotation)

def SYK_trotter_fetch_res(number_qubits: int, 
                         time: float,
                         epsilon: float,
                         J: float=24, 
                         coefs: list | None = None,
                         random_seed: int | None = None,
                         break_rot=False):
    '''
    Function returning QREs for hamiltonian simulation using
    SYK model

    Parameters:
    number_qubits (int): number of qubits for the SYK hamiltonian (half the number of majoranas)
    time (float): simulation time
    epsilon (float): error tolerance
    J: scaling constant for SYK model
    coefs: Given list of coefficients for the Hamiltonian
    break_rot: Boolean variable indicating if rotations should be broken up using rs-synth-filter
    '''
    steps = get_commutator_bound_steps(n=number_qubits, t=time, epsilon=epsilon, J=J)

    if break_rot:
        qpu = QPU(num_qubits = number_qubits, filters = [">>clean-ladder-filter>>", ">>single-control-filter>>", ">>rs-synth-filter>>"])
        qpu.reset(num_qubits = number_qubits)
    else:
        qpu = QPU(num_qubits = number_qubits, filters =[">>witness>>"])
        qpu.reset(num_qubits = number_qubits)
    
    qubits = Qubits(num_qubits=number_qubits,qpu=qpu)

    ppr = PPR()
    ham = SYK_hamil(n=number_qubits, J=24, coefs = coefs, random_seed=random_seed)
    second_order_trotter(hamiltonian=ham, qubits=qubits, ppr_instance=ppr, time=time, num_trotter_steps=steps)


    res = resource_estimator(qpu).resources()
    print(res)
    return res["rotations"]


def get_analytical_syk_lambda(n, J=24.0):
    """Calculates the L1 norm (lambda) for SYK-4 analytically."""
    N = 2 * n
    L = comb(N, 4)
    # Average |c| for SYK coefficients with variance 3!J^2/N^3
    sigma = np.sqrt(6.0 * (J**2) / (N**3))
    avg_abs_c = sigma * np.sqrt(2.0 / np.pi)
    return L * avg_abs_c

def get_commutator_bound_steps(n, t, epsilon, J=24.0, p=2):
    """
    Calculates steps r using the Commutator Bound (Corollary 2).
    r = (alpha_comm * t^(p+1) / epsilon)^(1/p)
    """
    N = 2 * n
    lam = get_analytical_syk_lambda(n, J=J)
    alpha_comm = (N**2) * lam
    r = np.power((alpha_comm * np.power(t, p+1)) / epsilon, 1/p)
    return int(np.ceil(max(r, 1)))

def calculate_t_gate_cost(rotations, epsilon_total):
    """
    Calculates T-gates using the exact relationship:
    eps_rot_single = 1 - (1 - eps_total)**(1/rotations)
    Then applies the Mixed Fallback mean cost formula: 0.53*log2(1/eps) + 4.86
    """
    if rotations <= 0:
        return 0

    # eps_total = 1 - (1 - eps_single)**rotations
    eps_single = 1 - np.power((1 - epsilon_total), 1/rotations)
    
    # Mixed Fallback variant formula from arXiv:2203.10064 (Table 1)
    t_per_rotation = 0.53 * np.log2(1 / eps_single) + 4.86
    
    return rotations * t_per_rotation

def run_targeted_estimates():
    J = 24.0
    time = 1.0
    header = "type,N,J,time,epsilon,random_seed,rotations,t_gates,total_t_gates,qubit_highwater"

    #  Vary Epsilon 
    print(f"--- DATA FOR CSV: VARYING EPSILON ---\n{header}")
    big_N_values = [16, 32, 64, 100]
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    
    for N in big_N_values:
        n_qubits = N // 2
        for eps in epsilons:
            steps = get_commutator_bound_steps(n_qubits, time, eps, J)
            rotations = 2 * comb(N, 4) * steps
            total_t = calculate_t_gate_costs(rotations, eps)
            print(f"Trotter,{N},{J},{time},{eps},0,{rotations},0,{total_t},{n_qubits}")

    # Fixed Epsilon
    print(f"\n--- DATA FOR CSV: VARYING N ---\n{header}")
    fixed_eps = 0.001
    detailed_N = [8, 12, 16, 20, 24, 28, 32, 36, 42, 46, 48, 56, 64, 68, 72, 76, 
                  80, 90, 94, 96, 100, 104, 108, 112, 116, 120, 124, 128, 130, 
                  134, 132, 138, 142, 146, 152, 156, 160, 164, 168, 172, 178, 
                  182, 186, 190, 194, 200]
    
    for N in detailed_N:
        n_qubits = N // 2
        steps = get_commutator_bound_steps(n_qubits, time, fixed_eps, J)
        rotations = 2 * comb(N, 4) * steps
        total_t = calculate_t_gate_costs(rotations, fixed_eps)
        print(f"Trotter,{N},{J},{time},{fixed_eps},0,{rotations},0,{total_t},{n_qubits}")



if __name__ == "__main__":
    #SYK_trotter_fetch_res(8, 1, 0.001)
    run_targeted_estimates()