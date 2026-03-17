import numpy as np
from scipy.linalg import expm
from psiqworkbench import QPU, Qubits

from .ppr.ppr import PPR
from .jw_transform.hamiltonian import SYK_hamil
from .trotter.trotter import second_order_trotter
from .qdrift.qdrift import qdrift
from workbench_algorithms.utils import pauli_sum_to_numpy

def compute_fidelity(matrix1, matrix2):
    """Computes overlap between two unitaries: 1.0 is perfect match."""
    m1 = np.array(matrix1, dtype=complex)
    m2 = np.array(matrix2, dtype=complex)
    return np.abs(np.trace(m1.conj().T @ m2)) / m1.shape[0]

def run_comparison():
    n_qubits = 4        
    time = 0.5          
    trotter_steps = 15  
    qdrift_samples = 3000 
    
    print(f"--- 4-QUBIT SYK CROSS-COMPARISON ---")
    print(f"Goal: Compare Accuracy & Gate Cost at Time t={time}")

    # 4 qubits = 8 Majoranas = 70 terms (8 choose 4)
    ham = SYK_hamil(n=n_qubits, J=1.0, random_seed=42)
    num_terms = len(ham)
    ham_mat = pauli_sum_to_numpy(ham)
    
    exact_u = expm(-1j * ham_mat * time)
    
    ppr = PPR()

    qpu_t = QPU(num_qubits=n_qubits, filters=">>unitary>>")
    qubits_t = Qubits(qpu=qpu_t, num_qubits=n_qubits)
    second_order_trotter(ham, qubits_t, ppr, time, trotter_steps)
    ## extract final unitary matrix for trotter
    trotter_u = qpu_t.get_filter_by_name(">>unitary>>").get()
    trotter_gates = 2 * num_terms * trotter_steps

    qpu_q = QPU(num_qubits=n_qubits, filters=">>unitary>>")
    qubits_q = Qubits(qpu=qpu_q, num_qubits=n_qubits)
    qdrift(ham, qubits_q, ppr, time, qdrift_samples, random_seed=42)
    ## extract final unitary matrix for qdrift
    qdrift_u = qpu_q.get_filter_by_name(">>unitary>>").get()
    qdrift_gates = qdrift_samples

    fid_t_vs_exact = compute_fidelity(exact_u, trotter_u)
    fid_q_vs_exact = compute_fidelity(exact_u, qdrift_u)
    fid_cross = compute_fidelity(trotter_u, qdrift_u)

    print(f"\n{'='*60}")
    print(f"{'Algorithm':<15} | {'Gates':<10} | {'Fidelity':<10} | {'Error':<10}")
    print(f"{'-'*60}")
    print(f"{'Trotter (2nd)':<15} | {trotter_gates:<10} | {fid_t_vs_exact:<10.6f} | {1-fid_t_vs_exact:.6f}")
    print(f"{'qDRIFT':<15} | {qdrift_gates:<10} | {fid_q_vs_exact:<10.6f} | {1-fid_q_vs_exact:.6f}")
    print(f"{'='*60}")

    print(f"Direct Trotter-to-qDRIFT Match: {fid_cross:.6f}")
    
    print(f"\n[HAMILTONIAN STATS]")
    print(f"Total Terms: {num_terms}")
    print(f"Trotter uses {trotter_gates / qdrift_gates:.2f}x more gates than qDRIFT")

    ## The dt for trotter could be too big, or the num of samples for qdrift could be too small
    if fid_cross > 0.95:
        print("\nVERDICT: SUCCESS. Both methods converge")
    else:
        print("\nVERDICT: MARGINAL. Increase samples or steps for better agreement.")

if __name__ == "__main__":
    run_comparison()