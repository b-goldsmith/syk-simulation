"""
Tests for Trotterization implementations.
"""

import numpy as np
import pytest
from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask
from workbench_algorithms import TrotterQuery
from scipy.linalg import expm

from ppr import PPR

from .trotter import (
    first_order_trotter,
    second_order_trotter,
    trotter_evolution
)


def create_hamiltonian_from_terms(terms: list[tuple[float, int, int]]) -> PauliSum:
    """
    Create a PauliSum from (coeff, x_mask, z_mask) tuples.
    
    Args:
        terms: List of (coefficient, x_mask, z_mask)
    
    Returns:
        PauliSum Hamiltonian
    """
    pauli_terms = []
    for coeff, x_mask, z_mask in terms:
        pauli_mask = PauliMask(x_mask, z_mask)
        pauli_terms.append([coeff, pauli_mask])
    return PauliSum(*pauli_terms)


def pauli_string_to_matrix(x_mask: int, z_mask: int, num_qubits: int) -> np.ndarray:
    """Convert Pauli masks to matrix representation."""
    pauli_matrices = {
        (0, 0): np.array([[1, 0], [0, 1]]),      # I
        (1, 0): np.array([[0, 1], [1, 0]]),      # X
        (0, 1): np.array([[1, 0], [0, -1]]),     # Z
        (1, 1): np.array([[0, -1j], [1j, 0]])    # Y
    }
    
    result = np.array([[1.0]])
    for i in range(num_qubits):
        x_bit = (x_mask >> i) & 1
        z_bit = (z_mask >> i) & 1
       
        result = np.kron(pauli_matrices[(x_bit, z_bit)], result)
    
    return result


def hamiltonian_to_matrix(hamiltonian: PauliSum, num_qubits: int) -> np.ndarray:
    """Convert PauliSum Hamiltonian to matrix form."""
    dim = 2 ** num_qubits
    H_matrix = np.zeros((dim, dim), dtype=complex)
    
    for i in range(len(hamiltonian)):
        coeff = hamiltonian.get_coefficient(i)
        mask = hamiltonian.get_mask(i)
        x_mask = mask[0]
        z_mask = mask[1]
        
        pauli_matrix = pauli_string_to_matrix(x_mask, z_mask, num_qubits)
        H_matrix += coeff * pauli_matrix
    
    return H_matrix


def exact_time_evolution(hamiltonian: PauliSum, num_qubits: int, time: float) -> np.ndarray:
    """Compute exact time evolution operator e^(-iHt)."""
    H_matrix = hamiltonian_to_matrix(hamiltonian, num_qubits)
    return expm(-1j * H_matrix * time)


def compute_fidelity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Compute fidelity between two unitary matrices."""
    return np.abs(np.trace(matrix1.conj().T @ matrix2)) / matrix1.shape[0]


def test_first_order_simple():
    """Test first-order Trotter on a simple 2-qubit system."""
    num_qubits = 2
    
    # H = 0.5 * X0*X1 + 0.3 * Z0*Z1
    hamiltonian = create_hamiltonian_from_terms([
        (0.5, 0b11, 0b00),  # X0*X1
        (0.3, 0b00, 0b11),  # Z0*Z1
    ])
    
    time = 0.5
    num_steps = 10
    
    # Get Trotter result
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
    ppr = PPR()
    
    first_order_trotter(hamiltonian, qubits, ppr, time, num_steps)
    
    ufilter = qpu.get_filter_by_name(">>unitary>>")
    trotter_matrix = ufilter.get()
    
    # Get exact result
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    # Check fidelity
    fidelity = compute_fidelity(trotter_matrix, exact_matrix)
    
    print(f"First-order fidelity (N={num_steps}): {fidelity:.6f}")
    
    # First-order with N=10 should have reasonable fidelity
    assert fidelity > 0.95, f"Fidelity {fidelity} too low"


def test_second_order_simple():
    """Test second-order Trotter on a simple 2-qubit system."""
    num_qubits = 2
    
    # H = 0.5 * X0*X1 + 0.3 * Z0*Z1
    hamiltonian = create_hamiltonian_from_terms([
        (0.5, 0b11, 0b00),  # X0*X1
        (0.3, 0b00, 0b11),  # Z0*Z1
    ])
    
    time = 0.5
    num_steps = 10
    
    # Get Trotter result
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
    ppr = PPR()
    
    second_order_trotter(hamiltonian, qubits, ppr, time, num_steps)
    
    ufilter = qpu.get_filter_by_name(">>unitary>>")
    trotter_matrix = ufilter.get()
    
    # Get exact result
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    # Check fidelity
    fidelity = compute_fidelity(trotter_matrix, exact_matrix)
    
    print(f"Second-order fidelity (N={num_steps}): {fidelity:.6f}")
    
    # Second order should have higher fidelity
    assert fidelity > 0.99, f"Fidelity {fidelity} too low for second order"


def test_second_order_better_than_first():
    """Verify second-order Trotter is more accurate than first-order."""
    num_qubits = 2
    
    # H = 0.8 * X0*X1 + 0.6 * Z0*Z1 + 0.4 * X0
    hamiltonian = create_hamiltonian_from_terms([
        (0.8, 0b11, 0b00),  # X0*X1
        (0.6, 0b00, 0b11),  # Z0*Z1
        (0.4, 0b01, 0b00),  # X0
    ])
    
    time = 1.0
    num_steps = 5  # Use small N to see difference
    
    # First-order
    qpu1 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits1 = Qubits(qpu=qpu1, num_qubits=num_qubits)
    ppr1 = PPR()
    first_order_trotter(hamiltonian, qubits1, ppr1, time, num_steps)
    first_order_matrix = qpu1.get_filter_by_name(">>unitary>>").get()
    
    # Second-order
    qpu2 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits2 = Qubits(qpu=qpu2, num_qubits=num_qubits)
    ppr2 = PPR()
    second_order_trotter(hamiltonian, qubits2, ppr2, time, num_steps)
    second_order_matrix = qpu2.get_filter_by_name(">>unitary>>").get()
    
    # Exact
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    # Calculate fidelities
    fidelity_1st = compute_fidelity(first_order_matrix, exact_matrix)
    fidelity_2nd = compute_fidelity(second_order_matrix, exact_matrix)
    
    print(f"First-order fidelity:  {fidelity_1st:.6f}")
    print(f"Second-order fidelity: {fidelity_2nd:.6f}")
    
    # Second order should be more accurate
    assert fidelity_2nd > fidelity_1st, "Second order should be better than first order"


def test_convergence_with_trotter_steps():
    """Test that Trotter approximation improves as N increases."""
    num_qubits = 2
    
    hamiltonian = create_hamiltonian_from_terms([
        (0.5, 0b11, 0b00),  # X0*X1
        (0.3, 0b00, 0b11),  # Z0*Z1
    ])
    
    time = 0.5
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    fidelities = []
    trotter_steps = [5, 10, 20, 50]
    
    for num_steps in trotter_steps:
        qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
        qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
        ppr = PPR()
        
        second_order_trotter(hamiltonian, qubits, ppr, time, num_steps)
        trotter_matrix = qpu.get_filter_by_name(">>unitary>>").get()
        
        fidelity = compute_fidelity(trotter_matrix, exact_matrix)
        fidelities.append(fidelity)
        print(f"N={num_steps:3d}: fidelity = {fidelity:.8f}")
    
    # Check monotonic improvement
    for i in range(len(fidelities) - 1):
        assert fidelities[i+1] >= fidelities[i] - 1e-10, \
            "Fidelity should improve with more steps"


def test_trotter_evolution_interface():
    num_qubits = 2
    # Use a NON-COMMUTING Hamiltonian: H = X0 + Z0Z1
    hamiltonian = create_hamiltonian_from_terms([
        (0.5, 0b01, 0b00),  # X0
        (0.3, 0b00, 0b11),  # Z0Z1 (Anti-commutes with X0 on qubit 0)
    ])
    
    time = 0.5
    num_steps = 10
    
    # Test order=1
    qpu1 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits1 = Qubits(qpu=qpu1, num_qubits=num_qubits)
    ppr1 = PPR()
    trotter_evolution(hamiltonian, qubits1, ppr1, time, num_steps, order=1)
    matrix1 = qpu1.get_filter_by_name(">>unitary>>").get()
    
    # Test order=2
    qpu2 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits2 = Qubits(qpu=qpu2, num_qubits=num_qubits)
    ppr2 = PPR()
    trotter_evolution(hamiltonian, qubits2, ppr2, time, num_steps, order=2)
    matrix2 = qpu2.get_filter_by_name(">>unitary>>").get()
    
    # Should produce different results
    assert not np.allclose(matrix1, matrix2), "First and second order should differ"
    
    # Test invalid order
    qpu3 = QPU(num_qubits=num_qubits)
    qubits3 = Qubits(qpu=qpu3, num_qubits=num_qubits)
    ppr3 = PPR()
    
    with pytest.raises(ValueError):
        trotter_evolution(hamiltonian, qubits3, ppr3, time, num_steps, order=3)


def test_transverse_field_ising():
    """Test on Transverse-Field Ising Model: H = -J*Z0*Z1 - h*X0 - h*X1"""
    num_qubits = 2
    J = 1.0
    h = 0.5
    
    hamiltonian = create_hamiltonian_from_terms([
        (-J, 0b00, 0b11),  # -J*Z0*Z1
        (-h, 0b01, 0b00),  # -h*X0
        (-h, 0b10, 0b00),  # -h*X1
    ])
    
    time = 1.0
    num_steps = 20
    
    # Get Trotter result
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
    ppr = PPR()
    
    second_order_trotter(hamiltonian, qubits, ppr, time, num_steps)
    trotter_matrix = qpu.get_filter_by_name(">>unitary>>").get()
    
    # Get exact result
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    # Check fidelity
    fidelity = compute_fidelity(trotter_matrix, exact_matrix)
    print(f"TFIM Fidelity: {fidelity:.8f}")
    
    assert fidelity > 0.99, f"TFIM fidelity {fidelity} too low"


@pytest.mark.parametrize("num_steps", [5, 10, 20])
def test_different_step_sizes(num_steps):
    """Test Trotter with different numbers of steps."""
    num_qubits = 2
    
    hamiltonian = create_hamiltonian_from_terms([
        (0.5, 0b11, 0b00),
        (0.3, 0b00, 0b11),
    ])
    
    time = 0.5
    
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
    ppr = PPR()
    
    second_order_trotter(hamiltonian, qubits, ppr, time, num_steps)
    trotter_matrix = qpu.get_filter_by_name(">>unitary>>").get()
    
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    fidelity = compute_fidelity(trotter_matrix, exact_matrix)
    
    # Fidelity should be reasonable for all tested step sizes
    assert fidelity > 0.95, f"Fidelity {fidelity} too low for N={num_steps}"

def test_compare_with_workbench():
    """Verify our Trotter matches the official Workbench version."""
    num_qubits = 2
    evo_time = 0.5
    steps = 2
    
    # Non-commuting Hamiltonian: H = 1.0*X0 + 0.7*Z0Z1
    ham = PauliSum(
        [1.0, PauliMask(1, 0)], # X0
        [0.7, PauliMask(0, 3)]  # Z0Z1
    )

    # --- 1. RUN OFFICIAL WORKBENCH TROTTER ---
    qpu_wb = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    psi_wb = Qubits(qpu=qpu_wb, num_qubits=num_qubits)
    
    # Official Workbench Trotter (First order)
    trotter_wb = TrotterQuery(ham, trotter_order=1)
    trotter_wb.compute(psi_wb, steps, evo_time)
    
    matrix_wb = qpu_wb.get_filter_by_name(">>unitary>>").get()

    # --- 2. RUN YOUR TROTTER ---
    qpu_mine = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    psi_mine = Qubits(qpu=qpu_mine, num_qubits=num_qubits)
    
    # Using your first_order_trotter function and the fixed PPR
    my_ppr = PPR()
    first_order_trotter(ham, psi_mine, my_ppr, evo_time, steps)
    
    matrix_mine = qpu_mine.get_filter_by_name(">>unitary>>").get()

    # --- 3. COMPARE ---
    # Compute fidelity between the two unitaries
    inner_prod = np.trace(matrix_wb.conj().T @ matrix_mine)
    fidelity = np.abs(inner_prod) / (2**num_qubits)
    
    print(f"Comparison Fidelity: {fidelity:.10f}")
    print(f"Matrix WB (0,0): {matrix_wb[0,0]}")
    print(f"Matrix Mine (0,0): {matrix_mine[0,0]}")
    
    # If this is 1.0, your implementation is functionally identical to Workbench
    assert np.isclose(fidelity, 1.0, atol=1e-8), "Implementation differs from TrotterQuery!"

def test_trotter_error_scaling():
    """
    Mathematically verify Trotter order by checking error convergence rates.
    1st order should decrease linearly; 2nd order should decrease quadratically.
    """
    num_qubits = 2
    # Use non-commuting terms to ensure Trotter error actually exists
    hamiltonian = create_hamiltonian_from_terms([
        (1.0, 0b01, 0b00),  # X0
        (1.0, 0b00, 0b11),  # Z0Z1
    ])
    
    time = 0.2
    exact_matrix = exact_time_evolution(hamiltonian, num_qubits, time)
    
    steps_list = [2, 4, 8, 16, 32]
    errors_1st = []
    errors_2nd = []

    for N in steps_list:
        # Measure 1st Order Error
        q1 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
        first_order_trotter(hamiltonian, Qubits(qpu=q1, num_qubits=num_qubits), PPR(), time, N)
        u1 = q1.get_filter_by_name(">>unitary>>").get()
        # Error = ||U_exact - U_trotter||
        errors_1st.append(np.linalg.norm(exact_matrix - u1))

        # Measure 2nd Order Error
        q2 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
        second_order_trotter(hamiltonian, Qubits(qpu=q2, num_qubits=num_qubits), PPR(), time, N)
        u2 = q2.get_filter_by_name(">>unitary>>").get()
        errors_2nd.append(np.linalg.norm(exact_matrix - u2))

    # Calculate convergence slopes (log-log)
    slope1 = np.polyfit(np.log(steps_list), np.log(errors_1st), 1)[0]
    slope2 = np.polyfit(np.log(steps_list), np.log(errors_2nd), 1)[0]

    print(f"1st Order Empirical Slope: {slope1:.2f} (Target: -1.0)")
    print(f"2nd Order Empirical Slope: {slope2:.2f} (Target: -2.0)")

    # Assertions: Slopes should be near -1 and -2 respectively
    assert -1.2 < slope1 < -0.8, "First order convergence rate is incorrect."
    assert -2.2 < slope2 < -1.8, "Second order convergence rate is incorrect."

def test_commuting_hamiltonian_exactness():
    """Verify that Trotter is exact (fidelity=1) for commuting terms."""
    num_qubits = 3
    # All Z-basis terms commute
    hamiltonian = create_hamiltonian_from_terms([
        (0.5, 0b000, 0b011), # Z0Z1
        (0.8, 0b000, 0b110), # Z1Z2
        (-0.3, 0b000, 0b101),# Z0Z2
    ])
    
    time = 1.5
    # Even with 1 step, it should be exact
    qpu = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    first_order_trotter(hamiltonian, Qubits(qpu=qpu, num_qubits=num_qubits), PPR(), time, 1)
    
    trotter_u = qpu.get_filter_by_name(">>unitary>>").get()
    exact_u = exact_time_evolution(hamiltonian, num_qubits, time)
    
    fidelity = compute_fidelity(trotter_u, exact_u)
    print(f"Commuting Hamiltonian Fidelity (N=1): {fidelity:.12f}")
    assert np.isclose(fidelity, 1.0, atol=1e-10)

def test_energy_conservation():
    """Verify that the expectation value of H is conserved during Trotter evolution."""
    num_qubits = 2
    hamiltonian = create_hamiltonian_from_terms([(1.0, 0b11, 0b00), (0.5, 0b00, 0b11)])
    H_mat = hamiltonian_to_matrix(hamiltonian, num_qubits)
    
    # Start with a random state
    state = np.random.rand(2**num_qubits) + 1j*np.random.rand(2**num_qubits)
    state /= np.linalg.norm(state)
    
    initial_energy = np.real(state.conj().T @ H_mat @ state)
    
    qpu = QPU(num_qubits=num_qubits)
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
    qubits.push_state(state)
    
    # Evolve
    second_order_trotter(hamiltonian, qubits, PPR(), time=1.0, num_trotter_steps=20)
    final_state = qpu.pull_state()
    
    final_energy = np.real(final_state.conj().T @ H_mat @ final_state)
    print(f"Initial Energy: {initial_energy:.6f}, Final Energy: {final_energy:.6f}")
    assert np.isclose(initial_energy, final_energy, atol=1e-8)

def create_heisenberg_xxx(num_qubits: int, J: float = 1.0) -> PauliSum:
    terms = []
    for i in range(num_qubits - 1):
        # Interaction between qubit i and i+1
        # X_i X_{i+1}
        terms.append((J, 1 << i | 1 << (i+1), 0))
        # Y_i Y_{i+1} (X and Z masks are both 1 at those positions)
        mask = 1 << i | 1 << (i+1)
        terms.append((J, mask, mask))
        # Z_i Z_{i+1}
        terms.append((J, 0, 1 << i | 1 << (i+1)))
    
    return create_hamiltonian_from_terms(terms)

def test_heisenberg_xxx_simulation():
    """Verify Trotter simulation of a 4-qubit Heisenberg XXX chain."""
    num_qubits = 4
    J = 1.0
    time = 0.5
    num_steps = 20
    
    hamiltonian = create_heisenberg_xxx(num_qubits, J)
    
    # 1. Exact Evolution
    exact_u = exact_time_evolution(hamiltonian, num_qubits, time)
    
    # 2. First-Order Trotter
    q1 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    first_order_trotter(hamiltonian, Qubits(qpu=q1, num_qubits=num_qubits), PPR(), time, num_steps)
    u1 = q1.get_filter_by_name(">>unitary>>").get()
    fid1 = compute_fidelity(u1, exact_u)
    
    # 3. Second-Order Trotter
    q2 = QPU(num_qubits=num_qubits, filters=">>unitary>>")
    second_order_trotter(hamiltonian, Qubits(qpu=q2, num_qubits=num_qubits), PPR(), time, num_steps)
    u2 = q2.get_filter_by_name(">>unitary>>").get()
    fid2 = compute_fidelity(u2, exact_u)
    
    print(f"Heisenberg XXX (L={num_qubits}, t={time})")
    print(f"  1st Order Fidelity (N={num_steps}): {fid1:.8f}")
    print(f"  2nd Order Fidelity (N={num_steps}): {fid2:.8f}")
    
    assert fid2 > fid1, "2nd order should be more accurate for Heisenberg model"
    assert fid2 > 0.999, "Second order fidelity is lower than expected for Heisenberg"

def test_heisenberg_magnetization_conservation():
    """Verify that total Z-magnetization is conserved in the Heisenberg model."""
    num_qubits = 4
    hamiltonian = create_heisenberg_xxx(num_qubits)
    
    # Magnetization operator Mz = Z0 + Z1 + Z2 + Z3
    Mz_terms = [(1.0, 0, 1 << i) for i in range(num_qubits)]
    Mz_mat = hamiltonian_to_matrix(create_hamiltonian_from_terms(Mz_terms), num_qubits)
    
    # Start in a state with known magnetization (e.g., |0101>)
    state = np.zeros(2**num_qubits)
    state[0b0101] = 1.0 
    
    initial_mz = np.real(state.conj().T @ Mz_mat @ state)
    
    qpu = QPU(num_qubits=num_qubits)
    qubits = Qubits(qpu=qpu, num_qubits=num_qubits)
    qubits.push_state(state)
    
    # Evolve under Heisenberg
    second_order_trotter(hamiltonian, qubits, PPR(), time=1.0, num_trotter_steps=50)
    
    final_state = qpu.pull_state()
    final_mz = np.real(final_state.conj().T @ Mz_mat @ final_state)
    
    print(f"Initial Mz: {initial_mz}, Final Mz: {final_mz:.6f}")
    # Note: Small errors are expected due to Trotterization, but it should be very close
    assert np.isclose(initial_mz, final_mz, atol=1e-5)

if __name__ == "__main__":
    # Run tests manually for debugging

    print("Testing against Workbench official TrotterQuery...")
    test_compare_with_workbench()
    print("✓ Passed\n")
    
    print("Testing first-order Trotter...")
    test_first_order_simple()
    print("✓ Passed\n")
    
    print("Testing second-order Trotter...")
    test_second_order_simple()
    print("✓ Passed\n")
    
    print("Testing second-order vs first-order...")
    test_second_order_better_than_first()
    print("✓ Passed\n")
    
    print("Testing convergence...")
    test_convergence_with_trotter_steps()
    print("✓ Passed\n")
    
    print("Testing interface...")
    test_trotter_evolution_interface()
    print("✓ Passed\n")
    
    print("Testing Transverse-Field Ising Model...")
    test_transverse_field_ising()
    print("✓ Passed\n")
    
    print("All tests passed!")