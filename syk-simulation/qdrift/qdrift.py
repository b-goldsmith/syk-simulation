"""
qDRIFT algorithm for quantum Hamiltonian simulation.

qDRIFT is a randomized alternative to Trotter that samples Hamiltonian terms
according to their coefficients. This can be more efficient for sparse
Hamiltonians with many terms.

"""

import numpy as np
from psiqworkbench import Qubits
from workbench_algorithms.utils.paulimask import PauliSum
from utils import sample_distribution


def qdrift(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time: float,
    num_samples: int,
    random_seed: int | None = None
) -> None:
    """
    qDRIFT algorithm for Hamiltonian simulation.
    
    Instead of deterministically applying all Hamiltonian terms (like Trotter),
    qDRIFT randomly samples terms with probability proportional to their
    coefficients.
    
    Algorithm:
        1. Compute λ = Σⱼ |cⱼ| (sum of absolute coefficients)
        2. For each of N samples:
           a. Sample term j with probability |cⱼ|/λ
           b. Apply e^(-i·sign(cⱼ)·λ·Pⱼ·t/N)
    
    Args:
        hamiltonian: PauliSum Hamiltonian H = Σ cⱼ Pⱼ
        qubits: Qubits to evolve
        ppr_instance: PPR object for applying rotations
        time: Total evolution time t
        num_samples: Number of random samples N
        random_seed: Optional random seed for reproducibility
        
    Note:
        The number of samples N should be chosen based on desired accuracy ε:
            N = O((λt)² / ε²)
        where λ = Σⱼ |cⱼ|
        
    Example:
        >>> from workbench_algorithms.utils.paulimask import PauliSum, PauliMask
        >>> from psiqworkbench import QPU, Qubits
        >>> from syksimulation.ppr.ppr import PPR
        >>> 
        >>> # H = 0.5·X₀X₁ + 0.3·Z₀Z₁
        >>> hamiltonian = PauliSum(
        ...     [0.5, PauliMask(0b11, 0b00)],
        ...     [0.3, PauliMask(0b00, 0b11)]
        ... )
        >>> 
        >>> qpu = QPU(num_qubits=2)
        >>> qubits = Qubits(qpu=qpu, num_qubits=2)
        >>> ppr = PPR()
        >>> 
        >>> # Evolve using qDRIFT
        >>> qdrift(hamiltonian, qubits, ppr, time=1.0, num_samples=100)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract coefficients from Hamiltonian
    coefficients = [hamiltonian.get_coefficient(i) for i in range(len(hamiltonian))]
    
    # Compute λ = Σⱼ |cⱼ|
    lambda_norm = sum(abs(c) for c in coefficients)
    
    if lambda_norm == 0:
        # Empty or zero Hamiltonian - nothing to do
        return
    
    # Time step for each sample
    dt = time / num_samples
    
    # Perform N random samples
    for _ in range(num_samples):
        # Sample term j with probability |cⱼ|/λ
        j = sample_distribution(coefficients)
        
        # Get the Pauli masks for term j
        mask = hamiltonian.get_mask(j)
        x_mask = mask[0]
        z_mask = mask[1]
        
        # Skip identity terms
        qubits_in_masks = x_mask | z_mask
        if qubits_in_masks == 0:
            continue
        
        # Apply e^(-i·sign(cⱼ)·λ·Pⱼ·dt)
        # Note: We apply with angle sign(cⱼ)·λ·dt
        coeff = coefficients[j]
        theta = np.sign(coeff) * lambda_norm * dt
        
        ppr_instance.compute(qubits, theta=theta, x_mask=x_mask, z_mask=z_mask)


def qdrift_with_epsilon(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time: float,
    epsilon: float,
    random_seed: int | None = None
) -> int:
    """
    qDRIFT algorithm with automatic sample count based on desired accuracy.
    
    Computes the required number of samples N to achieve accuracy ε using:
        N = 2 * (λt)² / ε²
    
    where λ = Σⱼ |cⱼ| is the 1-norm of the Hamiltonian.
    
    Args:
        hamiltonian: PauliSum Hamiltonian H = Σ cⱼ Pⱼ
        qubits: Qubits to evolve
        ppr_instance: PPR object for applying rotations
        time: Total evolution time t
        epsilon: Desired accuracy ε
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Number of samples used
        
    Example:
        >>> # Automatically determine N for ε = 0.01 accuracy
        >>> num_samples = qdrift_with_epsilon(
        ...     hamiltonian, qubits, ppr, time=1.0, epsilon=0.01
        ... )
        >>> print(f"Used {num_samples} samples")
    """
    # Extract coefficients
    coefficients = [hamiltonian.get_coefficient(i) for i in range(len(hamiltonian))]
    
    # Compute λ = Σⱼ |cⱼ|
    lambda_norm = sum(abs(c) for c in coefficients)
    
    if lambda_norm == 0:
        return 0
    
    # Compute required number of samples
    # N = 2 * (λt)² / ε²
    num_samples = int(np.ceil(2 * (lambda_norm * time) ** 2 / (epsilon ** 2)))
    
    # Run qDRIFT with computed N
    qdrift(hamiltonian, qubits, ppr_instance, time, num_samples, random_seed)
    
    return num_samples


def qdrift_vs_trotter_cost(
    hamiltonian: PauliSum,
    time: float,
    epsilon: float
) -> dict:
    """
    Compare the gate cost of qDRIFT vs second-order Trotter for given accuracy.
    
    Args:
        hamiltonian: PauliSum Hamiltonian
        time: Evolution time
        epsilon: Desired accuracy
        
    Returns:
        Dictionary with cost comparison:
            {
                'qdrift_samples': int,
                'qdrift_gates': int,
                'trotter_steps': int,
                'trotter_gates': int,
                'advantage': float  # ratio of Trotter gates to qDRIFT gates
            }
    """
    # Extract coefficients
    coefficients = [hamiltonian.get_coefficient(i) for i in range(len(hamiltonian))]
    lambda_norm = sum(abs(c) for c in coefficients)
    num_terms = len(hamiltonian)
    
    # qDRIFT cost
    qdrift_samples = int(np.ceil(2 * (lambda_norm * time) ** 2 / (epsilon ** 2)))
    qdrift_gates = qdrift_samples  # Each sample = 1 PPR gate
    
    # Second-order Trotter cost (error ≈ (λt)³/(12N²))
    # Solving for N: N ≈ (λt)^(3/2) / sqrt(12ε)
    trotter_steps = int(np.ceil((lambda_norm * time) ** 1.5 / np.sqrt(12 * epsilon)))
    trotter_gates = 2 * num_terms * trotter_steps  # 2 sweeps × M terms × N steps
    
    return {
        'qdrift_samples': qdrift_samples,
        'qdrift_gates': qdrift_gates,
        'trotter_steps': trotter_steps,
        'trotter_gates': trotter_gates,
        'advantage': trotter_gates / qdrift_gates if qdrift_gates > 0 else 0
    }