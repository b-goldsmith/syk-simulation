"""
Trotterization algorithms for quantum Hamiltonian simulation.

This module implements first-order and second-order Trotter-Suzuki decomposition
for approximating time evolution under a Hamiltonian H = Σ cⱼ Pⱼ.
"""

from psiqworkbench import Qubits
from workbench_algorithms.utils.paulimask import PauliSum


def apply_hamiltonian_as_pprs(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time_step: float
) -> None:
    """
    Apply all terms of a Hamiltonian as PPR operations in forward order.
    
    Applies terms in order: P₁, P₂, ..., Pₘ
    
    Args:
        hamiltonian: PauliSum Hamiltonian H = Σ cⱼ Pⱼ
        qubits: Qubits to apply operations on
        ppr_instance: PPR object for applying rotations
        time_step: Time step dt for evolution (each term gets e^(-i·cⱼ·Pⱼ·dt))
    """
    for i in range(len(hamiltonian)):
        coeff = hamiltonian.get_coefficient(i)
        mask = hamiltonian.get_mask(i)
        
        x_mask = mask[0]
        z_mask = mask[1]
        
        # Skip identity terms (no gates needed)
        qubits_in_masks = x_mask | z_mask
        if qubits_in_masks == 0:
            continue
        
        # Apply e^(-i·coeff·P·dt)
        theta = coeff * time_step
        ppr_instance.compute(qubits, theta=theta, x_mask=x_mask, z_mask=z_mask)


def apply_hamiltonian_as_pprs_reversed(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time_step: float
) -> None:
    """
    Apply all terms of a Hamiltonian as PPR operations in reverse order.
    
    Applies terms in order: Pₘ, ..., P₂, P₁
    
    This is critical for second-order Trotter to achieve symmetric splitting.
    
    Args:
        hamiltonian: PauliSum Hamiltonian H = Σ cⱼ Pⱼ
        qubits: Qubits to apply operations on
        ppr_instance: PPR object for applying rotations
        time_step: Time step dt for evolution
    """
    # Apply terms in reverse order
    for i in range(len(hamiltonian) - 1, -1, -1):
        coeff = hamiltonian.get_coefficient(i)
        mask = hamiltonian.get_mask(i)
        
        x_mask = mask[0]
        z_mask = mask[1]
        
        # Skip identity terms
        qubits_in_masks = x_mask | z_mask
        if qubits_in_masks == 0:
            continue
        
        # Apply e^(-i·coeff·P·dt)
        theta = coeff * time_step
        ppr_instance.compute(qubits, theta=theta, x_mask=x_mask, z_mask=z_mask)


def first_order_trotter(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time: float,
    num_trotter_steps: int
) -> None:
    """
    First-order Trotter-Suzuki decomposition.
    
    Approximates e^(-iHt) as:
        (∏ⱼ₌₁ᴹ e^(-iPⱼt/N))^N
    
    where N is the number of Trotter steps.
    
    Error: O(t²/N)
    
    Args:
        hamiltonian: PauliSum Hamiltonian H = Σ cⱼ Pⱼ
        qubits: Qubits to evolve
        ppr_instance: PPR object for applying rotations
        time: Total evolution time t
        num_trotter_steps: Number of Trotter steps N
        
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
        >>> first_order_trotter(hamiltonian, qubits, ppr, time=1.0, num_trotter_steps=10)
    """
    dt = time / num_trotter_steps
    
    for _ in range(num_trotter_steps):
        apply_hamiltonian_as_pprs(hamiltonian, qubits, ppr_instance, dt)


def second_order_trotter(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time: float,
    num_trotter_steps: int
) -> None:
    """
    Second-order Trotter-Suzuki decomposition.
    
    Approximates e^(-iHt) as:
        (∏ₖ₌₁ᴹ e^(-iPₖt/2N) · ∏ⱼ₌ᴹ¹ e^(-iPⱼt/2N))^N
    
    This uses symmetric splitting (forward then backward) for better accuracy.
    
    Error: O(t³/N²) - much better than first-order!
    
    Args:
        hamiltonian: PauliSum Hamiltonian H = Σ cⱼ Pⱼ
        qubits: Qubits to evolve
        ppr_instance: PPR object for applying rotations
        time: Total evolution time t
        num_trotter_steps: Number of Trotter steps N
        
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
        >>> second_order_trotter(hamiltonian, qubits, ppr, time=1.0, num_trotter_steps=10)
    """
    # Time step is t/(2N) because we apply forward AND backward each step
    dt = time / (2 * num_trotter_steps)
    
    for _ in range(num_trotter_steps):
        # Forward sweep: P₁, P₂, ..., Pₘ
        apply_hamiltonian_as_pprs(hamiltonian, qubits, ppr_instance, dt)
        
        # Backward sweep: Pₘ, ..., P₂, P₁
        apply_hamiltonian_as_pprs_reversed(hamiltonian, qubits, ppr_instance, dt)


def trotter_evolution(
    hamiltonian: PauliSum,
    qubits: Qubits,
    ppr_instance,
    time: float,
    num_trotter_steps: int,
    order: int = 2
) -> None:
    """
    General Trotter evolution interface.
    
    Args:
        hamiltonian: PauliSum Hamiltonian
        qubits: Qubits to evolve
        ppr_instance: PPR object
        time: Total evolution time
        num_trotter_steps: Number of Trotter steps
        order: Trotter order (1 or 2)
        
    Raises:
        ValueError: If order is not 1 or 2
        
    Example:
        >>> # Use second-order by default
        >>> trotter_evolution(hamiltonian, qubits, ppr, time=1.0, num_trotter_steps=20)
        >>> 
        >>> # Or specify first-order
        >>> trotter_evolution(hamiltonian, qubits, ppr, time=1.0, num_trotter_steps=20, order=1)
    """
    if order == 1:
        first_order_trotter(hamiltonian, qubits, ppr_instance, time, num_trotter_steps)
    elif order == 2:
        second_order_trotter(hamiltonian, qubits, ppr_instance, time, num_trotter_steps)
    else:
        raise ValueError(f"Order must be 1 or 2, got {order}")

