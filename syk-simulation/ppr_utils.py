import numpy as np
from workbench_algorithms.utils.paulimask import PauliSum, PauliMask


def pauli_string_to_masks(pauli_string: str) -> tuple[int, int]:
    """
    Convert a Pauli string to x_mask and z_mask for PPR.
    
    Args:
        pauli_string: String like "X0 Y1 Z2" or "XYZI"
        
    Returns:
        tuple: (x_mask, z_mask) as integers
        
    Example:
        >>> pauli_string_to_masks("X0 Y1")
        (3, 2)  # x_mask=0b11, z_mask=0b10
    """
    # handle both formats: "X0 Y1 Z2" and "XYZI"
    if ' ' in pauli_string:
        # Format: "X0 Y1 Z2"
        x_mask = 0
        z_mask = 0
        
        parts = pauli_string.strip().split()
        for part in parts:
            pauli = part[0]
            qubit_idx = int(part[1:])
            
            if pauli == 'X' or pauli == 'Y':
                x_mask |= (1 << qubit_idx)
            if pauli == 'Z' or pauli == 'Y':
                z_mask |= (1 << qubit_idx)
                
    else:
        # Format: "XYZI"
        x_mask = 0
        z_mask = 0
        
        for i, pauli in enumerate(reversed(pauli_string)):
            if pauli == 'X' or pauli == 'Y':
                x_mask |= (1 << i)
            if pauli == 'Z' or pauli == 'Y':
                z_mask |= (1 << i)
    
    return x_mask, z_mask


def get_ppr_args_from_ham_term(ham_term: tuple) -> list:
    """
    Extract PPR arguments from a Hamiltonian term.
    
    my version of workbench's get_ppr_args_from_ham function
    
    Args:
        ham_term: A tuple (coefficient, PauliMask) from a PauliSum
        
    Returns:
        list: [theta, x_mask, z_mask] ready for PPR.compute()
        
    Examples:
        >>> from workbench_algorithms.utils.paulimask import PauliMask
        >>> term = (0.5, PauliMask(3, 2))  # 0.5 * X0 Y1
        >>> get_ppr_args_from_ham_term(term)
        [0.5, 3, 2]
    """
    coefficient = ham_term[0]
    pauli_mask = ham_term[1]
    
    x_mask = pauli_mask[0]
    z_mask = pauli_mask[1]
    
    theta = coefficient
    
    return [theta, x_mask, z_mask]


def hamiltonian_to_ppr_terms(hamiltonian: PauliSum) -> list[dict]:
    """
    Convert a PauliSum Hamiltonian into a list of PPR terms.
    
    Args:
        hamiltonian: PauliSum object
        
    Returns:
        list of dicts, each containing:
            - 'theta': rotation angle (coefficient)
            - 'x_mask': x component mask
            - 'z_mask': z component mask
            - 'pauli_string': string representation (for debugging)
            
    Example:
        >>> from workbench_algorithms.utils.hamiltonian_utils import generate_h2_hamiltonian
        >>> ham = generate_h2_hamiltonian(0.7)
        >>> terms = hamiltonian_to_ppr_terms(ham)
        >>> print(f"Hamiltonian has {len(terms)} terms")
    """
    ppr_terms = []
    
    for i in range(len(hamiltonian)):
        coeff = hamiltonian.get_coefficient(i)
        mask = hamiltonian.get_mask(i)
        pauli_string = hamiltonian.get_pauli_string(i)
        
        x_mask = mask[0]
        z_mask = mask[1]
        
        ppr_terms.append({
            'theta': coeff,
            'x_mask': x_mask,
            'z_mask': z_mask,
            'pauli_string': pauli_string,
            'coefficient': coeff  
        })
    
    return ppr_terms


def apply_hamiltonian_as_pprs(
    hamiltonian: PauliSum,
    qubits,
    ppr_instance,
    time_step: float = 1.0
):
    """
    Apply all terms of a Hamiltonian as PPR operations.

    Skips pure-identity terms (x_mask == 0 and z_mask == 0) because PPR expects
    at least one target qubit (otherwise shifting by -1 occurs).
    """
    for i in range(len(hamiltonian)):
        coeff = hamiltonian.get_coefficient(i)
        mask = hamiltonian.get_mask(i)

        x_mask = mask[0]
        z_mask = mask[1]

        #if both masks are zero, this is the identity term -> skip it.
        qubits_in_masks = x_mask | z_mask
        if qubits_in_masks == 0:
            continue

        # scale by time_step for time evolution
        theta = coeff * time_step

        ppr_instance.compute(qubits, theta=theta, x_mask=x_mask, z_mask=z_mask)



def create_simple_hamiltonian(terms: list[tuple[float, str]]) -> PauliSum:
    """
    Create a PauliSum from a list of (coefficient, pauli_string) pairs
    
    Args:
        terms: List of (coefficient, pauli_string) tuples
        
    Returns:
        PauliSum object
        
    Example:
        >>> # Create H = 0.5*X0*X1 + 0.3*Z0*Z1
        >>> ham = create_simple_hamiltonian([
        ...     (0.5, "X0 X1"),
        ...     (0.3, "Z0 Z1")
        ... ])
    """
    pauli_terms = []
    
    for coeff, pauli_string in terms:
        x_mask, z_mask = pauli_string_to_masks(pauli_string)
        pauli_mask = PauliMask(x_mask, z_mask)
        pauli_terms.append([coeff, pauli_mask])
    
    return PauliSum(*pauli_terms)


def print_hamiltonian_ppr_decomposition(hamiltonian: PauliSum):
    """
    Print the PPR decomposition of a Hamiltonian
    
    Args:
        hamiltonian: PauliSum to analyze
    """
    print(f"Hamiltonian Decomposition:")
    print(f"  Number of terms: {len(hamiltonian)}")
    print(f"  Number of qubits: {hamiltonian.wires()}")
    print(f"  Norm (λ): {hamiltonian.norm():.6f}")
    print("\nPPR Terms:")
    
    for i in range(len(hamiltonian)):
        coeff = hamiltonian.get_coefficient(i)
        pauli_string = hamiltonian.get_pauli_string(i)
        mask = hamiltonian.get_mask(i)
        x_mask = mask[0]
        z_mask = mask[1]
        
        print(f"  Term {i}:")
        print(f"    Pauli String: {pauli_string}")
        print(f"    Coefficient:  {coeff:.6f}")
        print(f"    x_mask:       {x_mask:b} (0b{x_mask:b})")
        print(f"    z_mask:       {z_mask:b} (0b{z_mask:b})")
        print()