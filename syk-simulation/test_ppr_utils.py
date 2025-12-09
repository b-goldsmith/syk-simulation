import pytest
import numpy as np
from psiqworkbench import QPU, Qubits
from workbench_algorithms.utils.paulimask import PauliMask, PauliSum
from workbench_algorithms.utils.hamiltonian_utils import generate_h2_hamiltonian



def test_pauli_string_to_masks():
    """Test conversion of Pauli strings to masks."""
    
    x_mask, z_mask = pauli_string_to_masks("X0 X1")
    assert x_mask == 0b11
    assert z_mask == 0b00
    
    
    x_mask, z_mask = pauli_string_to_masks("Z0 Z1")
    assert x_mask == 0b00
    assert z_mask == 0b11
    
    # test X0 Y1 (Y = XZ)
    x_mask, z_mask = pauli_string_to_masks("X0 Y1")
    assert x_mask == 0b11
    assert z_mask == 0b10
    
    # test compact format
    x_mask, z_mask = pauli_string_to_masks("XYZI")
    assert x_mask == 0b0011
    assert z_mask == 0b0110


def test_get_ppr_args_from_ham_term():
    """Test extraction of PPR args from Hamiltonian term."""
    term = (0.5, PauliMask(3, 2))
    
    theta, x_mask, z_mask = get_ppr_args_from_ham_term(term)
    
    assert theta == 0.5
    assert x_mask == 3
    assert z_mask == 2


def test_masks_match_workbench():
    """Verify our masks match workbench's PauliMask format."""
    
    wb_mask = PauliMask.from_pauli_string("X0 Y1 Z2")
    our_x_mask, our_z_mask = pauli_string_to_masks("X0 Y1 Z2")
    
    assert wb_mask[0] == our_x_mask
    assert wb_mask[1] == our_z_mask