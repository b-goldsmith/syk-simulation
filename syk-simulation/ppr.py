from psiqworkbench import Qubits, Qubrick, QPU, Units

class PPR(Qubrick):
    """ This class implements Pauli Product Rotations (PPR) on a set of qubits."""
    def _compute(self, qubits: Qubits, theta: float | Units.RotationAngle | tuple[int, int], x_mask: int, z_mask: int):
        """ Apply a Pauli Product Rotation on the specified qubits.

        Args:
            qubits (Qubits): The qubits to apply the rotation on.
            theta (float | RotationAngle | tuple[int, int]): The rotation angle.
            x_mask (int): Bitmask indicating which qubits have X in the Pauli product.
            z_mask (int): Bitmask indicating which qubits have Z in the Pauli product.
        """

        # get QPU from qubits to use QPU gates
        ppr_qpu = qubits.qpu
        
        # Adjust any qubits with Clifford gates to get them into Z basis
        ppr_qpu.had(x_mask)
        ppr_qpu.s(x_mask & z_mask)


        # CNOT chain for Z parity
        for i in range(qubits.__len__() - 1):
            qubits[i].x(cond=qubits[qubits.__len__() -1])
        
        # Apply Rz rotation on the last qubit
        qubits[qubits.__len__() -1].rz(theta)

        # Uncompute CNOT chain
        for i in range(qubits.__len__() - 2, -1, -1):
            qubits[i].x(cond=qubits[qubits.__len__() -1])

        # Uncompute basis changes
        ppr_qpu.s_inv(x_mask & z_mask)
        ppr_qpu.had(x_mask)