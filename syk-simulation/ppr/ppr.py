from psiqworkbench import Qubits, Qubrick, QPU, Units


class PPR(Qubrick):
    """This class implements Pauli Product Rotations (PPR) on a set of qubits."""

    def _compute(
        self,
        qubits: Qubits,
        theta: float | Units.RotationAngle | tuple[int, int],
        x_mask: int,
        z_mask: int,
    ):
        """Apply a Pauli Product Rotation on the specified qubits.

        Args:
            qubits (Qubits): The qubits to apply the rotation on.
            theta (float | RotationAngle | tuple[int, int]): The rotation angle.
            x_mask (int): Bitmask indicating which qubits have X in the Pauli product.
            z_mask (int): Bitmask indicating which qubits have Z in the Pauli product.
        """

        # Get active qubits from masks to determine target qubit
        qubits_in_masks = x_mask | z_mask
        target = qubits_in_masks.bit_length() - 1

        # If there are no active qubits break out of the function
        if qubits_in_masks == 0:
            return

        # get QPU from qubits to use QPU gates
        ppr_qpu = qubits.qpu

        # Adjust any qubits with Clifford gates to get them into Z basis
        ppr_qpu.s_inv(x_mask & z_mask)
        ppr_qpu.had(x_mask)

        uncomputation_controls = []

        # CNOT chain for Z parity (make sure more than 1 active qubit in masks)
        if qubits_in_masks > 1:
            controls_mask = qubits_in_masks & ~(1 << target)

            while controls_mask:
                lsb = controls_mask & -controls_mask
                q = lsb.bit_length() - 1
                uncomputation_controls.append(q)
                qubits[target].x(cond=qubits[q])
                controls_mask ^= lsb

        # Apply Rz rotation on the last qubit
        # qubits[target].rz(2.0*theta)
        if isinstance(theta, tuple):
            double_theta = (theta[0] * 2, theta[1])
        else:
            double_theta = 2 * theta
        qubits[target].rz(double_theta)

        # Uncompute CNOT chain from Z parity
        if qubits_in_masks > 1:
            for i in reversed(uncomputation_controls):
                qubits[target].x(cond=qubits[i])

        # Uncompute basis changes
        ppr_qpu.had(x_mask)
        ppr_qpu.s(x_mask & z_mask)
