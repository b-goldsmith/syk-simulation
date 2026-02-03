import numpy as np
from psiqworkbench import Qubits, Qubrick, Units


class PPR(Qubrick):
    def _compute(self, qubits: Qubits, theta, x_mask: int, z_mask: int, ctrl: Qubits | None = None):
        """Apply a Pauli Product Rotation on the specified qubits.

        Args:
            qubits (Qubits): The qubits to apply the rotation on.
            theta (float | RotationAngle | tuple[int, int]): The rotation angle.
            x_mask (int): Bitmask indicating which qubits have X in the Pauli product.
            z_mask (int): Bitmask indicating which qubits have Z in the Pauli product.
        """
        # Get active qubits from masks to determine target qubit
        qubits_in_masks = x_mask | z_mask

        # Needed for QPU gate calls with masks
        ctrl_mask = 0
        if ctrl is not None:
            ctrl_mask = ctrl.mask()

        # If there are no active qubits break out of the function
        if qubits_in_masks == 0:
            return

        # Standardize Angle
        if hasattr(theta, "to"):
            angle_deg = theta.to("deg").mag
        elif isinstance(theta, tuple):
            # Convert fraction of pi to degrees
            angle_deg = np.rad2deg(np.pi * theta[0] / theta[1])
        else:
            # Raw floats: assume degrees if they are > 2pi, else radians
            if abs(theta) > 2 * np.pi:
                angle_deg = theta
            else:
                angle_deg = np.rad2deg(theta)

        # get QPU from qubits to use QPU gates
        ppr_qpu = qubits.qpu
        y_mask = x_mask & z_mask

        # Adjust any qubits with Clifford gates to get them into Z basis
        if y_mask:
            ppr_qpu.s_inv(y_mask, condition_mask=ctrl_mask)
        if x_mask:
            ppr_qpu.had(x_mask, condition_mask=ctrl_mask)

        target = qubits_in_masks.bit_length() - 1
        controls_mask = qubits_in_masks & ~(1 << target)

        uncomputation_controls = []
        temp_mask = controls_mask

        # CNOT chain for Z parity (make sure more than 1 active qubit in masks)
        while temp_mask:
            lsb = temp_mask & -temp_mask
            control_idx = lsb.bit_length() - 1
            qubits[target].x(cond=(ctrl | qubits[control_idx]))
            uncomputation_controls.append(control_idx)
            temp_mask ^= lsb

        # Apply Rz rotation on the last qubit
        qubits[target].rz(2.0 * angle_deg)

        # Uncompute CNOT chain from Z parity
        for control_idx in reversed(uncomputation_controls):
            qubits[target].x(cond=(ctrl | qubits[control_idx]))

        # Uncompute basis changes
        if x_mask:
            ppr_qpu.had(x_mask, condition_mask=ctrl_mask)
        if y_mask:
            ppr_qpu.s(y_mask, condition_mask=ctrl_mask)
