import numpy as np
from psiqworkbench import Qubits, Qubrick, Units

class PPR(Qubrick):
    def _compute(self, qubits: Qubits, theta, x_mask: int, z_mask: int):
        qubits_in_masks = x_mask | z_mask
        if qubits_in_masks == 0: 
            return

        #Standardize Angle 
        if hasattr(theta, 'to'): 
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

        ppr_qpu = qubits.qpu
        y_mask = x_mask & z_mask
        
        if y_mask:
            ppr_qpu.s_inv(y_mask)
        if x_mask:
            ppr_qpu.had(x_mask)

        target = qubits_in_masks.bit_length() - 1
        controls_mask = qubits_in_masks & ~(1 << target)
        
        uncomputation_controls = []
        temp_mask = controls_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            control_idx = lsb.bit_length() - 1
            qubits[target].x(cond=qubits[control_idx])
            uncomputation_controls.append(control_idx)
            temp_mask ^= lsb

        qubits[target].rz(2.0 * angle_deg)

        for control_idx in reversed(uncomputation_controls):
            qubits[target].x(cond=qubits[control_idx])

        if x_mask:
            ppr_qpu.had(x_mask)
        if y_mask:
            ppr_qpu.s(y_mask)