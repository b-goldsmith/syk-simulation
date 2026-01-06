"""This file contains the utilities used for implementing asymmetric qubitization from
https://arxiv.org/abs/2203.07303
"""

from psiqworkbench import Qubits, Qubrick
from ..ppr import PPR


class AsymmetricQubitization(Qubrick):
    """This class implements asymmetric qubitization of the SYK model"""

    def _compute(self, branch: Qubits, index: Qubits, system: Qubits, depth: int = 5, terms: list[str] = None):
        """Apply asymmetric qubitization on the given qubits.
        Args:
            branch (Qubits): The branch qubits for oracle B.
            index (Qubits): The index qubits for oracle A.
            system (Qubits): The system qubits to apply the Hamiltonian on.
            depth (int): The depth of the oracle A preparation.
            terms (list[str]): The list of Pauli terms to apply in the SELECT operation.
        """
        quant_qpu = branch.qpu
        quant_qpu.had(branch)

        # Run PREPARE for qubitization
        oracleA = OracleA()
        oracleB = OracleB()
        oracleA.compute(cond=branch == 0, index=index, depth=depth)
        oracleB.compute(cond=branch == 1, index=index)

        # Run SELECT for qubitization
        for idx, pauli_string in enumerate(terms):
            # TODO x_mask, z_mask = NISHNA_CODE.pauli_string_to_masks(pauli_string)
            x_mask, z_mask = 0b10101, 0b01101
            quant_qpu.x(x_mask, cond=index == idx)
            quant_qpu.z(z_mask, cond=index == idx)

        # Run UNPREPARE for qubitization
        oracleA.uncompute(cond=branch == 0, index=index, depth=depth)
        oracleB.uncompute(cond=branch == 1, index=index)

        # We have constructed U but we need to do the reflection
        quant_qpu.x(branch)
        quant_qpu.x(index)
        quant_qpu.z(index, cond=branch == 1)
        quant_qpu.x(branch)
        quant_qpu.x(index)


class OracleA(Qubrick):
    """This class implements the oracle A for asymmetric qubitization."""

    def _compute(self, index: Qubits, depth: int = 5, ctrl: Qubits | None = None):
        """Implements the oracle A which prepares the state |A> = sum_j sqrt(|a_j|/lambda) |j>
        on the branch qubits and encodes the sign of a_j in the index qubits.

        Args:
            index (Qubits): The index qubits to encode the sign information.
        """
        for _ in range(depth):
            for q in index:
                theta = np.random.normal()
                # TODO: Clean up qpu with something from self versus using index
                index.qpu.ry(theta, q)

            for i in range(len(index) - 1):
                index.qpu.x(index[i], index[i + 1])


class OracleB(Qubrick):
    """Implements the oracle B which prepares the state |B> = sum_j sqrt(b_j/lambda) |j>
    on the branch qubits.

    Args:
        index (Qubits): The index qubits to prepare the state on.
    """

    def _compute(self, index: Qubits, ctrl: Qubits | None = None):
        for q in index:
            index.qpu.had(q)
