"""This file contains the utilities used for implementing asymmetric qubitization from
https://arxiv.org/abs/2203.07303
"""

from workbench_algorithms.utils.paulimask import PauliMask, PauliSum
from psiqworkbench import Qubits, Qubrick
import numpy as np


class AsymmetricQubitization(Qubrick):
    """This class implements asymmetric qubitization of the SYK model"""

    def _compute(
        self,
        hamiltonian: PauliSum,
        branch: Qubits,
        index: Qubits,
        system: Qubits,
        depth: int = 5,
        ctrl: Qubits | None = None,
    ):
        """Apply asymmetric qubitization on the given qubits.
        Args:
            branch (Qubits): The branch qubits for oracle B.
            index (Qubits): The index qubits for oracle A.
            system (Qubits): The system qubits to apply the Hamiltonian on.
            depth (int): The depth of the oracle A preparation.
            terms (list[str]): The list of Pauli terms to apply in the SELECT operation.
        """

        terms = hamiltonian.get_pauli_strings()

        branch.had(ctrl=ctrl)

        oracleA = OracleA()
        oracleB = OracleB()
        select = Select()
        reflection = Reflection()

        # Run PREPARE for qubitization
        oracleA.compute(index=index, depth=depth, ctrl=(ctrl | branch == 0))
        oracleB.compute(index=index, ctrl=(ctrl | branch == 1))

        # Run SELECT for qubitization
        select.compute(index=index, system=system, terms=terms, ctrl=ctrl)

        # Run UNPREPARE for qubitization
        oracleB.uncompute()
        oracleA.uncompute()

        # We have constructed U but we need to do the reflection
        reflection.compute(branch=branch, index=index, ctrl=ctrl)


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
                q.ry(theta, cond=ctrl)
            for i in range(len(index) - 1):
                index[i].x(ctrl | index[i + 1])


class OracleB(Qubrick):
    """Implements the oracle B which prepares the state |B> = sum_j sqrt(b_j/lambda) |j>
    on the branch qubits.

    Args:
        index (Qubits): The index qubits to prepare the state on.
    """

    def _compute(self, index: Qubits, ctrl: Qubits | None = None):
        index.had(cond=ctrl)


class Select(Qubrick):
    """This class implements the SELECT operation for asymmetric qubitization."""

    def _compute(self, index: Qubits, system: Qubits, terms: list[str], ctrl: Qubits | None = None):
        """Apply the SELECT operation on the given qubits.

        Args:
            index (Qubits): The index qubits to select the Pauli terms.
            system (Qubits): The system qubits to apply the Pauli terms on.
            terms (list[str]): The list of Pauli terms to apply.
        """
        for idx, pauli_string in enumerate(terms):
            term = PauliMask.from_pauli_string(pauli_string)
            for pauli in term.get_indices():
                pauli_method = getattr(system[pauli], term.get_pauli(pauli).lower())
                pauli_method(cond=(ctrl | index == idx))


class Reflection(Qubrick):
    """This class implements the reflection about |0> for asymmetric qubitization."""

    def _compute(self, branch: Qubits, index: Qubits, ctrl: Qubits | None = None):
        """Apply the reflection about |0> on the given qubits.

        Args:
            qubits (Qubits): The qubits to apply the reflection on.
        """
        branch.x(ctrl)
        index.x(ctrl)
        index.z(cond=(ctrl | branch == 1))
        branch.x(ctrl)
        index.x(ctrl)
