"""This file contains the utilities used for implementing asymmetric qubitization from
https://arxiv.org/abs/2203.07303
"""

from psiqworkbench.qubricks import Reflect
from workbench_algorithms.utils.paulimask import PauliMask, PauliSum
from psiqworkbench import Qubits, Qubrick
import numpy as np


class AsymmetricQubitization(Qubrick):
    """This class implements asymmetric qubitization of the SYK model"""

    def _compute(
        self,
        branch: Qubits,
        index: Qubits,
        system: Qubits,
        random_depth: int = 2,
        random_seed: int | None = None,
        ctrl: Qubits | None = None,
    ):
        """Apply asymmetric qubitization on the given qubits.
        Args:
            N (int): Number of Majorana fermions.
            branch (Qubits): The branch qubits for oracle B.
            index (Qubits): The index qubits for oracle A.
            system (Qubits): The system qubits to apply the Hamiltonian on.
            depth (int): The depth of the oracle A preparation.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        branch.had(cond=ctrl)

        oracleA = OracleA()
        oracleB = OracleB()
        select = Select()
        reflect = Reflect()

        # Run PREPARE for qubitization
        oracleA.compute(index=index, random_depth=random_depth, ctrl=(ctrl | ~branch))
        oracleB.compute(index=index, ctrl=(ctrl | branch))

        # Run SELECT for qubitization
        select.compute(index=index, system=system, ctrl=ctrl)

        # Run UNPREPARE for qubitization
        oracleB.uncompute()
        oracleA.uncompute()

        # We have constructed U but we need to do the reflection

        reflect.compute(target_qreg=index, ctrl=(ctrl | branch))


class OracleA(Qubrick):
    """This class implements the oracle A for asymmetric qubitization."""

    def _compute(self, index: Qubits, random_depth: int = 5, ctrl: Qubits | None = None):
        """Implements the oracle A which prepares the state |A> = sum_j sqrt(|a_j|/lambda) |j>
        on the branch qubits and encodes the sign of a_j in the index qubits.

        Args:
            index (Qubits): The index qubits to encode the sign information.
        """
        for _ in range(random_depth):
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

    system_index: int

    def _compute(self, index: Qubits, system: Qubits, ctrl: Qubits | None = None):
        """Apply the SELECT operation on the given qubits.

        Args:
            index (Qubits): The index qubits used for unary iteration
            system (Qubits): The system qubits to apply the Pauli terms on.
        """

        range_flag = self.alloc_temp_qreg(1, "unary_range")
        index_chunk = len(index) // 4
        auxiliary = self.alloc_temp_qreg(index_chunk, "unary_aux")

        p = index[0:index_chunk]
        q = index[index_chunk : 2 * index_chunk]
        r = index[2 * index_chunk : 3 * index_chunk]
        s = index[3 * index_chunk :]

        def apply_majorana_operation(
            self, auxiliary, range_flag, indeces, system, aux_index, ctrl: Qubits | None = None
        ):
            # if at least significant bit (start) of index register : aux_index uses BIG ENDIAN
            if aux_index == len(indeces) - 1:
                range_flag.x(ctrl)
                lelbow_control = ctrl | ~indeces[aux_index]
                relbow_control = ctrl | indeces[aux_index]
                x_control = ctrl
            else:
                lelbow_control = auxiliary[aux_index + 1] | ~indeces[aux_index]
                relbow_control = auxiliary[aux_index + 1] | indeces[aux_index]
                x_control = auxiliary[aux_index + 1]

            if aux_index == 0:
                auxiliary[aux_index].lelbow(cond=lelbow_control)
                range_flag.x(cond=auxiliary[aux_index])
                system[self.system_index].x(cond=auxiliary[aux_index])
                system[self.system_index].z(cond=range_flag)
                if self.system_index != len(system) - 1:
                    self.system_index = self.system_index + 1
                    auxiliary[aux_index].x(cond=auxiliary[aux_index + 1])
                    range_flag.x(cond=auxiliary[aux_index])
                    system[self.system_index].x(cond=auxiliary[aux_index])
                auxiliary[aux_index].relbow(cond=relbow_control)
                return

            auxiliary[aux_index].lelbow(cond=lelbow_control)
            apply_majorana_operation(self, auxiliary, range_flag, indeces, system, aux_index - 1)
            if self.system_index != len(system) - 1:
                system[self.system_index].z(cond=range_flag)
                self.system_index = self.system_index + 1
                auxiliary[aux_index].x(cond=x_control)
                apply_majorana_operation(self, auxiliary, range_flag, indeces, system, aux_index - 1)
            auxiliary[aux_index].relbow(cond=relbow_control)

        self.system_index = 0
        apply_majorana_operation(self, auxiliary, range_flag, p, system, len(p) - 1, ctrl=ctrl)
        self.system_index = 0
        apply_majorana_operation(self, auxiliary, range_flag, q, system, len(q) - 1, ctrl=ctrl)
        self.system_index = 0
        apply_majorana_operation(self, auxiliary, range_flag, r, system, len(r) - 1, ctrl=ctrl)
        self.system_index = 0
        apply_majorana_operation(self, auxiliary, range_flag, s, system, len(s) - 1, ctrl=ctrl)
