"""This file contains the utilities used for implementing asymmetric qubitization from
https://arxiv.org/abs/2203.07303
"""

from psiqworkbench.qubricks import Reflect
from psiqworkbench import Qubits, Qubrick, Units

import numpy as np


class OracleA(Qubrick):
    """This class implements the oracle A for asymmetric qubitization."""

    def __init__(self, random_seed=None, **kwargs):
        super().__init__(**kwargs)
        if random_seed is None:
            random_seed = np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(random_seed)

    def _compute(self, index: Qubits, random_depth: int, ctrl: Qubits | None = None):
        """Implements the oracle A which prepares the state |A> = sum_j a_l |l>
        on the branch qubits and encodes the sign of a_j in the index qubits.

        Args:
            index (Qubits): The index qubits to encode the sign information.
        """
        for depth in range(random_depth):
            for q in index:
                theta = self.rng.uniform(0, 2 * np.pi)
                q.ry(theta * Units.rad, cond=ctrl)
            pattern = depth % 3
            if pattern == 0:
                start, wrap = 0, False
            elif pattern == 1:
                start, wrap = 1, False
            else:
                start, wrap = 0, True
            for i in range(start, len(index) - 1, 2):
                if self.rng.integers(2):
                    index[i].x(ctrl | index[i + 1])
                else:
                    index[i + 1].x(ctrl | index[i])
            if wrap:
                index[-1].x(ctrl | index[0])


class OracleB(Qubrick):
    """Implements the oracle B which prepares the state |B> = sum_j sqrt(b_j/lambda) |j>
    on the branch qubits.

    Args:
        index (Qubits): The index qubits to prepare the state on.
    """

    def _compute(self, index: Qubits, ctrl: Qubits | None = None):
        index.had(cond=ctrl)


class MajoranaOperator(Qubrick):
    """Implements a single Majorana operator γ_ℓ via unary iteration.

    Applies X_ℓ · Z_{ℓ-1} · Z_{ℓ-2} · ... · Z_0 to the system register,
    controlled on the index register encoding ℓ in binary.
    Based on Figure 1 of Babbush et al. 2019.

    Args:
        system (Qubits): The system qubits to apply the Majorana operator on.
        indices (Qubits): The index qubits encoding ℓ in binary.
        ctrl (Qubits): Optional control qubits.
    """

    def _compute(self, system: Qubits, indices: Qubits, ctrl: Qubits | None = None):
        auxiliary = self.alloc_temp_qreg(len(indices), "unary_aux", release_after_compute=True)
        range_flag = self.alloc_temp_qreg(1, "unary_range", release_after_compute=True)
        system_index = 0

        def apply_majorana_operation(aux_index):
            nonlocal system_index

            if aux_index == len(indices) - 1:
                range_flag.x(ctrl)
                lelbow_control = ctrl | ~indices[aux_index]
                relbow_control = ctrl | indices[aux_index]
                x_control = ctrl
            else:
                lelbow_control = auxiliary[aux_index + 1] | ~indices[aux_index]
                relbow_control = auxiliary[aux_index + 1] | indices[aux_index]
                x_control = auxiliary[aux_index + 1]

            if aux_index == 0:
                auxiliary[aux_index].lelbow(cond=lelbow_control)
                range_flag.x(cond=auxiliary[aux_index])
                if system_index < len(system):
                    system[system_index].x(cond=auxiliary[aux_index])
                    system[system_index].z(cond=range_flag)
                system_index = system_index + 1
                auxiliary[aux_index].x(cond=auxiliary[aux_index + 1])
                range_flag.x(cond=auxiliary[aux_index])
                if system_index < len(system):
                    system[system_index].x(cond=auxiliary[aux_index])
                auxiliary[aux_index].relbow(cond=relbow_control)
                return

            auxiliary[aux_index].lelbow(cond=lelbow_control)
            apply_majorana_operation(aux_index - 1)
            if system_index < len(system):
                system[system_index].z(cond=range_flag)
            system_index = system_index + 1
            auxiliary[aux_index].x(cond=x_control)
            apply_majorana_operation(aux_index - 1)
            auxiliary[aux_index].relbow(cond=relbow_control)

        apply_majorana_operation(len(indices) - 1)


class Select(Qubrick):
    """This class implements the SELECT operation for asymmetric qubitization."""

    def _compute(self, index: Qubits, system: Qubits, ctrl: Qubits | None = None):
        """Apply the SELECT operation on the given qubits.

        Args:
            index (Qubits): The index qubits used for unary iteration
            system (Qubits): The system qubits to apply the Pauli terms on.
        """

        index_chunk = len(index) // 4

        p = index[0:index_chunk]
        q = index[index_chunk : 2 * index_chunk]
        r = index[2 * index_chunk : 3 * index_chunk]
        s = index[3 * index_chunk :]

        majoranaOperator = MajoranaOperator()
        majoranaOperator.compute(system, p, ctrl=ctrl)
        majoranaOperator.compute(system, q, ctrl=ctrl)
        majoranaOperator.compute(system, r, ctrl=ctrl)
        majoranaOperator.compute(system, s, ctrl=ctrl)


class AsymmetricQubitization(Qubrick):
    """This class implements asymmetric qubitization of the SYK model"""

    def __init__(self, random_seed=None, **kwargs):
        super().__init__(**kwargs)
        if random_seed is None:
            random_seed = np.random.SeedSequence().entropy
        self.random_seed = random_seed

    def _compute(
        self,
        branch: Qubits,
        index: Qubits,
        system: Qubits,
        ctrl: Qubits | None = None,
    ):
        """Apply asymmetric qubitization on the given qubits.
        Args:
            branch (Qubits): The branch qubits for oracle B.
            index (Qubits): The index qubits for unary iteration
            system (Qubits): The system qubits to apply the Hamiltonian on.
            select_class: This allows a user to provide a `Select` class and defaults to the optimized select
        """

        oracleA = OracleA(random_seed=self.random_seed)
        oracleB = OracleB()
        select = Select()
        reflect = Reflect()

        # Set random_depth based on empircal testing of test_oraclea(N)
        N = len(system)
        n = len(index)
        random_depth = 2 * n

        # Run PREPARE for qubitization
        oracleA.compute(index=index, random_depth=random_depth, ctrl=(ctrl | ~branch))
        oracleB.compute(index=index, ctrl=(ctrl | branch))

        # Run SELECT for qubitization
        select.compute(index=index, system=system, ctrl=ctrl)

        # NOT on branch
        branch.x(cond=ctrl)

        # Run UNPREPARE for qubitization
        oracleB.uncompute()
        oracleA.uncompute()

        # We have constructed U but we need to do the reflection

        reflect.compute(target_qreg=branch, ctrl=(ctrl | index))
