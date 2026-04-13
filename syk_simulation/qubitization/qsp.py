"""This file contains the QSP algorithm for qubitization-based simulation of the SYK model."""

from psiqworkbench import Qubits, Qubrick, Units, QPU
import numpy as np
import math

# from pyqsp.poly import PolyCosineTX, PolySineTX
from pyqsp.poly import PolyTaylorSeries
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from syk_simulation.qubitization import AsymmetricQubitization

from syk_simulation.qubitization.pyqsp_poly import PolyCosineTX, PolySineTX

import warnings


def run_qsp(N: int, time: float, J: float, epsilon: float, random_seed: int = None, phases=None):
    # Qubit register sizes
    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1
    mode_size = 1
    selection_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size + mode_size + selection_size

    # Setup QPU and Qubit registers
    # (except for aux_unary and range_flag as they are auxiliary Qubrick qubits)
    qpu = QPU(num_qubits=num_qubits, filters=[">>state-vector-sim>>"])
    qpu.reset(num_qubits)

    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")
    mode = Qubits(mode_size, "mode", qpu=qpu)
    selection = Qubits(selection_size, "selection", qpu=qpu)

    # PREPROCESSING
    # 1) set branch and mode to |+>
    branch.had()

    qsp_evolution(
        N=N,
        J=1,
        branch=branch,
        index=index,
        system=system,
        mode=mode,
        selection=selection,
        time=time,
        epsilon=epsilon,
        random_seed=random_seed,
    )
    branch.had()

    if (
        np.isclose(
            branch.peek_read_probability(0),
            1,
        )
        and np.isclose(mode.peek_read_probability(0), 1)
        and np.isclose(selection.peek_read_probability(0), 1, atol=1e-3)
    ):
        return system.pull_state()
    else:
        warnings.warn("QSP did not evovle e**(-iHt)")


def qsp_evolution(
    N: int,
    J: float,
    branch: Qubits,
    index: Qubits,
    system: Qubits,
    mode: Qubits,
    selection: Qubits,
    time: float,
    epsilon: float = 1e-3,
    random_seed: int | None = None,
):
    """Perform qubitization-based QSP evolution for the SYK model.

    Args:
        N (int): Number of Majorana fermions.
        hamiltonian (PauliSum): The Hamiltonian to be simulated.
        time (float): The time for simulation.
        epsilon (float): The desired precision.
    """

    lambda_ = N ** (5 / 2) * J * np.sqrt(math.factorial(3)) / (4 * math.factorial(4))

    print(f"lambda={lambda_} tau={lambda_ * time}")

    # Call pyqsp to get the angles for QSP
    # phases, reduced_phases, parity = get_qsp_phases(lambda_, time, epsilon)

    # phases = (cos angles, sin angles)
    phases = get_qsp_phases(lambda_, time, epsilon)

    # Start QSP process
    qsp = QSP()
    qsp.compute(
        phases=phases,
        branch=branch,
        index=index,
        system=system,
        mode=mode,
        selection=selection,
        random_seed=random_seed,
    )


def get_qsp_phases(lambda_: float, t: float, epsilon: float):
    """Get the QSP phases using pyQSP for given parameters.

    Args:
        lambda_ (float): The normalization factor of the Hamiltonian.
        t (float): The time for simulation.
        epsilon (float): The desired precision."""

    # pg = PolyCosineTX()
    # pcoefs, scale_c = pg.generate(tau=tau, epsilon=epsilon, ensure_bounded=True, return_scale=True)
    # pcoefs = pg.generate(tau=tau, epsilon=epsilon, ensure_bounded=False)
    # print(f"cos-pcoefs:{pcoefs}")
    # cos_angles = QuantumSignalProcessingPhases(pcoefs)

    # pg = PolySineTX()
    # pcoefs, scale_s = pg.generate(tau=tau, epsilon=epsilon, ensure_bounded=True, return_scale=True)
    # pcoefs = pg.generate(tau=tau, epsilon=epsilon, ensure_bounded=False)
    # print(f"sin-pcoefs:{pcoefs}")
    # sin_angles = QuantumSignalProcessingPhases(pcoefs)

    tau = lambda_ * t
    k_paper = 2 * (tau + (3 ** (2 / 3) / 2) * (tau ** (1 / 3)) * (np.log(1 / epsilon) ** (2 / 3)))
    print(f"poly_degree={k_paper}")
    max_scale = 0.95

    sin_poly, scale_s = PolyTaylorSeries().taylor_series(
        func=lambda x: -np.sin(tau * x),
        degree=int(k_paper),
        return_scale=True,
        max_scale=max_scale,
        chebyshev_basis=True,
        cheb_samples=2 * int(k_paper),
    )
    sin_angles, _, _ = QuantumSignalProcessingPhases(sin_poly, method="sym_qsp", chebyshev_basis=True)

    cos_poly, scale_c = PolyTaylorSeries().taylor_series(
        func=lambda x: np.cos(tau * x),
        degree=int(k_paper),
        return_scale=True,
        max_scale=max_scale,
        chebyshev_basis=True,
        cheb_samples=2 * int(k_paper),
    )
    cos_angles, _, _ = QuantumSignalProcessingPhases(cos_poly, method="sym_qsp", chebyshev_basis=True)

    return (cos_angles, sin_angles)


class QSP(Qubrick):
    def _compute(
        self,
        phases,
        branch: Qubits,
        index: Qubits,
        system: Qubits,
        mode: Qubits,
        selection: Qubits,
        random_seed: int | None = None,
    ):
        """Apply the QSP sequence with given phases on the walk qubits.

        Args:
            phases (list[float]): The list of phases for the QSP sequence.
            walk (Qubits): The walk qubits to apply the QSP sequence on.
        """
        aqubitization = AsymmetricQubitization(random_seed=random_seed)

        cos_phases, sin_phases = phases

        selection.had()
        selection.s()

        # perform loop of rotation(s) then Walk
        min_angles = min(len(sin_phases), len(cos_phases)) - 1
        for idx in range(min_angles):
            mode.rz(cos_phases[idx] * Units.rad, cond=~selection)
            mode.rz(sin_phases[idx] * Units.rad, cond=selection)
            dagger_flag = idx % 2 == 1
            aqubitization.compute(
                branch=branch,
                index=index,
                system=system,
                dagger=dagger_flag,
                ctrl=mode,
            )

        mode.rz(cos_phases[min_angles] * Units.rad, cond=~selection)
        mode.rz(sin_phases[min_angles] * Units.rad, cond=selection)

        if len(cos_phases) > len(sin_phases):
            dagger_flag = min_angles % 2 == 1
            aqubitization.compute(
                branch=branch,
                index=index,
                system=system,
                dagger=dagger_flag,
                ctrl=(mode | ~selection),
            )
            mode.rz(cos_phases[-1] * Units.rad, cond=~selection)
        else:
            dagger_flag = min_angles % 2 == 1
            aqubitization.compute(
                branch=branch,
                index=index,
                system=system,
                dagger=dagger_flag,
                ctrl=(mode | selection),
            )
            mode.rz(sin_phases[-1] * Units.rad, cond=selection)

        selection.had()
        # selection.ry(-theta * Units.rad)
