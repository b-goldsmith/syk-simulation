"""This file contains the QSP algorithm for qubitization-based simulation of the SYK model."""

from psiqworkbench import Qubits, Qubrick, QPU
from workbench_algorithms.utils.paulimask import PauliSum
import numpy as np
import math

from pyqsp.poly import PolyTaylorSeries
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

from syk_simulation.qubitization import AsymmetricQubitization


def qsp_evolution(
    N: int,
    J: float,
    branch: Qubits,
    index: Qubits,
    system: Qubits,
    time: float,
    epsilon: float = 1e-3,
    random_depth: int = 2,
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

    # Call pyqsp to get the angles for QSP
    phases, reduced_phases, parity = get_qsp_phases(lambda_, time, epsilon)

    # Start QSP process
    qsp = QSP()
    qsp.compute(
        phases=phases,
        branch=branch,
        index=index,
        system=system,
        random_depth=random_depth,
        random_seed=random_seed,
    )


def get_qsp_phases(lambda_: float, t: float, epsilon: float):
    """Get the QSP phases using pyQSP for given parameters.

    Args:
        lambda_ (float): The normalization factor of the Hamiltonian.
        t (float): The time for simulation.
        epsilon (float): The desired precision."""

    def target_function(x):
        return np.cos(lambda_ * t * x)
        # return np.exp(-1j * lambda_ * t * x)

    # call QSP qubrick to run QSP
    degree_constant = 1
    degree = int(np.ceil(degree_constant * lambda_ * t + np.log(1 / epsilon)))

    poly = PolyTaylorSeries().taylor_series(
        func=target_function,
        degree=degree,
        max_scale=1.0,
        chebyshev_basis=True,
        cheb_samples=2 * degree,  # avoid aliasing
    )

    return QuantumSignalProcessingPhases(poly, method="sym_qsp", chebyshev_basis=True)


class QSP(Qubrick):
    def _compute(
        self,
        phases: list[float],
        branch: Qubits,
        index: Qubits,
        system: Qubits,
        random_depth: int,
        random_seed: int | None = None,
    ):
        """Apply the QSP sequence with given phases on the walk qubits.

        Args:
            phases (list[float]): The list of phases for the QSP sequence.
            walk (Qubits): The walk qubits to apply the QSP sequence on.
        """
        aqubitization = AsymmetricQubitization()

        branch.rz(phases[0])
        for phi in phases[1:]:
            aqubitization.compute(
                branch=branch,
                index=index,
                system=system,
                random_depth=random_depth,
                random_seed=random_seed,
            )
            branch.rz(phi)
