from psiqworkbench import Qubits, QPU
from psiqworkbench.ops.qpu_ops import convert_ops_to_cpp
from psiqworkbench.resource_estimation.qre import resource_estimator
from psiqworkbench.resource_estimation.witness_counter.witness_metrics_functions import metrics
from syk_simulation.qubitization.qsp import qsp_evolution, qsp_evolution_re
from syk_simulation.qubitization.asymmetric_qubitization import AsymmetricQubitization
import numpy as np


def gather_aq_estimates(
    N: int, J: int = 24, time: float = 1.0, epsilon: float = 1e-3, random_seed: int = 3, extra_qubits: int = 0
):
    # Qubit register sizes
    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1
    mode_size = 1
    selection_size = 1
    control_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size + mode_size + selection_size + control_size + extra_qubits

    print(f"num_qubits: {num_qubits} - without extra: {num_qubits - extra_qubits}")
    qpu = QPU(
        num_qubits=num_qubits,
        filters=[">>clean-ladder-filter>>", ">>toffoli-filter>>", ">>single-control-filter>>"],
        # num_qubits=num_qubits, filters=[">>clean-ladder-filter>>", ">>single-control-filter>>"],
    )
    qpu.reset(num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")
    mode = Qubits(mode_size, "mode", qpu=qpu)
    selection = Qubits(selection_size, "selection", qpu=qpu)

    system[2].x()

    branch.had()
    qsp_evolution(
        N=N,
        J=J,
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
    resources = resource_estimator(qpu).resources()
    return resources


def capture_walk_ops(N, random_seed, extra_qubits):
    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1
    mode_size = 1
    selection_size = 1
    control_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size + mode_size + selection_size + control_size + extra_qubits

    quick_qpu = QPU(
        num_qubits=num_qubits,
        filters=[">>clean-ladder-filter>>", ">>toffoli-filter>>", ">>single-control-filter>>", ">>witness>>"],
    )
    walk = Qubits(walk_size, "walk", qpu=quick_qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")
    mode = Qubits(mode_size, "mode", qpu=quick_qpu)
    selection = Qubits(selection_size, "selection", qpu=quick_qpu)

    cap_walk = quick_qpu.start_capture()

    aqubitization = AsymmetricQubitization(random_seed=random_seed)
    aqubitization.compute(
        branch=branch,
        index=index,
        system=system,
        dagger=False,
        ctrl=mode,
    )

    cap_walk.end_capture()
    walk_cpp = convert_ops_to_cpp(cap_walk.ops)

    quick_qpu.reset(num_qubits)
    walk = Qubits(walk_size, "walk", qpu=quick_qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")
    mode = Qubits(mode_size, "mode", qpu=quick_qpu)
    selection = Qubits(selection_size, "selection", qpu=quick_qpu)

    cap_walk_dagger = quick_qpu.start_capture()

    aqubitization.compute(
        branch=branch,
        index=index,
        system=system,
        dagger=True,
        ctrl=(mode | ~selection),
    )

    cap_walk_dagger.end_capture()
    walk_dagger_cpp = convert_ops_to_cpp(cap_walk_dagger.ops)
    return walk_cpp, walk_dagger_cpp


def gather_aq_re_cpp(
    N: int, J: int = 24, time: float = 1.0, epsilon: float = 1e-3, random_seed: int = 3, extra_qubits: int = 0
):
    # Qubit register sizes
    system_size = N
    index_chunk = int(np.ceil(np.log2(N)))
    index_size = 4 * index_chunk
    aux_unary_size = index_chunk
    range_flag_size = 1
    branch_size = 1
    mode_size = 1
    selection_size = 1
    control_size = 1

    walk_size = branch_size + index_size + system_size
    num_qubits = walk_size + aux_unary_size + range_flag_size + mode_size + selection_size + control_size + extra_qubits

    walk_ops = capture_walk_ops(N=N, random_seed=random_seed, extra_qubits=extra_qubits)

    print(f"num_qubits: {num_qubits} - without extra: {num_qubits - extra_qubits}")
    qpu = QPU(
        num_qubits=num_qubits,
        filters=[">>clean-ladder-filter>>", ">>toffoli-filter>>", ">>single-control-filter>>", ">>witness>>"],
        # num_qubits=num_qubits, filters=[">>clean-ladder-filter>>", ">>single-control-filter>>"],
    )
    qpu.reset(num_qubits)
    walk = Qubits(walk_size, "walk", qpu)
    branch = Qubits(walk[0:branch_size], "branch")
    index = Qubits(walk[branch_size : index_size + branch_size], "index")
    system = Qubits(walk[index_size + branch_size :], "system")
    mode = Qubits(mode_size, "mode", qpu=qpu)
    selection = Qubits(selection_size, "selection", qpu=qpu)

    branch.had()
    qsp_evolution_re(
        N=N,
        J=J,
        branch=branch,
        index=index,
        system=system,
        mode=mode,
        selection=selection,
        time=time,
        epsilon=epsilon,
        walk_ops=walk_ops,
        random_seed=random_seed,
    )
    branch.had()
    resources = resource_estimator(qpu).resources()
    return resources
