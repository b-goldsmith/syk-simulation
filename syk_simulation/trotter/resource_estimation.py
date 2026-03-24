import numpy as np
import matplotlib.pyplot as plt
from psiqworkbench import QPU, Qubits, resource_estimator
from ..ppr.ppr import PPR
from ..jw_transform.hamiltonian import SYK_hamil
from .trotter import second_order_trotter

def run_resource_comparison():
    
    # Parameters
    n_values = [2, 4, 8]
    step_values = [5, 10, 15, 20]
    time = 0.5
    ppr = PPR()


    filter_configs = {
        "2_Filters (Rotations)": [">>clean-ladder-filter>>", ">>single-control-filter>>"],
        "3_Filters (T-Gates)": [">>clean-ladder-filter>>", ">>single-control-filter>>", ">>rs-synth-filter>>"]
    }

    # Data storage: results[n][config][steps]
    results = {n: {config: [] for config in filter_configs} for n in n_values}

    for n in n_values:
        print(f"\n--- Analyzing n={n} Qubits ---")
        ham = SYK_hamil(n=n, J=1.0, random_seed=42)
        
        for config_name, filters in filter_configs.items():
            print(f"  Testing {config_name}...")
            for steps in step_values:
                # Initialize QPU with specific filter set
                qpu = QPU(num_qubits=n, filters=filters)
                qubits = Qubits(qpu=qpu, num_qubits=n)
                
                # Run Trotter
                second_order_trotter(ham, qubits, ppr, time, steps)
                
                # Estimate resources
                res_est = resource_estimator(qpu)
                data = res_est.resources()
                
                # Capture the relevant metric
                if "3_Filters" in config_name:
                    val = data.get('t_gates', 0)
                else:
                    val = data.get('rotations', 0)
                
                results[n][config_name].append(val)
                print(f"    Steps {steps}: {val}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for n in n_values:
        # Plot Rotations (2 Filters)
        axes[0].plot(step_values, results[n]["2_Filters (Rotations)"], marker='o', label=f'n={n}')
        # Plot T-Gates (3 Filters)
        axes[1].plot(step_values, results[n]["3_Filters (T-Gates)"], marker='s', label=f'n={n}')

    axes[0].set_title("Logical Rotations (2 Filters)")
    axes[0].set_xlabel("Trotter Steps")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Synthesized T-Gates (3 Filters)")
    axes[1].set_xlabel("Trotter Steps")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("syk_trotter_resource_scaling.png")
    print("\nSUCCESS: Graphs saved to 'syk_trotter_resource_scaling.png'")


def run_resource_comparison():
    n_qubits = 4
    time = 0.5
    trotter_steps = 15
    qdrift_samples = 3000
    
    ham = SYK_hamil(n=n_qubits, J=1.0, random_seed=42)
    ppr = PPR()

    #qpu_t = QPU(num_qubits=n_qubits, filters=[">>clean-ladder-filter>>", ">>single-control-filter>>"])
    qpu_t = QPU(num_qubits=n_qubits, filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>rs-synth-filter>>"])
    qubits_t = Qubits(qpu=qpu_t, num_qubits=n_qubits)
    second_order_trotter(ham, qubits_t, ppr, time, trotter_steps)
    
    print("\n[TROTTER RESOURCES]")
    res_est_t = resource_estimator(qpu_t)
    t_resources = res_est_t.resources()
    print(t_resources)

    t_cost_per_rot = 9
    
    t_rotations = t_resources.get('rotations', 0)
    
    print(f"\n--- Estimated T-Gate Cost (Post-Processed) ---")
    print(f"Trotter T-gates: {t_rotations * t_cost_per_rot}")


if __name__ == "__main__":
    run_resource_comparison()