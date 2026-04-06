import numpy as np
import matplotlib.pyplot as plt
from psiqworkbench import QPU, Qubits, resource_estimator
from ..ppr.ppr import PPR
from ..jw_transform.hamiltonian import SYK_hamil
from .trotter import second_order_trotter

def run_resource_comparison():
    
    # Parameters
    n_values = [2, 4, 8, 12, 16]
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

   # Plotting 
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for n in n_values:
        rot_key = "2_Filters (Rotations)"
        tg_key = "3_Filters (T-Gates)"

        # Linear Plots 
        axes[0, 0].plot(step_values, results[n][rot_key], marker='o', label=f'n={n}')
        axes[0, 1].plot(step_values, results[n][tg_key], marker='s', label=f'n={n}')
        
        # Log Plots
        axes[1, 0].plot(step_values, results[n][rot_key], marker='o', label=f'n={n}')
        axes[1, 1].plot(step_values, results[n][tg_key], marker='s', label=f'n={n}')

    axes[0, 0].set_title("Logical Rotations (Linear Scale)")
    axes[0, 1].set_title("Synthesized T-Gates (Linear Scale)")
    
    axes[1, 0].set_title("Logical Rotations (Log Scale)")
    axes[1, 0].set_yscale('log')
    axes[1, 1].set_title("Synthesized T-Gates (Log Scale)")
    axes[1, 1].set_yscale('log')

    # General Labels
    for ax in axes.flat:
        ax.set_xlabel("Trotter Steps")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("syk_resource_scaling_comparison.png")
    print("\nSUCCESS: Multi-scale graphs saved to 'syk_resource_scaling_comparison.png'")


def get_resource_estimate():
    n_qubits = 16
    time = 0.5
    trotter_steps = 1
    
    ham = SYK_hamil(n=n_qubits, J=1.0, random_seed=42)
    ppr = PPR()

    #qpu_t = QPU(num_qubits=n_qubits, filters=[">>clean-ladder-filter>>", ">>single-control-filter>>"])
    #qpu_t = QPU(num_qubits=n_qubits, filters=[">>clean-ladder-filter>>", ">>single-control-filter>>", ">>rs-synth-filter>>"])
    qpu_t = QPU(num_qubits=n_qubits, filters=[">>witness>>"])
    qubits_t = Qubits(qpu=qpu_t, num_qubits=n_qubits)
    second_order_trotter(ham, qubits_t, ppr, time, trotter_steps)
    
    print("\n[TROTTER RESOURCES]")
    res_est_t = resource_estimator(qpu_t)
    t_resources = res_est_t.resources()
    print(t_resources)
    
    rotations = t_resources.get('rotations', 0)
    
    print(f"\n--- Rotations for 1 trotter step ---")
    print(f"Trotter rotation gates: {rotations} ")





if __name__ == "__main__":
    run_resource_comparison()
    #get_resource_estimate()