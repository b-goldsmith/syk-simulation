import numpy as np
import matplotlib.pyplot as plt
from psiqworkbench import QPU, Qubits, resource_estimator
from ..ppr.ppr import PPR
from ..jw_transform.hamiltonian import SYK_hamil
from .trotter import second_order_trotter

def run_resource_estimates():

    """
        This function will graph the number of rotation gates needed for varying values of numbers of qubits (n)
        and varying trotter steps, using a set time. This will be graphed on a linear and log scale

        Note: For increasing n the number of rotation will scale quartically O(n^4)
    """
    
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

def run_epsilon_resource_estimates(): 
    """
        This function will graph the number of rotation gates needed for varying values of numbers of qubits (n)
        and varying epsilons, using a set time. This will be graphed on a linear and log scale
    """

    # ham = SYK_hamil(n=8, J=1.0)
    # steps = get_commutator_bound_steps(ham, t=0.5, epsilon=0.05)
    # print(f"Required Steps for Epsilon 0.05: {steps}")    
    
    # Parameters
    n_values = [4, 8, 12, 16, 20]
    eps_values = [0.1, 0.05, 0.01, 0.005]
    time = 0.5
    
    results = {eps: [] for eps in eps_values}

    print(f"{'n':<5} | {'eps':<10} | {'Steps (r)':<10} | {'Rotations':<15}")
    print("-" * 50)

    for eps in eps_values:
        for n in n_values:
            # Get Hamiltonian terms L
            L = comb(2*n, 4)
            
            # Get steps from Commutator Bound
            r = get_commutator_bound_steps(n, time, eps)
            
            # Calculate total rotations (2 passes per step)
            rotations = 2 * L * r
            
            results[eps].append(rotations)
            print(f"{n:<5} | {eps:<10} | {r:<10} | {rotations:<15,}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Rotations vs n (System Size Scaling)
    for eps in eps_values:
        axes[0].plot(n_values, results[eps], marker='o', label=f'eps={eps}')
    
    axes[0].set_title("Total Rotations vs. System Size ($n$)\n(Commutator Bound Scaling)")
    axes[0].set_xlabel("Number of Qubits (n)")
    axes[0].set_ylabel("Total Rotations")
    axes[0].set_yscale('log')
    axes[0].grid(True, which="both", ls="-", alpha=0.3)
    axes[0].legend()

    # Plot 2: Rotations vs epsilon (Precision Scaling)
    for n_idx, n in enumerate(n_values):
        y_vals = [results[eps][n_idx] for eps in eps_values]
        axes[1].plot(eps_values, y_vals, marker='s', label=f'n={n}')

    axes[1].set_title("Total Rotations vs. Precision ($\epsilon$)\n(Commutator Bound Scaling)")
    axes[1].set_xlabel("Target Epsilon (Error)")
    axes[1].set_ylabel("Total Rotations")
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].invert_xaxis() # Smaller epsilon (higher precision) to the right
    axes[1].grid(True, which="both", ls="-", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("syk_epsilon_scaling_bound.png")
    print("\nSUCCESS: Resource scaling plots saved to 'syk_epsilon_scaling_bound.png'")


def get_commutator_bound_steps(ham, t, epsilon, p=2):
   
    """
    Calculates the number of Trotter steps r using (Commutator Scaling).
    r = (alpha_comm^{1/p} * t^{1+1/p}) / epsilon^{1/p}
    for second order trotter, p = 2

    Args:
        ham: hamiltonian
        epsilon: an epsilon for the trotter-suzuki
        p: product formula order
    
    Returns:
        r: number of trotter steps
    """
    
    def calculate_alpha_comm(ham_dict):
        # alpha_comm = sum || [H_k, [H_j, H_i]] ||
        # For SYK, we can approximate this based on the number of 
        # anti-commuting Majorana pairs.
        terms = list(ham_dict.values())
        L = len(terms)
        
        # If the Hamiltonian is too large, use the analytical SYK bound
        # alpha_comm for SYK-4 scales as N^2 * lambda (Childs et al. 2021)
        if L > 500:
            n_qubits = int(np.round(np.power(L * 24 / 16, 1/4) / 2)) # Reverse L = (2n choose 4)
            N = 2 * n_qubits
            # Analytical alpha_comm for SYK
            return (N**2) * sum(abs(c) for c in terms) 
        
        # Manual calculation for small n
        alpha_comm = 0
        # Here we use the SYK 1-norm as a conservative proxy for the commutator
        lam = sum(abs(c) for c in terms)
        alpha_comm = lam**3 / (L**0.5) 
        return alpha_comm

    alpha_comm = calculate_alpha_comm(ham)
    # r = (alpha_comm * t^(p+1) / epsilon)^(1/p)
    r = np.power((alpha_comm * np.power(t, p+1)) / epsilon, 1/p)
    
    return int(np.ceil(r))

if __name__ == "__main__":
    #run_resource_comparison()
    #get_resource_estimate()
    run_epsilon_resource_estimates()