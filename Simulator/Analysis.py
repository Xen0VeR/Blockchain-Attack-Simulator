import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from Simulator import Simulator, BlockchainAnalyzer

def compute_longest_chain_ratios(sim):
    analyzer = BlockchainAnalyzer(sim)
    analyzer.analyze_longest_chain_contribution()
    return analyzer.ratios

def plot_metric_vs_param(param_values, peer_type_metric_map, param_name, metric_name, filename):
    plt.figure(figsize=(10, 6))
    label_map = {
        'z0': 'Percentage of Slow Nodes',
        'z1': 'Percentage of Low CPU Nodes',
        'n': 'Number of Peers',
        't_tx': 'Transaction Generation Interval (sec)',
        'block_interval': 'Mean Block Interval (sec)'
    }
    xlabel = label_map.get(param_name, param_name)

    for peer_type, metrics in peer_type_metric_map.items():
        if len(param_values) != len(metrics):
            print(f"Skipping {peer_type}: Mismatched lengths: x={len(param_values)}, y={len(metrics)}")
            continue
        plt.plot(param_values, metrics, label=peer_type)

    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs {xlabel}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    os.makedirs("Varying_parameters_plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()

def run_simulation(param_name, val, fixed_params):
    params = fixed_params.copy()
    params[param_name] = val
    # Compute dynamic run_time
    block_interval = params.get('block_interval', 100)
    run_time = max(1000, 10 * block_interval)
    sim = Simulator(n=params['n'], z0=params['z0'], z1=params['z1'],
                    t_tx=params['t_tx'], block_interval=params['block_interval'])
    sim.initialize()
    sim.run(end_time=run_time)
    ratios = compute_longest_chain_ratios(sim)
    peer_types = ["Fast High CPU", "Fast Low CPU", "Slow High CPU", "Slow Low CPU"]
    ptype_counts = defaultdict(list)
    for pid, data in ratios.items():
        cpu = 'High CPU' if 'high' in data['cpu'].lower() else 'Low CPU'
        speed = data['speed'].capitalize()
        ptype = f'{speed} {cpu}'
        ptype_counts[ptype].append(data['ratio'])
    result = {}
    for ptype in peer_types:
        result[ptype] = np.mean(ptype_counts[ptype]) if ptype_counts[ptype] else 0
    return val, result
    

def sweep_parameter(param_name, values, fixed_params):
    peer_types = ["Fast High CPU", "Fast Low CPU", "Slow High CPU", "Slow Low CPU"]
    peer_type_to_metrics = {ptype: [] for ptype in peer_types}

    futures = []
    with ProcessPoolExecutor() as executor:
        for val in values:
            futures.append(executor.submit(run_simulation, param_name, val, fixed_params))

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    results.sort()  # sort by parameter value to align with x-axis

    for val, result in results:
        for ptype in peer_types:
            peer_type_to_metrics[ptype].append(result[ptype])

    plot_metric_vs_param(
        param_values=[val for val, _ in results],
        peer_type_metric_map=peer_type_to_metrics,
        param_name=param_name,
        metric_name="Avg Longest Chain Ratio",
        filename=f"{param_name}_vs_chain_ratio.png"
    )

def run_all_sweeps():
    fixed = dict(n=40, z0=30, z1=30, t_tx=30, block_interval=100)
    z_range = list(range(0, 101, 10))
    t_tx_range = list(range(10, 61, 10))
    tk_range = list(range(100, 801, 100))
    n_range = list(range(10, 201, 20))

    sweep_parameter("z0", z_range, fixed)
    sweep_parameter("z1", z_range, fixed)
    sweep_parameter("t_tx", t_tx_range, fixed)
    sweep_parameter("block_interval", tk_range, fixed)
    sweep_parameter("n", n_range, fixed)

if __name__ == "__main__":
    start = time()
    run_all_sweeps()
    end = time()
    print(f"Completed execution in : {((end - start)/60):.2f} minutes")

