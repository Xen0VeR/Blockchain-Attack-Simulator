

# Blockchain Attack Simulator

This project simulates various mining and network attacks on blockchain systems. It provides a framework to analyze the impact of different parameters and attack strategies on blockchain consensus, network efficiency, and security.

## Project Structure

- **Attacks/**: Contains simulation scripts and plots related to different attack strategies.
- **P2P/**: Implements the Peer-to-Peer network used for simulation.
- **Simulator/**: Main simulation engine, analysis scripts, and output data.

## How to Run

### 1. Simulator

The main simulation is run using `Simulator/Simulator.py`. This script simulates the blockchain network with customizable parameters.

**Example usage:**
```bash
python Simulator/Simulator.py 30 40 --n 50 --Ttx 10 --I 30 --end 300 > Simulator/output.txt
```
- `30` and `40`: Example values for z0 and z1 (attack parameters)
- `--n 50`: Number of peers
- `--Ttx 10`: Mean transaction generation interval
- `--I 30`: Block interval
- `--end 300`: Simulation end time

You can adjust these parameters as needed.

### 2. Analysis

To generate plots analyzing the effect of various parameters, run:
```bash
python Simulator/Analysis.py
```
*Note: This script runs multiple simulations and may take several hours to complete.*

### 3. Peer-to-Peer Network

To run the P2P network simulation:
```bash
python P2P/P2P.py
```

## Output and Results

- Simulation outputs (block trees, logs, etc.) are saved in `Simulator/Run_of_Simulator/`.
- Plots analyzing branches and chain contributions are in `Simulator/Run_of_Simulator/Plots_branches_analysis/`.
- Plots for varying parameters are in `Simulator/Varying_parameters_plots/`.

## Visualizations

- Block tree images for each peer are in `Simulator/Block_Tree_Plots/`.
- Various plots (e.g., chain ratio vs. parameters) are in `Simulator/Varying_parameters_plots/`.

## More Information

For detailed instructions and explanations, see the `Readme.md` files in each subdirectory.
