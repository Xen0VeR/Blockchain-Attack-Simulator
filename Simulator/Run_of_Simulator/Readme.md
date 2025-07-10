This file contains all the outputs of a single run of simulation.

#### The Run is as follows:
```bash
python3 Simulator.py 30 40 --n 50 --Ttx 10 --I 30 --end 300 > output.txt
```
The directory Block_Tree_Files contains the files containing block trees of each peer (one file per peer).
## Plots:
The directory Plot_Branches_analysis contains the plots of branch analysis corresponding to the above run which includes three plots:

1. Bar graph indicating the number of branches per peer in the run . Colors are given on the basis of peer classification : (high CPU, fast),(high CPU, slow),(low CPU, fast),(low CPU, slow).
  
2. Box plot of Branch length (in terms of blocks) v/s type of peer based on CPU power and speed.
 
3. A line plot of the longest chain contribution ratio v/s peer ID for all the peers in the run. 
