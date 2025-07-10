## Contents of the directory:
1. Simulator.py : This file contains the code of Simulator class and all other classes required to run a simulation.
#### Instructions to run Simulator.py : 
A sample run : 
```bash
python3 Simulator.py 30 40 --n 50 --Ttx 10 --I 30 --end 300 > output.txt 
```
Here, the parameters such as z0(first command line argument), z1(second command argument), n(written after --n), Ttx, I, end can be changed as required.

2. Analysis.py : This file plots the longest chain contribution ratio versus varying parameters. It is for generating plots to analyse the effect of various parameters on the ratio.
#### Instructions to run Analysis.py :
A sample run : 
```bash
python3 Analysis.py
```
Warning: This file takes a lot of time to run as it has to run a lot of simulations with different parameters (to be specific it runs 46 simulations). For my device(10 cores) it took almost 3 hours to run this and generate the plots.
