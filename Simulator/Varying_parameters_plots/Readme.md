## Contents of the directory:
This directory contains the plots obtained by running the Analysis.py file which plots the longest chain contribution ratio versus varying parameters.

The fixed parameters were : **n=40**, **z0=30**, **z1=30**, **t_tx=30**, **block_interval=100** , **end_time = max(1000,10*block_interval)** 

The **end_time** is chosen in the above way so that a good amount of blocks are mined even if **block_interval** is as high as 800.

In every simulation one of the 5 parameters is varied as follows:

1. **z0** is varied from 0 to 100 at steps of 10 and a total of 11 simulations are run to plot the variation in the longest chain contribution ratio versus the varying **z0**.
 
2. **z1** is varied from 0 to 100 at steps of 10 and a total of 11 simulations are run to plot the variation in the longest chain contribution ratio versus the varying **z1**.
 
3. **t_tx(mean transaction generation intervals)** is varied from 10 to 60 at steps of 10 and a total of 6 simulations are run to plot the variation in the longest chain contribution ratio versus the varying **t_tx**.
 
4. **block_interval** is varied from 100 to 800 at steps of 100 and a total of 8 simulations are run to plot the variation in the longest chain contribution ratio versus the varying **block_interval**.
 
5. **n** is varied from 10 to 200 at steps of 20 and a total of 10 simulations are run to plot the variation in the longest chain contribution ratio versus the varying **n**.
