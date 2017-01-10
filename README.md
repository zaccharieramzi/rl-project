# Multi-player Bandit
*This project is basically an implementation of different collision avoidance techniques for comparison.*

We implemented the following algorithms :
* TDFS
* rho rand
* MEGA
* Musical Chairs

## Requirements

This project use **Python3** and the requirements found in *requirements.txt*

## Organisation

### Algorithm Implementations
Each algorithm is implemented in its own folder with a *routines.py* and a *users.py* file.
*users.py* contains different types of bandit algorithms such as UCB or TS. And *routines.py* contains a routine with a simulation
and the collision avoidance mechanism.

Different types of Arms can be found in *arms.py*

### Notebooks

Experiments showcasing the algorithms are contained in three interactive Python notebooks.
They can be run by using `jupyter notebook`

They are :
* *benchmark.ipynb* : each routine is tested once, with regret plot.
* *Parameters Sensibility.ipynb* : Tests the effects of different parameters on Musical Chairs and MEGA
* *head2head.ipynb* : Comparison of all algorithm variants with four different scenarios
