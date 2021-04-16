Short trajectory in state A and B for alanine dipeptide

Descriptors (input features):
* 45 atomic distances
* 10 angles
* 6 diehdral angles 

To enforce periodicity, both angles and diedhrals have been converted into pairs of sin, cos. The total number of features is then 45 + 2x10 + 2x6 = 77

The files ala2_stateA.dat and ala2_stateB.dat contains the input data in the following format: (N_data, N_features + 1). Since they are trajectories, the first colum represents time, and can be discarded. 
