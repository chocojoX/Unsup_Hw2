# Unsup_Hw2
Homework 2 for Unsupervised learning

## Prerequisites

The code was written in *Python*

To be able to run the code, please download the following packages :
- munkres
- sklearn
- scipy
- numpy

When these packages are available, you should create a folder called _data_ in the project root. Files _ExtendedYaleB.mat_ and the folder _Hopkins155_ should be added to this directory.
Note that the _data_ folder will also contain the so called _distance matrix_ containing euclidean distance between all images. This is done to ensure that this distance is computed once and for all, which can lead to a great gain of time.


## Run the code

Running the command _python main.py_, tests should be starting tu run. First, the faces challenge will run.

For respectively _Spectral Clustering_, _K-subspaces_ and _SSC_, the best parameters will be computed on 2 individuals, then, with these parameters, the performance for 2, 10, 20, 30 and 38 individuals clustering will be computed.

Some plots will appear to provide insights on the parameters. Ths code will be paused while these plots are active. As soon as you'll escape them, the code will start to run again.
