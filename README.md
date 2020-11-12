# Parallel Classics
A set of classic algorithms, implemented both through serial and parallel programming, comparing results and purposing solutions. This project uses random numbers for calculating problems, to proof and to present the capabilities of parallel programming. MPI and CUDA are used for the implementations.


## knn
A knn implementation. We assume that the space is divided into boxes (cubes) that model neighborhoods. In every box there are randomly generated q or c type of points. For every q point we need to find the nearest c neighbor. In the project, there is a serial and a parallel implementation using [MPI](https://www.open-mpi.org/).

### Serial Implementation
The serial implementation divides the space into v boxes and creates Numq random q points and Numc random c points. The points, based on their position belong to some boxes. To find the nearest c point of each q point, the algorithm searches in the neighbor boxes and select the nearest c of this neighborhood. The Numc and the Numq parameters must be powers of two.

##### Compile
- To compile the source code open a terminal in the same folder with "knn.c" and run:

```$ gcc knn.c -o executable-file-name```

	*executable-file-name = the name of the final executable

##### Run
- To run the knn algorithm for a random set of points of the desired amount open a terminal in the same folder with the executable and run:

```./executable-file-name Nq Nc v```
	
	* executable-file-name = the name of the final executable
	* Nq = the superscript of 2 for the number of q points as: Numq = 2^Nq
	* Nc = the superscript of 2 for the number of c points as: Numc = 2^Nc
	* v = the superscript of 2 for the number of boxes to divide the space as: boxes = 2^v


### Parallel Implementation
The parallel implementation of the knn algorithm is based on the serial one. The implementation is working on several processes. Each process is responsible for a set of consecutive boxes based on its rank. Separate processes that are responsible for neighbor boxes, exchange the required messages to check for q-c distances. Each process calculates the nearest c neighbor for any q that belongs to its boxes, no matter if the c belongs to another process.
 
##### Compile
- To compile the source code open a terminal in the same folder with "MPIknn.c" and run:

```$ mpicc MPIknn.c -o executable-file-name```

	*executable-file-name = the name of the final executable

##### Run
- To run the knn algorithm for a random set of points of the desired amount open a terminal in the same folder with the executable and run:

```./mpirun -n num_procs executable-file-name Nq Nc v```
	
	* executable-file-name = the name of the final executable
	* num_procs = number of MPI tasks to use
	* Nq = the superscript of 2 for the number of q points as: Numq = 2^Nq
	* Nc = the superscript of 2 for the number of c points as: Numc = 2^Nc
	* v = the superscript of 2 for the number of boxes to divide the space as: boxes = 2^v


### Results
In the "results" folder you can find multiple runtimes of the parallel implementation. For a complete comparison the results present the runtimes of 2, 4 and 32 MPI task, for several amount of points (Numq and Numc). The runtimes present the total time until finding all the nearest neighbors, as long as the time for saving the points before the calculations. All the timing presented in the text files is counted in seconds.
