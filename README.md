


# Parallel Classics
A set of algorithms requiring strong computational power, implemented both through serial and parallel programming, comparing results, tools, techniques and purposing solutions. This project uses random numbers for calculating problems, to proof and to present the capabilities of parallel programming. [MPI](https://www.open-mpi.org/) and [CUDA](https://developer.nvidia.com/cuda-zone) are used for the implementations.


## knn
A serial and a parallel knn-like problem solving implementation. We assume that the space is divided into boxes (cubes) that model neighborhoods. In every box there are randomly generated q or c type of points. For every q point we need to find the nearest c neighbor. In the project, there is a serial and a parallel implementation using C and [MPI](https://www.open-mpi.org/). Both implementations contain validating functions, to proof that the results are correct.

### Serial Implementation
The serial implementation divides the space into v boxes and creates Numq random q points and Numc random c points. Every point, based on its position belong to a box. To find the nearest c point of each q point, the algorithm searches in the neighbor boxes and select the nearest c of this neighborhood. The Numc and the Numq and the number of boxes must be powers of two. The number of points and the number of boxes could be selected by the user.

##### Compile
- To compile the source code open a terminal in the same folder with "knn.c" and run:

```$ gcc -std=gnu89 knn.c -o executable-file-name -lm```

	*executable-file-name = the name of the final executable

##### Run
- To run the knn-like algorithm for a random set of points of the desired amount open a terminal in the same folder with the executable and run:

```./executable-file-name Nq Nc v```
	
	* executable-file-name = the name of the final executable
	* Nq = the superscript of 2 for the number of q points as: Numq = 2^Nq
	* Nc = the superscript of 2 for the number of c points as: Numc = 2^Nc
	* v = the superscript of 2 for the number of boxes to divide the space as: boxes = 2^v


### Parallel Implementation
The parallel implementation of the knn-like algorithm is based on the serial one. The implementation is working on several processes. Each process is responsible for a set of consecutive boxes based on its rank. Separate processes that are responsible for neighbor boxes, exchange the required messages to check for q-c distances. Each process calculates the nearest c neighbor for any q that belongs to its boxes, no matter if the c belongs to another process. The number of points and the number of boxes must be powers of two. The number of processes, the number of points and the number of boxes could be selected by the user.
 
##### Compile
- To compile the source code open a terminal in the same folder with "MPIknn.c" and run:

```$ mpicc -std=gnu89 MPIknn.c -o executable-file-name -lm```

	*executable-file-name = the name of the final executable

##### Run
- To run the parallel knn-like algorithm for a random set of points of the desired amount open a terminal in the same folder with the executable and run:

```mpirun -n num_procs executable-file-name Nq Nc v```
	
	* executable-file-name = the name of the final executable
	* num_procs = number of MPI tasks to use
	* Nq = the superscript of 2 for the number of q points as: Numq = 2^Nq
	* Nc = the superscript of 2 for the number of c points as: Numc = 2^Nc
	* v = the superscript of 2 for the number of boxes to divide the space as: boxes = 2^v


### Results
In the "results" folder you can find multiple runtimes of the parallel implementation. For a complete comparison the results present the runtimes of 2, 4 and 32 MPI task, for several amount of points (Numq and Numc). The runtimes present the total time until finding all the nearest neighbors, as long as the time for saving the points before the calculations. All the timing presented in the text files is counted in seconds.


## Floyd-Warshall
This is a set of a serial and a parallel implementation of the Floyd-Warshall algorithm. Floyd-Warshall is an algorithm for finding shortest paths in a weighted graph with positive or negative edge weights (without negative cycles). A single execution of the algorithm can find the lengths (summed weights) of shortest paths between all pairs of vertices. Although it does not return details of the paths themselves, it is possible to reconstruct the paths with simple modifications to the algorithm. Both implementations contain validating functions, to proof that the results are correct.

### Serial Implementation
The serial implementation is based on a NxN array that stores the weight of the edge from vertex i (row index) to vertex j (column index) and on a distance array that contains the shortest paths the same way. The random graph generation is based on the MATLAB function "makeAdjacency.m" that is given too. The number of vertices is decided by the user and then the graph is generated randomly. The missing edges between vertices are implemented as weights above the weight limit w. Therefore, the Floyd-Warshall algorithm is applied into the graph.


##### Compile
- To compile the source code open a terminal in the same folder with "Apsp.c" and run:

```$ gcc -std=gnu89 Apsp.c -o executable-file-name -lm```

	*executable-file-name = the name of the final executable

##### Run
- To run the Floyd-Warshall algorithm for a random graph of the desired size open a terminal in the same folder with the executable and run:

```./executable-file-name n w p```
	
	* executable-file-name = the name of the final executable
	* n = the number of vertices into the graph
	* w = the max weight between vertices
	* p = the probability of generating edge


### Parallel Implementation
The parallel implementation of the Floyd-Warshall algorithm is based on the serial one. The implementation is working with CUDA. The implementation starts NxN CUDA threds where every thread is responsible for a single cell of the described distance array. The CUDA threads are divided into blocks. The number of blocks is a variable but the total number of threads is always NxN. The CUDA version includes only implementation using the GPU global memory, without any shared memory usage.
 
##### Compile
Before compiling CUDA code, install CUDA following the [NVIDIA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
- To compile the source code open a terminal in the same folder with "CUDAapsp.cu" and run:

```$ nvcc -std=gnu89 CUDAapsp.cu -o executable-file-name -lm```

	* executable-file-name = the name of the final executable

##### Run
Before running a CUDA executable, get sure that your GPU supports CUDA runtime environment [here](https://developer.nvidia.com/cuda-gpus).
- To run the parallel Floyd-Warshall algorithm for a random graph of the desired size open a terminal in the same folder with the executable and run:

```./executable-file-name n w p```
	
	* executable-file-name = the name of the final executable
	* n = the number of vertices into the graph
	* w = the max weight between vertices
	* p = the probability of generating edge


### Results
In the "results" folder you can find runtimes of the serial and the parallel implementation. For a complete comparison, the results present the runtimes for several combinations of inputs. There are times for graphs from 128 to 4096 vertices with several w and p values. All the timing presented in the text files is counted in seconds. The parallel algorithm in the results is using block size of 16, one CUDA thread per vertex and no shared memory in the GPU.

## Too Parallel
This is a serial and a parallel implementation of the Floyd-Warshall algorithm using simultaneously CUDA and MPI to improve even more the performance. Floyd-Warshall is described above and this project is an extension of the previous one, introducing MPI in the implementation. All the implementations contain validating functions, to proof that the results are correct.

### Serial and Parallel Implementation
The serial implementation is similar with the Floyd-Warshall project serial implementation. The main difference is that this implementation is included into the parallel project, using only the master (rank = 0) MPI task to run the algorithm. The serial implementation is not a standalone one as in the previous project, but it is given only for comparison with the parallel implementations.

The parallel implementations of the Floyd-Warshall algorithm are using MPI to spawn tasks and divide the vertices of the graph to them to run calculations in parallel. Each task is also using CUDA threads to calculate distances in the GPU and this structure provides a nested parallelism. The number of the MPI task could be selected by the user, but it must be a power of 2, to allow the program divide the work fairly to the tasks. The number of CUDA threads is always NxN, one thread per distance between vertices. There are two parallel implementation, with difference only in the CUDA part. The first uses only the global memory of the GPU and the second uses also shared memory for critical data, to work faster. 

##### Compile
The MPI part and the CUDA part of the project must be compiled separately, form different tools. The linking process must be done afterwards, to export the final executable.

- First of all the CUDA part. Before compiling CUDA code, install CUDA following the [NVIDIA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). To compile the CUDA source code open a terminal in the same folder with "tooCUDA.cu" and run:

```$ nvcc -c -std=gnu89 tooCUDA.cu -o obj-cuda -lm```

	*obj-cuda = the name of the CUDA object file

- Now the MPI part. To compile the MPI source code open a terminal in the same folder with "tooMPI.c" and run:

```$ mpicc -c -std=gnu89 tooMPI.cu -o obj-mpi -lm```

	*obj-mpi = the name of the MPI object file

- Now the link. To link the two objects in the same executable open a terminal in the same folder with obj-cuda and obj-mpi and run:

```$ mpicc -std=gnu89 obj-mpi obj-cuda -o executable-file-name -lm -L/usr/local/cuda/lib64 -l cudart```

	* executable-file-name = the name of the final executable
	* obj-mpi = the name of the MPI object file
	* obj-cuda = the name of the CUDA object file

##### Run
Before running a CUDA executable, get sure that your GPU supports CUDA runtime environment [here](https://developer.nvidia.com/cuda-gpus).
- To run the serial and the parallel algorithms for a random graph of the desired size open a terminal in the same folder with the executable and run:

```mpirun -n num_procs executable-file-name n w p```
	
	* executable-file-name = the name of the final executable
	* num_procs = number of MPI tasks to use
	* n = the number of vertices into the graph
	* w = the max weight between vertices
	* p = the probability of generating edge


### Results
In the "results" folder you can find runtimes of the serial and the parallel implementations. For a complete comparison, the results present the runtimes for 2, 4, 8 and 16 MPI tasks. There are times for graphs from 128 to 4096 vertices with several w and p values. All the timing presented in the text files is counted in seconds. The parallel algorithm in the results is using 256 threads per block for a total of one CUDA thread per vertex. As "1st PARALLEL ALGORITHM" is mentioned the implementation using pure global memory in GPU and as "2nd PARALLEL ALGORITHM" is mentioned the implementation using also shared memory in GPU.
