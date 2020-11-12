/*
 tooCUDA.cu 

* This is the CUDA part of a parallel implementation of the 
* Floyd-Warshall algorithm, for finding shortest paths in a 
* weighted graph. 
* 
* The implementation is used together with a C programm.
* This is only the CUDA part and it should be compiled 
* separately and linked afterwards.
* 
* The CUDA contains two implementations. One using shared memory and one using
* only the global memory of the GPU.
*
* Without Shared Memory
* host  : 	oneOneNo()
* kernel:	floydWarshall_p1(float *dev_dist, int n, int nt, int k, float *dev_step);
*
* With Shared Memory
* host  : 	oneOneYes()
* kernel:	floydWarshall_p2(float *dev_dist, int n, int nt, int k, float *dev_step);
* 
*/

/* 
------- ---------------------- 
   Brouzos Rafael 	rnm1816@gmail.com	www.github.com/bronzeRaf
-----------------------------
*/
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

//declare functions
__global__ void floydWarshall_p1(float *dev_dist, int n, int nt, int k, float *dev_step);
__global__ void floydWarshall_p2(float *dev_dist, int n, int nt, int k, float *dev_step);

/**
Host Function for executing 1 cell per thread without shared memory (host function)
**/
extern "C" void oneOneNo(float *dist, int n, int nt, int k, float *step){
  //variables
  float *dev_dist, *dev_step;	//device memory dist
  size_t size= nt*sizeof(float);
  size_t size2= n*sizeof(float);

  cudaMalloc(&dev_dist,size);	//memory allocate to device
  cudaMalloc(&dev_step,size2);	//memory allocate to device
  //copy array to device memory 
  cudaMemcpy(dev_dist, dist, size, cudaMemcpyHostToDevice); 
  //copy array to device memory
  cudaMemcpy(dev_step, step, size2, cudaMemcpyHostToDevice);  
  
  //make properties and call kernel
  int threadsPerBlock= 256;
  int blocksPerGrid= (nt + threadsPerBlock - 1) / threadsPerBlock;
  //call kernel
  floydWarshall_p1<<<blocksPerGrid, threadsPerBlock>>>(dev_dist, n, nt, k, dev_step);
  //wait all threds
  cudaDeviceSynchronize();
  //copy results to host memory
  cudaMemcpy(dist, dev_dist, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_dist);
  cudaFree(dev_step);
}

/**
Kernel Function for executing 1 cell per thread without shared memory (kernel function)
**/
__global__ void floydWarshall_p1(float *dev_dist, int n, int nt, int k, float *dev_step){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int ii,jj;
  float temp,std;
  if(i < nt){
  	ii= (int)i/n;			//row of current cell
  	jj=i-ii*n;				//column of current cell
  	std=dev_dist[ii*n+k];
  	temp=std+dev_step[jj];	//dist[ii][k]+dist[k][jj]
  	if(temp<dev_dist[i]){
  	  dev_dist[i]=temp;		//update value of cell
	}
  }
}

/**
Host Function for executing 1 cell per thread with shared memory (host function)
**/
extern "C" void oneOneYes(float *dist, int n, int nt, int k, float *step){
  //variables
  float *dev_dist, *dev_step;	//device memory dist
  size_t size= nt*sizeof(float);
  size_t size2= n*sizeof(float);

  cudaMalloc(&dev_dist,size);	//memory allocate to device
  cudaMalloc(&dev_step,size2);	//memory allocate to device
  //copy matrix to device memory 
  cudaMemcpy(dev_dist, dist, size, cudaMemcpyHostToDevice);	 
  //copy matrix to device memory 
  cudaMemcpy(dev_step, step, size2, cudaMemcpyHostToDevice); 
  
  //make properties and call kernel
  int threadsPerBlock= 256;
  int blocksPerGrid= (nt + threadsPerBlock - 1) / threadsPerBlock;
  //call kernel
  floydWarshall_p2<<<blocksPerGrid, threadsPerBlock>>>(dev_dist, n, nt, k, dev_step);
  //wait all threds
  cudaDeviceSynchronize();
  //copy results to host memory
  cudaMemcpy(dist, dev_dist, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_dist);
  cudaFree(dev_step);
}

/**
Kernel Function for executing 1 cell per thread without shared memory (kernel function)
There are no loops for big data (matrixes) so we dont need shared memory for variables that are called once.
**/
__global__ void floydWarshall_p2(float *dev_dist, int n, int nt, int k, float *dev_step){
  __shared__ int i;
  __shared__ int ii,jj,s_n;
  __shared__ float temp,std;
  
  i = blockDim.x * blockIdx.x + threadIdx.x;  
  //change n parameter to shared memory (used more than once)
  s_n=n;
  if(i < nt){
  	ii= (int)i/s_n;			//row of current cell
  	jj=i-ii*s_n;			//column of current cell
  	std=dev_dist[ii*s_n+k];
  	temp=std+dev_step[jj];	//dist[ii][k]+dist[k][jj]
  	if(temp<dev_dist[i]){
  	  dev_dist[i]=temp;		//update value of cell
	}
  }
}
