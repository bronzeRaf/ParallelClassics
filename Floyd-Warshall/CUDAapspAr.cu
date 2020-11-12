/*
 CUDAapsp.cu 

* This is a parallel implementation of the Floyd-Warshall algorithm, for 
* finding shortest paths in a weighted graph with positive or negative 
* edge weights (without negative cycles). A single execution of the 
* algorithm can find the lengths (summed weights) of shortest paths 
* between all pairs of vertices.
* 
*  A random graph is generated, modeled by a NxN array of weights 
* between the graph vertices. The missing edges between vertices are 
* implemented as weights above the weight limit w.
* 
* The implementation uses CUDA.
* For every block, BLOCK_SIZExBLOCK_SIZE CUDA threads are used for a 
* total of NxN CUDA threds. Every thread is responsible for a single 
* cell of the Floyd-Warshall distance array.
* The number of blocks is variable but the number of threads is constant.
* 
* The stable version includes only implementation without any shared 
* memory. oneOneNo() and floydWarshall_p1(float *dev_dist,size_t pitch,int n)
* are ready to use. 
* The shared memory extension will be attached on a later commit.
 
 Command line arguments:
 n = the number of vertices into the graph
 w = the max weight between vertices
 p = the probability of generating edge
        
 */

/* 
------- ---------------------- 
   Brouzos Rafael 	rnm1816@gmail.com	www.github.com/bronzeRaf
-----------------------------
*/#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define BLOCK_SIZE 16

//global variables
struct timeval startwtime, endwtime;
double seq_time;
int n,w;
float **a,**dist,**tesd,p;

//declare functions
void makeAdjacency();
void hostAlloc();
void init();
void floydWarshall_serial();
void oneOneNo();
void oneOneYes();
void tester();
void initTest();
__global__ void floydWarshall_p1(float *dev_dist,size_t pitch,int en);
__global__ void floydWarshall_p2(float *dev_dist,size_t pitch,int en);

int main(int argc, char **argv){
  //check arguments  
  if (argc != 4) {
    printf("non acceptable input error\n");
    exit(3);		//error code 3 if arqs less or more
  }

  n= 1<<atoi(argv[1]);
  w= atoi(argv[2]);
  p= atof(argv[3]);
 
  printf("n       w       p       serial       OneOneNo\n");
  printf("%d  %d  %f  ",n,w,p);
  
  hostAlloc();
  
  makeAdjacency();
  
  gettimeofday (&startwtime, NULL);
  
  floydWarshall_serial();
  
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
  printf("%f  ", seq_time);
  
  initTest();

  gettimeofday (&startwtime, NULL);
  
  oneOneNo();
  
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
  printf("%f\n", seq_time);  
  
//  oneOneYes(); //not ready yet

    
  free(tesd);
  free(dist);
  free(a);
}

//make adjacency matrix a(1:n,1:n) where a edge is generated with probability p
//and random edge weights (0:w). Instead of infity (if vertexes unconnected) we put a value over w
void makeAdjacency(){
  int i,j;
  float ran;
  srand(time(NULL));//initializing rand()
  
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
  	  ran=((float)rand())/(float)(RAND_MAX);//random float [0,1]
  	  //check if i--->j vertexes conected
  	  if(ran>p){
  	  	//if not connected weight is out of the limit
  	  	a[i][j]=w+100;
	  }
	  else{
	  	ran=((float)rand())/(float)(RAND_MAX);//random float [0,1]
	    a[i][j]=ran*w;//random float [0,w]
	  }
	}
	//i-->i weight 0
	a[i][i]=0;
  }
}

//simple Floy-Warshall algorith based on wikipedia
void floydWarshall_serial(){
  int i, j, k;
  float temp;
  //init dist
  init();
  //main algorithm
  for(k=0;k<n;k++){
  	for(i=0;i<n;i++){
  	  for(j=0;j<n;j++){
  	  	temp=dist[i][k]+dist[k][j];
  	  	if(dist[i][j]>temp){
  	  	  dist[i][j]=temp;
		}
	  }
	}
  }
  
}

//creating a,dist
void hostAlloc(){
  int i;
  a=(float **) malloc(n*sizeof(float*));
  dist=(float **) malloc(n*sizeof(float*));
  tesd=(float **) malloc(n*sizeof(float*));
  
  for(i=0;i<n;i++){
  	a[i]=(float *) malloc(n*sizeof(float));
	dist[i]=(float *) malloc(n*sizeof(float));
	tesd[i]=(float *) malloc(n*sizeof(float));
  }
}

//initializing dist matrix with a values
void init(){
  int i,j;
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
  	  dist[i][j]=a[i][j];//simple weight initialization of dist array (over the w limit is infinity)
	}
	dist[i][i]=0;//vertex from itself distance(weight) is 0
  }
}

//host code for executing 1 cell 1 thread without shared memory (host function)
void oneOneNo(){
  //init dist
  init();
    
  float *dev_dist;//device memory dist
  size_t pitch;
  cudaMallocPitch(&dev_dist, &pitch, n * sizeof(float), n);//malloc in device memory
  //copy dist array to global memory at device
  cudaMemcpy2D(dev_dist,pitch,dist,n*sizeof(float),n*sizeof(float),n,cudaMemcpyHostToDevice);
  
  //call kernel
  dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);//threads per block = BLOCK_SIZE^2
  dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y);//blocks per grid
  floydWarshall_p1<<<numBlocks, threadsPerBlock>>>(dev_dist, pitch, n);//call kernel
  
  cudaDeviceSynchronize();
  
  //get results from device to host memory
  cudaMemcpy2D(dist,n*sizeof(float),dev_dist,pitch,n*sizeof(float),n,cudaMemcpyDeviceToHost);
  //we have results (minimun weight path) in dist array 

  cudaFree(dev_dist);
}

//kernel function for 1thread per cell without shared memory
__global__ void floydWarshall_p1(float *dev_dist,size_t pitch,int en){
  float temp, d1, d2, *row;
  int k;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i<en && j<en){
    for(k=0;k<en;k++){
	  row = (float*)((char*)dev_dist + i*pitch);
      d1=row[k];//=dist[i][k]
      row = (float*)((char*)dev_dist + k*pitch);
	  d2=row[j];//=dist[k][j]
	  row = (float*)((char*)dev_dist + i*pitch);
      temp=d1+d2;
  	  if(row[j]>temp){
  	  	row[j]=temp;//=dist[i][j]
  	  }
	}
  }
  __syncthreads();
}

//host code for executing 1 cell 1 thread with shared memory (host function)
void oneOneYes(){
  init();
  
  float *dev_dist;//device memory dist
  size_t pitch;
  cudaMallocPitch(&dev_dist, &pitch, n * sizeof(float), n);//malloc in device memory
  //copy dist array to global memory at device
  cudaMemcpy2D(dev_dist,pitch,dist,n*sizeof(float),n*sizeof(float),n,cudaMemcpyHostToDevice);
  
  //call kernel
  dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);//threads per block = BLOCK_SIZE^2
  dim3 numBlocks(n / threadsPerBlock.x, n / threadsPerBlock.y);
  floydWarshall_p2<<<numBlocks, threadsPerBlock>>>(dev_dist, pitch, n);
  
  cudaDeviceSynchronize();
  
  //get results from device to host memory
  cudaMemcpy2D(dist,n*sizeof(float),dev_dist,pitch,n*sizeof(float),n,cudaMemcpyDeviceToHost);
  //we have results (minimun weight path) in dist array
  
  cudaFree(dev_dist);
}

//kernel function for 1thread per cell with shared memory
//first BLOCK_SIZE "checks" will be done inside shared memory, the other inside global memory
__global__ void floydWarshall_p2(float *dev_dist,size_t pitch,int en){
  float temp, d1=0, d2=0,*row;
  __shared__ float sh[BLOCK_SIZE][BLOCK_SIZE];//den katafera na sxediasw dynamic pinaka me extern edw opws i8ela logo xronou. opote kanw kai xrisi global apo ena simeia
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int ti = threadIdx.x;
  int tj = threadIdx.y;
  int k;

  if (i<en && j<en){
  	//copy in shared memory
    row = (float*)((char*)dev_dist + i*pitch);
    sh[ti][tj]=row[j];//=dist[i][j]
    for(k=0;k<en;k++){
      if(k<BLOCK_SIZE){//if in shared check
      	d1=sh[ti][k];
      	d2=sh[k][tj];
      	temp=d1+d2;
      	if(sh[ti][tj]>temp){
      	  sh[ti][tj]=temp;//=dist[i][j]
		}
	  }
	  else{//else if in global check
	  	row = (float*)((char*)dev_dist + i*pitch);
        d1=row[k];//=dist[i][k]
        row = (float*)((char*)dev_dist + k*pitch);
	    d2=row[j];//=dist[k][j]
        temp=d1+d2;
  	    if(sh[ti][tj]>temp){
  	      sh[ti][tj]=temp;//=dist[i][j]
  	    }
	  }
    }
    __syncthreads();
    //copy from shared memory
    row = (float*)((char*)dev_dist + i*pitch);
    row[j]=sh[ti][tj];//=dist[i][j]
  }
  
}

//initializing tesd array with dist values. Its called after serial algorithm to make a clone of dist
void initTest(){
  int i,j;
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      tesd[i][j]=dist[i][j];
	}
  }
}

//test every cell of tesd (serial dist) with dist if its different (must be serial=parallel)
void tester(){
  int i,j,flag=0;
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
  	  if(dist[i][j] != tesd[i][j]){
  	  	flag=1;
  	  	break;
	  }
	}
	if(flag==1){
	  printf("ALERT'''''''''''''different than serial'''''''''''''\n");
	  break;
	}
  }
  if(flag==0)
    printf("everything ok in test\n");
}
