/*
 tooMPI.c 

* This is both a parallel and a serial implementation of the Floyd-Warshall 
* algorithm, for finding shortest paths in a weighted graph with positive 
* or negative edge weights (without negative cycles). A single execution of 
* the algorithm can find the lengths (summed weights) of shortest paths 
* between all pairs of vertices.
* 
* A random graph is generated, modeled by a NxN array of weights 
* between the graph vertices. The missing edges between vertices are 
* implemented as weights above the weight limit w.
*
* The serial version uses one process.
* The parallel version uses MPI to share the vertices of the graph to
* many tasks and it is also able to call 2 different CUDA versions
* to use the GPU for calculating the distances. The CUDA host and kernel
* functions are stored in the tooCuda.cu file.
 
 Command line arguments:
 n = the number of vertices into the graph
 w = the max weight between vertices
 p = the probability of generating edge
        
 */

/* 
------- ---------------------- 
   Brouzos Rafael 	rnm1816@gmail.com	www.github.com/bronzeRaf
-----------------------------
*/
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

//global variables
struct timeval startwtime, endwtime;
double seq_time;
int n, nt, w, rank, numtasks;
float *a, *dist, *step, *part_dist, *tesd, p;//pointers are 2D matrixes with [i*columns+j] instead of [i][j] (except of step that is normal 1D matrix)

//declare functions
void makeAdjacency();
void hostAlloc();
void init();
void floydWarshall_serial();
void initTest();
void tester();
void printMatrixes();
void updateSlaves();
void updateStep(int k);
void updateMaster();
void callParallel ();
void parallel(int flag);
//CUDA host functions
void oneOneNo(float *dist, int n, int nt, int k, float *step);
void oneOneYes(float *dist, int n, int nt, int k, float *step);

int main(int argc, char **argv){
  int i, j;		//indices
  
  //Mpi initialization
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  //check arguments
  if (argc != 4) {
  	if(rank==0)
      printf("non acceptable input error\n");
    exit(3);	//error code 3 if arqs less or more
  }  
  //obtain the arguments
  n= 1<<atoi(argv[1]);	//n=(int)2^arg1
  w= atoi(argv[2]);		//w=(int)arg2
  p= atof(argv[3]);		//p=(float)arg3
  nt= (n*n)/numtasks;	//rows from a to every task
  //check how many tasks made
  if(numtasks>n || n%numtasks!=0){
  	if(rank==0)
	  printf("number of tasks must be less than the number of vertexes and be able to get divided by n\n");
  	exit(4);	//error code 4 if number of tasks is more or is not able to get divided by n
  }
  //make full and partial matrixes
  hostAlloc();
  
  if(rank==0){
	//fill adjacency matrix
    makeAdjacency();
	//serial algorithm
	gettimeofday (&startwtime, NULL);	//take start time
    floydWarshall_serial();				//serial algorithm
    gettimeofday (&endwtime, NULL);		//take stop time
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("SERIAL ALGORITHM TIME:   %f\n\n", seq_time);

	//potential test initializing(tesd matrix)
	initTest();
	init();		//init distance array
  }
  //start of parallel solution (2 parallel algorithms)
  callParallel();
  MPI_Barrier(MPI_COMM_WORLD);

//printMatrixes();

  free(step);
  free(tesd);
  free(dist);
  free(part_dist);
  free(a);
  MPI_Finalize();
}

/** 
Makes adjacency matrix a(1:n,1:n) where a edge is generated with 
probability p and random edge weights (0:w).
Instead of infity (if vertexes unconnected) we put a value over w
**/
void makeAdjacency(){
  int i,j;
  float ran;
  srand(time(NULL));	//initializing rand()
  
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
  	  ran=((float)rand())/(float)(RAND_MAX);	//random float [0,1]
  	  //check if i--->j vertexes conected
  	  if(ran>p){
  	  	//if not connected weight is out of the limit (infinitive)
  	  	a[i*n+j]=w+100;
	  }
	  else{
	  	ran=((float)rand())/(float)(RAND_MAX);	//random float [0,1]
	    a[i*n+j]=ran*w;							//random float [0,w]
	  }
	}
	//i-->i weight 0
	a[i*n+i]=0;
  }
}

/**
Allocates memory for weight and distance arrays
**/
void hostAlloc(){
  int i;
  part_dist=(float *) malloc(nt*sizeof(float));	//2D matrix with only 1 pointer
  step=(float *) malloc(n*sizeof(float));		//normal 1D matrix
  //memory allocate for weights, test and distance array (only for rank=0 master)
  if(rank==0){
    a=(float *) malloc(n*n*sizeof(float));		//2d matrix with only 1 pointer
    tesd=(float *) malloc(n*n*sizeof(float));	//2d matrix with only 1 pointer
    dist=(float *) malloc(n*n*sizeof(float));	//2d matrix with only 1 pointer
  }
}

/**
initializing distance array with weight values
(called only from master rank=0)
**/
void init(){
  int i,j;
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
  	  dist[i*n+j]=a[i*n+j];//simple weight initialization of dist array (over the w limit is infinity)
	}
	dist[i*n+i]=0;//vertex from itself distance (weight) is 0
  }
}

/**
Applies the Floy-Warshall algorithm into the graph
**/
void floydWarshall_serial(){
  int i, j, k;
  float temp;
  //initialization of dist
  init();
  //main algorithm
  for(k=0;k<n;k++){
  	for(i=0;i<n;i++){
  	  for(j=0;j<n;j++){
  	  	temp=dist[i*n+k]+dist[k*n+j];
  	  	if(dist[i*n+j]>temp){
  	  	  dist[i*n+j]=temp;
		}
	  }
	}
  } 
}

/**
Initializes test array with distance values. It makes a clone of the
serial distance array for testing and validation
**/
void initTest(){
  int i,j;
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      tesd[i*n+j]=dist[i*n+j];
	}
  }
}

/**
It tests every cell of the parallel distance array 
with the serial one to test and validate results
**/
void tester(){
  int i,j,flag=0;
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
  	  if(dist[i*n+j] != tesd[i*n+j]){
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
    printf("TEST OK\n");
}

/**
Print the weight, the test and the distance arrays
Helpfull for debugging and checking the results
**/
void printMatrixes(){
  int i, j;
  if(rank==0){
    printf("________________n= %d  nt=%d  rank=%d  \n",n,nt,rank);

    printf("Here is A \n");
    for(i=0;i<n;i++){
      for(j=0;j<n;j++){
        printf("%f	",a[i*n+j]);
      }
      printf("\n");
    }

    printf("Here is TESD \n");
    for(i=0;i<n;i++){
      for(j=0;j<n;j++){
        printf("%f	",tesd[i*n+j]);
      }
      printf("\n");
    }

    printf("Here is DIST \n");
    for(i=0;i<n;i++){
      for(j=0;j<n;j++){
        printf("%f	",dist[i*n+j]);
      }
      printf("\n");
    }

  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
Gives mpi message from master task (rank=0) to all
The message contains the parts of the distance array
**/
void updateSlaves(){
  MPI_Scatter(dist,nt,MPI_FLOAT,part_dist,nt,MPI_FLOAT,0,MPI_COMM_WORLD);
}

/**
Updates the distance array in master rank=0 with all new information from every task.
**/
void updateMaster(){
    MPI_Gather (part_dist,nt,MPI_FLOAT,dist,nt,MPI_FLOAT,0,MPI_COMM_WORLD);
}

/**
Updates step array to every task according to last change.
Step array is the array that presents dist[k] row 
(k=0,1,...,n for n steps of execution of FLoyd-Warshall algorithm) from distance array
**/
void updateStep(int k){
  int root, i, j;
  root=(int)((k*numtasks)/n);	//root task that i row belongs

  if(rank==root){
  	i=k-(n*rank)/numtasks;		//row from part_dist we need to broadcast
  	for(j=0;j<n;j++){
  	  step[j]=part_dist[i*n+j];	//copy every column from i row
	}
  }
  //send step matrix to everyone for next step of algorithm
  MPI_Bcast (step,n,MPI_FLOAT,root,MPI_COMM_WORLD);
}

/**
Calls the two parallel implementations
First the one without the shared memory and then the one with
**/
void callParallel(){
  MPI_Barrier(MPI_COMM_WORLD);
  
  //Second parallel algorithm
  //only for master task
  if(rank==0)
    gettimeofday (&startwtime, NULL);	//take start time
  
  //1 thread 1 cell without shared memory CUDA-MPI
  parallel(1);
  
  MPI_Barrier(MPI_COMM_WORLD);
  //only for master task
  if(rank==0){
    gettimeofday (&endwtime, NULL);	//take stop time
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("1st PARALLEL ALGORITHM TIME:   %f	", seq_time);
  }
  //test and validate
  //only for master task
  if(rank==0){
    tester();
    init();	//init dist matrix
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  //Second parallel algorithm
  //only for master task
  if(rank==0)
    gettimeofday (&startwtime, NULL);	//take start time
  
  //1 thread 1 cell with shared memory CUDA-MPI
  parallel(2);
  
  MPI_Barrier(MPI_COMM_WORLD);
  //only for master task
  if(rank==0){
    gettimeofday (&endwtime, NULL);		//take stop time
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("2nd PARALLEL ALGORITHM TIME:   %f	", seq_time);
  }
  //only for master task
  if(rank==0){
    tester();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
Calls the Cuda part of the programm for each step of the algorithm
For flag=1 1 cell 1 thread without shared memory
For flag=2 1 cell 1 thread with shared memory
**/
void parallel(int flag){
  int k;
  updateSlaves();//send parts of distance array to every task

  if(flag==1){	//case 1
    for(k=0;k<n;k++){
	  //update step matrix for curent step of algorithm
	  updateStep(k);
	  //one cell one thread without shared memory
	  oneOneNo(part_dist,n,nt,k,step);
    }
	//update distance array (in rank=0 task)
	updateMaster();
  }
  else if(flag==2){	//case 2
    for(k=0;k<n;k++){
	  //update step matrix for curent step of algorithm
	  updateStep(k);
	  //one cell one thread with shared memory
	  oneOneYes(part_dist,n,nt,k,step);
    }
	//update distance array (in rank=0 task)
	updateMaster();
  }
}

