/*
 Apsp.c 

* This is a serial implementation of the Floyd-Warshall algorithm, for 
* finding shortest paths in a weighted graph with positive or negative 
* edge weights (without negative cycles). A single execution of the 
* algorithm can find the lengths (summed weights) of shortest paths 
* between all pairs of vertices.
* 
*  A random graph is generated, modeled by a NxN array of weights 
* between the graph vertices. The missing edges between vertices are 
* implemented as weights above the weight limit w.
 
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
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

//global variables
struct timeval startwtime, endwtime;
double seq_time;
int n,w;
float **a,**dist,p;


//declare functions
void makeAdjacency();
void floydWarshall();
void init();

int main(int argc, char **argv){
  int i,j;		//indices
  
  //check arguments  
  if (argc != 4) {
    printf("non acceptable input error\n");
    exit(3);	//error code 3 if arqs less or more
  }

  n = 1<<atoi(argv[1]);
  w = atoi(argv[2]);
  p = atof(argv[3]);
  
  init();
  
  makeAdjacency();
  
  floydWarshall();
  
  
//deeeeebug
  //~ printf("Start\n");
  //~ for(i=0;i<n;i++){
    //~ for(j=0;j<n;j++){
      //~ printf("%d--->%d  %f\n",i,j,a[i][j]);
	//~ }
  //~ }
  
  //~ for(i=0;i<n;i++){
    //~ for(j=0;j<n;j++){
      //~ printf("%d--->%d  %f\n",i,j,dist[i][j]);
	//~ }
  //~ }
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
  	  //check if i--->j vertices conected
  	  if(ran>p){
  	  	//if not connected weight is out of the limit
  	  	a[i][j]=w+100;
	  }
	  else{
	  	ran=((float)rand())/(float)(RAND_MAX);	//random float [0,1]
	    a[i][j]=ran*w;							//random float [0,w]
	  }
	}
	//i-->i weight 0
	a[i][i]=0;
  }
}

/**
Applies the Floy-Warshall algorithm into the graph
**/
void floydWarshall(){
  int i, j, k;
  float temp;
  //init dist
  for(i=0;i<n;i++){
  	for(j=0;j<n;j++){
      //simple weight initialization of dist array (over the w limit is infinity)
  	  dist[i][j]=a[i][j];
	}
	dist[i][i]=0;	//vertex from itself distance(weight) is 0
  }
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

/**
Creates the graph array and the distance array
**/
void init(){
  int i, j;
  a=(float **) malloc(n*sizeof(float*));
  dist=(float **) malloc(n*sizeof(float*));
  
  for(i=0;i<n;i++){
  	a[i]=(float *) malloc(n*sizeof(float));
	dist[i]=(float *) malloc(n*sizeof(float));
  }
}

//TODO dist, matlab D, init and serial floyd
