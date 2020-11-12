/*
 knn.c 

 This file contains a knn implementation. 
 We assume that the space is divided into boxes (cubes) that model neighborhoods.
 In every box there are randomly generated q or c type of points. 
 For every q point we find the nearest c neighbor 
 
 Command line arguments:
 nq = the superscript of 2, showing the number of q points
 nc = the superscript of 2, showing the number of c points
 v = the superscript of 2, showing the number of boxes
        
 */

/* 
------- ---------------------- 
   Brouzos Rafael 	rnm1816@gmail.com	www.github.com/bronzeRaf
-----------------------------
*/
//#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

//global variables
struct timeval startwtime, endwtime;
double seq_time;

//struct for points q,c
typedef struct{
int idp,kox,koy,koz; // unique point ids
double x,y,z;
} point;
//neighbor id is given by their position as x=0,1...m y=0,1...k z=0,1...n f(x,y,z)=x+n*y+m*n*z

int fffq,jjjq,fffc,jjjc;
//nc=Nc nq=Nq v=n*m*k p=P
int nc,nq, v, vv, p;
int n,m,k,*countq,*countc;
point **aaq,**aac;

//declare functions
void plegma(void);
void qc(void);
double dest(int iq,int jq,int ic,int jc);
void keep(point *a, int job);

int main(int argc, char **argv){
  int i; //index
  //check arguments  
  if (argc != 4) {
    printf("non acceptable input error\n");
    exit(3);
  }
  
  //Nc=2^arg2 Nq=2^arg1
  nq= 1<<atoi(argv[1]);
  nc= 1<<atoi(argv[2]);
  //v=m*n*k=2^arg2
  vv= atoi(argv[3]);
  v= 1<<atoi(argv[3]);
  //memory allocation for counting q points of the box
  countq= (int *) malloc(v * sizeof(int));
  //memory allocation for counting c points of the box
  countc= (int *) malloc(v * sizeof(int));
  
  plegma();
  
  //debuging
  //printf("\nv=%d  n=%d  m=%d  k=%d\n\n",v,n,m,k);
  qc();
  
  //debuging
  //printf(" FINAL \nid=%d x=%f y=%f z=%f kox=%d koy=%d koz=%d f=%d\n--------____________---------\n",aaq[fffq][jjjq].idp,aaq[fffq][jjjq].x,aaq[fffq][jjjq].y,aaq[fffq][jjjq].z,aaq[fffq][jjjq].kox,aaq[fffq][jjjq].koy,aaq[fffq][jjjq].koz,fffq);
  //printf(" FINAL \nid=%d x=%f y=%f z=%f kox=%d koy=%d koz=%d f=%d\n--------____________---------\n",aac[fffc][jjjc].idp,aac[fffc][jjjc].x,aac[fffc][jjjc].y,aac[fffc][jjjc].z,aac[fffc][jjjc].kox,aac[fffc][jjjc].koy,aac[fffc][jjjc].koz,fffc);
}

/**
Initializes counters with zeros and calculates the m,n,k divisions of each dimension
**/
void plegma(){
  int i=0;
  n=(int)vv/3;
  m=(int)vv/3;
  k=(int)vv/3;
  if(vv%3==1){
  	k=k+1;
  }
  if(vv%3==2){
  	m=m+1;
  	k=k+1;
  }
  n=pow(2,n);
  m=pow(2,m);
  k=pow(2,k);
  //initialize counter
  for(i=0;i<v;i++){
  	countq[i]=0;
  	countc[i]=0;
  }
}

/**
This function fills aq and ac arrays with q and c random points. It also provides the counters of the q and c point for every box. 
It then produces the arrays aaq and aac, where rows give boxes (index i=0,...,v-1) and columns give the q and c points of the box (index j=0,...,count[i]-1)
**/
void qc(){
  int i=0,f,*c1,*c2,j;
  point *aq,*ac;
  //initializing rand
  srand(time(NULL));
  //allocate memory for q points
  aq= (point *) malloc(nq * sizeof(point));
  //allocate memory for c points
  ac= (point *) malloc(nc * sizeof(point));
  
  for(i=0;i<nq;i++){
    aq[i].idp=i;
    aq[i].x=((double)rand())/(double)(RAND_MAX);
    aq[i].y=((double)rand())/(double)(RAND_MAX);
    aq[i].z=((double)rand())/(double)(RAND_MAX);
    aq[i].kox=n*aq[i].x;
    aq[i].koy=m*aq[i].y;
    aq[i].koz=k*aq[i].z;
    
    f=aq[i].kox+n*aq[i].koy+m*n*aq[i].koz;
    
	//debuging
	//if(i==8){
      //printf("id=%d x=%lf y=%lf z=%lf kox=%d koy=%d koz=%d f=%d\n--------____________---------\n",aq[i].idp,aq[i].x,aq[i].y,aq[i].z,aq[i].kox,aq[i].koy,aq[i].koz,f);
	//}
  }
  for(i=0;i<nc;i++){
    //same for c
    ac[i].idp=i;
    ac[i].x=((double)rand())/(double)(RAND_MAX);
    ac[i].y=((double)rand())/(double)(RAND_MAX);
    ac[i].z=((double)rand())/(double)(RAND_MAX);
    ac[i].kox=n*ac[i].x;
    ac[i].koy=m*ac[i].y;
    ac[i].koz=k*ac[i].z;
    
    f=ac[i].kox+n*ac[i].koy+m*n*ac[i].koz;
    
    //debuging
    //if(i==12){
     // printf("id=%d x=%lf y=%lf z=%lf kox=%d koy=%d koz=%d f=%d\n--------____________---------\n",ac[i].idp,ac[i].x,ac[i].y,ac[i].z,ac[i].kox,ac[i].koy,ac[i].koz,f);
	//}
  }
  
  //allocate memory for the boxes arrays
  aaq=(point **) malloc(v*sizeof(point*));
  aac=(point **) malloc(v*sizeof(point*));
  for(i=0;i<v;i++){
    aaq[i]=NULL;
    aac[i]=NULL;
  }
  
  //insert q,c point to aaq and aac
  keep(aq,1);
  keep(ac,2);
  free(aq);
  free(ac);
}

/** 
Keeps in the aaq and in the aac the aq and the ac array data, that the current process gets involved
**/
void keep(point *a, int job){
  int i,f,j;
  if(job==1){
  	for(i=0;i<nq;i++){
      //sort q to their neighbors' arrays
      f=a[i].kox+n*a[i].koy+m*n*a[i].koz;
      //dynamically reallocate memory for new q
      j=countq[f];
      aaq[f] = (point *)realloc(aaq[f], (j+1) * sizeof(point));
   	  aaq[f][j].idp=a[i].idp;
      aaq[f][j].x=a[i].x;
      aaq[f][j].y=a[i].y;
      aaq[f][j].z=a[i].z;
      aaq[f][j].kox=a[i].kox;
      aaq[f][j].koy=a[i].koy;
      aaq[f][j].koz=a[i].koz;
      countq[f]=countq[f]+1;
    
   	  //debuging
	  //if(i==8){
        //fffq=f;
        //jjjq=j;
	  //}
    }
  }
  else{
    for(i=0;i<nc;i++){
      //same for c
      f=a[i].kox+n*a[i].koy+m*n*a[i].koz;
      //dynamically reallocate memory for new c
      j=countc[f];
      aac[f] = (point *)realloc(aac[f], (j+1) * sizeof(point));
	  aac[f][j].idp=a[i].idp;
      aac[f][j].x=a[i].x;
      aac[f][j].y=a[i].y;
      aac[f][j].z=a[i].z;
      aac[f][j].kox=a[i].kox;
      aac[f][j].koy=a[i].koy;
      aac[f][j].koz=a[i].koz;
      countc[f]=countc[f]+1;
    
	  //debuging
	  //if(i==12){
        //fffc=f;
        //jjjc=j;
	  //}
    }
  }
}


/**
Receives 4 indices qi, qj, ci and cj that point to a q and a c point and returns the float distance these two points
**/
double dest(int iq,int jq,int ic,int jc){
  double d,dx,dy,dz;
  dx=aaq[iq][jq].x-aac[ic][jc].x;
  dy=aaq[iq][jq].y-aac[ic][jc].y;
  dz=aaq[iq][jq].z-aac[ic][jc].z;
  d=sqrt(dx*dx+dy*dy+dz*dz);
  return d;
}

