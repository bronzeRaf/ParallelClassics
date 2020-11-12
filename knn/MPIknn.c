/*
 MPIknn.c 

 This file contains a parallel knn implementation using MPI. 
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
*/#include <mpi.h>
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

MPI_Datatype pointtype;

//nc=Nc nq=Nq v=n*m*k p=P
int nc,nq,v,vi,vv,p,pid,nci,nqi;
int n,m,k,*countq,*countc,***min;
point **aaq,**aac;

//declare functions
void plegma(void);
void qc(void);
double dest(int fq,int jq,int fc,int jc);
void keep(point *a, int job);
void equate(point *a,point *b,int size);
int getpid(int f);
void neib(int f,int *neibs);
int findmin(int fq,int jq,int fc);
void minq();
void minc(int fs,int irec,point ***rec,int ***reccou);

int main(int argc, char **argv){
  int i,j; //indices
  
  //check arguments
  if (argc != 4) {
    printf("non acceptable input error\n");
    exit(3);
  }
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  //Nc=2^arg2 Nq=2^arg1
  nq= 1<<atoi(argv[1]);
  nc= 1<<atoi(argv[2]);
  //v=m*n*k=2^arg2
  vv= atoi(argv[3]);
  v= 1<<atoi(argv[3]);
  vi=v/p;
  nci=nc/p;
  nqi=nq/p;
  
  //struct as message
  MPI_Datatype oldtypes[2]; 
  int blockcounts[2];
  MPI_Aint offsets[2], extent;
  //first 4 ints
  offsets[0] = 0;
  oldtypes[0] = MPI_INT;
  blockcounts[0] = 4;
  //make offset
  MPI_Type_extent(MPI_INT, &extent);
  offsets[1] = 4 * extent;
  //and 3 doubles
  oldtypes[1] = MPI_DOUBLE;
  blockcounts[1] = 2;
  //define and commit type
  MPI_Type_struct(2, blockcounts, offsets, oldtypes, &pointtype);
  MPI_Type_commit(&pointtype);
  
  //memory allocation for counting q points of the box
  countq= (int *) malloc(vi * sizeof(int));
  //memory allocation for counting c points of the box
  countc= (int *) malloc(vi * sizeof(int));
  //aaq and aac, where rows give boxes (index i=0,...,v-1) and columns give the q and c points of the box (index j=0,...,count[i]-1)  aaq=(point **) malloc(vi*sizeof(point*));
  aac=(point **) malloc(vi*sizeof(point*));
  for(i=0;i<vi;i++){
    aaq[i]=NULL;
    aac[i]=NULL;
  }
  
  
  plegma();
  
  if(pid==0){
    printf("Tasks Nq=Nc Boxes Until_saving_ponints Until_finding_min \n");
    printf("%d %d %d ",p,nq,v);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(pid==0){
    //check time
    gettimeofday (&startwtime, NULL);
  }
    
  qc();
  
  //memory allocate for counting for every q the index of c
  min= (int ***)malloc(vi*sizeof(int**));
  for(i=0;i<vi;i++){
    min[i]= (int **) malloc(countq[i]*sizeof(int*));
    for(j=0;j<countq[i];j++){
      min[i][j]= (int *) malloc(3*sizeof(int));
	}
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(pid==0){
    gettimeofday (&endwtime, NULL);
  
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

    printf("%f ", seq_time);
  }

  minq();
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(pid==0){
    gettimeofday (&endwtime, NULL);
  
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);

    printf("%f \n", seq_time);
  }


  free(aac);
  free(aaq);
  free(countc);
  free(countq);
  free(min);
  MPI_Type_free(&pointtype);
  MPI_Finalize();
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
  for(i=0;i<vi;i++){
  	countq[i]=0;
  	countc[i]=0;
  }
}

/**
This function fills aq and ac arrays with q and c random points. It also provides the counters of the q and c point for every box. 
It then produces the arrays aaq and aac, where rows give boxes (index i=0,...,v-1) and columns give the q and c points of the box (index j=0,...,count[i]-1)
**/
void qc(){
  int i=0,f,j,r,next,previous;
  point *aq,*ac,*sendq,*sendc,*recq,*recc;
  //initializing rand
  srand(time(NULL)*pid);
  //allocate memory for q points
  aq= (point *) malloc(nqi * sizeof(point));
  //allocate memory for c points
  ac= (point *) malloc(nci * sizeof(point));
  //allocate memory for send q buffer
  sendq= (point *) malloc(nqi * sizeof(point));
  //allocate memory for receive q buffer
  recq= (point *) malloc(nqi * sizeof(point));
  //allocate memory for send c buffer
  sendc= (point *) malloc(nci * sizeof(point));
  //allocate memory for receive c buffer
  recc= (point *) malloc(nci * sizeof(point));
  //fill with random q points
  for(i=0;i<nqi;i++){
    aq[i].idp=i+pid*nqi;
    aq[i].x=((double)rand())/(double)(RAND_MAX);
    aq[i].y=((double)rand())/(double)(RAND_MAX);
    aq[i].z=((double)rand())/(double)(RAND_MAX);
    aq[i].kox=n*aq[i].x;
    aq[i].koy=m*aq[i].y;
    aq[i].koz=k*aq[i].z;
  
  }
  
  //put points to the send buffer
  equate(&sendq[0],&aq[0],nqi);
  
  for(i=0;i<nci;i++){
    //same for c
    ac[i].idp=i+pid*nci;
    ac[i].x=((double)rand())/(double)(RAND_MAX);
    ac[i].y=((double)rand())/(double)(RAND_MAX);
    ac[i].z=((double)rand())/(double)(RAND_MAX);
    ac[i].kox=n*ac[i].x;
    ac[i].koy=m*ac[i].y;
    ac[i].koz=k*ac[i].z;	
  }  
  //put points to the send buffer
  equate(&sendc[0],&ac[0],nci);
  
  //MPI variables for comunication
  MPI_Request reqsq[8],reqsc[8],reqq[8],reqc[8];
  MPI_Status statsq[8],statsc[8],statq[8],statc[8];
  int tagq=1,tagc=2;
  
  previous=pid;
  next=pid;
  //everyone send and receive to all in p steps
  for(r=0;r<p;r++){
  	
  	//previous and next go further for next step
    previous=previous-1;
    next=next+1;
    if(previous<0){ previous=p-1; }
    if(next>p-1){ next=0; }
    //in final step we skip message (to self) and we keep the data that the process gets involved
	if(next!=pid){
	  j=0;
	  for(i=0;i<8;i++){
		//send ac aq hashed
		//send 1/8 of array (8 send / receives to finish)
	    MPI_Isend(&sendq[j], nqi/8+nqi/32, pointtype, next, tagq, MPI_COMM_WORLD, &reqsq[i]);
	    
		MPI_Isend(&sendc[j], nci/8+nci/32, pointtype, next, tagc, MPI_COMM_WORLD, &reqsc[i]);
        //1/8 was calculated through trial and error to provide stability and error tolerance
		j=j+nqi/8;
	  }
	}
    if(previous!=pid){
      j=0;
	  for(i=0;i<8;i++){
        //receive ac aq  hashed
        MPI_Irecv(&recq[j], nqi/8+nqi/32, pointtype, previous, tagq, MPI_COMM_WORLD, &reqq[i]);//Omoiws me ta sxolia parapanw sta send oson afora stous count
        MPI_Irecv(&recc[j], nci/8+nci/32, pointtype, previous, tagc, MPI_COMM_WORLD, &reqc[i]);
        j=j+nci/8;
	  }
	}
	//insert q,c point to aaq and aac
    keep(&aq[0],1);
    
    keep(&ac[0],2);
    
    //wait loops
    
    if(previous!=pid){
      j=0;
      for(i=0;i<8;i++){
        //wait for q
        MPI_Wait (&(reqsq[i]),&(statsq[i]));//wait send q
        MPI_Wait (&(reqq[i]),&(statq[i]));//wait receive q
        //copy receive from buffer
        equate(&aq[j],&recq[j],nqi/8);
        j=j+nqi/8;
	  }
	  
      j=0;
      for(i=0;i<8;i++){
  	    //wait for c
  	    MPI_Wait (&(reqsc[i]),&(statsc[i]));//wait send c
        MPI_Wait (&(reqc[i]),&(statc[i]));//wait receive c
		//copy receive from buffer
        equate(&ac[j],&recc[j],nci/8);
		j=j+nci/8;
	  }
	}
  }
  
  free(aq);
  free(ac);
  free(sendq);
  free(sendc);
  free(recq);
  free(recc);
}

/** 
Keeps in the aaq and in the aac the aq and the ac array data, that the current process gets involved
**/
void keep(point *a, int job){
  int i,f,j;
  if(job==1){
  	for(i=0;i<nqi;i++){
      //sort q to their neighbors' arrays
      f=a[i].kox+n*a[i].koy+m*n*a[i].koz;
      if((f>=pid*vi)&&(f<=pid*vi+vi-1)){
      	//f of the process
      	f=f-pid*vi;
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
        //count in position
		countq[f]=countq[f]+1;
	  }
    }
  }
  else{
    for(i=0;i<nci;i++){
      //same for c
      f=a[i].kox+n*a[i].koy+m*n*a[i].koz;
      if((f>=pid*vi)&&(f<=pid*vi+vi-1)){
      	//f of the process
      	f=f-pid*vi;
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
        //count in position
        countc[f]=countc[f]+1;
      }
    }
  }
}

/**
Receives 4 indices qi, qj, ci and cj that point to a q and a c point and returns the float distance these two points
**/
double dest(int fq,int jq,int fc,int jc){
  double d,dx,dy,dz;
  dx=aaq[fq][jq].x-aac[fc][jc].x;
  dy=aaq[fq][jq].y-aac[fc][jc].y;
  dz=aaq[fq][jq].z-aac[fc][jc].z;
  d=sqrt(dx*dx+dy*dy+dz*dz);
  return d;
}

/**
Receives a box and an array pointer and returns the array filled with the f references of the neighbor boxes.
It works with 26 boxes per time and if a box is out range the f reference is filled by -1.
**/
void neib(int fs,int *neibs){
  int tx,ty,tz,ftemp,ptemp,f,i=0,tkox,tkoy,tkoz,skox,skoy,skoz;
  //absolute f
  f=fs+pid*vi;
  if(fs<vi&&fs>=0&&countq[fs]!=0){
  	skox=aaq[fs][0].kox;
  	skoy=aaq[fs][0].koy;
  	skoz=aaq[fs][0].koz;
  }
  else if(countq[fs]==0){
    printf("!!!!!!!!!!!!!!!!ALARM!!!!!!!!!!!! (empty box error)\n");
  	exit(1);
  }
  else{
  	printf("!!!!!!!!!!!!!!!!ALARM!!!!!!!!!!!! (wrong fs neibs input error)\n");
  	exit(1);
  }
  for(tx=-1;tx<2;tx++){
  	for(ty=-1;ty<2;ty++){
  	  for(tz=-1;tz<2;tz++){
  	  	tkox=skox+tx;
  	  	tkoy=skoy+ty;
  	  	tkoz=skoz+tz;
		if(tkox>=0&&tkox<n&&tkoy>=0&&tkoy<m&&tkoz>=0&&tkoz<k){
   	      ftemp=tkox+n*tkoy+m*n*tkoz;
		  if(ftemp!=f){
  	        neibs[i]=ftemp;
  	        i=i+1;
		  }
	    }
	    else{
	      neibs[i]=-1;
	      i=i+1;
		}
	  }
	}
  }
}

/**
Receives 2 array pointers and equates the second with first, element by element
**/
void equate(point *a,point *b,int size){
  int i;
  for(i=0;i<size;i++){
        a[i].idp=b[i].idp;
        a[i].x=b[i].x;
        a[i].y=b[i].y;
        a[i].z=b[i].z;
        a[i].kox=b[i].kox;
        a[i].koy=b[i].koy;
        a[i].koz=b[i].koz;
  }
}


/**
Receives the index of a box that a q point belongs to and the full index that q point. 
It also receives the index of another box.
It calculates the distances of q to the c of the second box and returns the index of 
the min distance c
**/
int findmin(int fq,int jq,int fc){
  int i,mini=-1;
  double min=20,temp;//distance must be <20 by definition
  for(i=0;i<countc[fc];i++){
    temp=dest(fq,jq,fc,i);
	if(temp<min){
	  min=temp;
	  mini=i;
    }
  }
  return mini;
}

/**
It makes all the comunications between processes, required for calculating 
distances between q and c points from different processes.
It also finds the nearest c for every q and stores it in the min array.
**/
void minq(){
  int i,j,f,fs,neibs[26],fcount,fdest,forig,pneib,***reccou,cou[2],tempmini,**checksend,**checkrec,frec,irec;
  point ***rec;
  double tempmin,currentmin;
  //MPI variables for comunication
  MPI_Request **reqs,**req,**coureqs,**coureq;
  MPI_Status **stats,**stat,**coustats,**coustat;
  //memory allocation for mpi request/status variables
  coureq= (MPI_Request **) malloc(vi * sizeof(MPI_Request *));
  coureqs= (MPI_Request **) malloc(vi * sizeof(MPI_Request *));
  req= (MPI_Request **) malloc(vi * sizeof(MPI_Request *));
  reqs= (MPI_Request **) malloc(vi * sizeof(MPI_Request *));
  stats= (MPI_Status **) malloc(vi * sizeof(MPI_Status *));
  stat= (MPI_Status **) malloc(vi * sizeof(MPI_Status *));
  coustats= (MPI_Status **) malloc(vi * sizeof(MPI_Status *));
  coustat= (MPI_Status **) malloc(vi * sizeof(MPI_Status *));
  for(i=0;i<vi;i++){
  	coureqs[i]=(MPI_Request *) malloc(26 * sizeof(MPI_Request));
  	coureq[i]=(MPI_Request *) malloc(26 * sizeof(MPI_Request));
    req[i]=(MPI_Request *) malloc(26 * sizeof(MPI_Request));
    reqs[i]=(MPI_Request *) malloc(26 * sizeof(MPI_Request));
    stats[i]=(MPI_Status *) malloc(26 * sizeof(MPI_Status));
    stat[i]=(MPI_Status *) malloc(26 * sizeof(MPI_Status));
    coustats[i]=(MPI_Status *) malloc(26 * sizeof(MPI_Status));
    coustat[i]=(MPI_Status *) malloc(26 * sizeof(MPI_Status));
  }
  
  //memory allocation for 3D arrays to receive
  rec= (point ***) malloc(vi * sizeof(point **));
  reccou = (int ***)malloc(vi*sizeof(int**));
  for (i = 0; i< vi; i++) {
  	rec[i]= (point **) malloc(26 * sizeof(point *));
    reccou[i] = (int **) malloc(26*sizeof(int *));
    for (j = 0; j < 26; j++) {
      reccou[i][j] = (int *)malloc(2*sizeof(int));
    }
  }
  
  //memory allocation and initiliazation for check arrays
  checkrec= (int **) malloc(v * sizeof(int *));
  checksend= (int **)malloc(p * sizeof(int*));
  for(i=0;i<v;i++){
	checkrec[i]= (int *)malloc(2*sizeof(int));
	checkrec[i][0]=-1;
    checkrec[i][1]=-1;
  }
  for(i=0;i<p;i++){
    checksend[i]= (int *)malloc(vi*sizeof(int));
    for(j=0;j<vi;j++)
	  checksend[i][j]=-1;
  }
 
  //f loop
  for(fs=0;fs<vi;fs++){
    //relative f=fs
    //absolute f=f
    f=fs+pid*vi;
    //create neighbor array
    neib(fs,&neibs[0]);
    for(i=0;i<26;i++){
	  pneib=getpid(neibs[i]);
      //check if neib's f is in our process and if neib is ok
  	  if(pneib!=pid&&neibs[i]!=-1){
        //cou[i] contains relative f we as from the neib
		cou[0]=neibs[i]-pneib*vi;
	    cou[1]=countc[fs];
	    
        //send cou
        MPI_Isend(&cou[0], 2, MPI_INT, pneib, f, MPI_COMM_WORLD, &coureqs[fs][i]);
        //receive reccou[i]
        
        MPI_Irecv(&reccou[fs][i][0], 2, MPI_INT, pneib, neibs[i], MPI_COMM_WORLD, &coureq[fs][i]);
	  }//if neib does not exist
	  else if(neibs[i]==-1){
	    reccou[fs][i][0]=-1;
	    reccou[fs][i][1]=-1;
	  }//if f belongs to curent process
	  else{
	    reccou[fs][i][0]=neibs[i]-pid*vi;
	    reccou[fs][i][1]=-10;//flag for neib in our process (no comunication needed)
	  }
    }
  }
  //before waitng for receiving reccou find min for our box
  for(fs=0;fs<vi;fs++){
  	for(i=0;i<countq[fs];i++){
  	  min[fs][i][0]=fs;
	  min[fs][i][1]=findmin(fs,i,fs);
	  min[fs][i][2]=-1;
	  
	}
  }
  //wait for sending
  for(fs=0;fs<vi;fs++){
  	neib(fs,&neibs[0]);
    for(i=0;i<26;i++){
      pneib=getpid(neibs[i]);
      //wait for sending cou
      if(pneib!=pid&&neibs[i]!=-1)
  	    MPI_Wait (&coureqs[fs][i],&coustats[fs][i]);
	}
  }
  
  //wait for receiving
  for(fs=0;fs<vi;fs++){
  	neib(fs,&neibs[0]);
    for(i=0;i<26;i++){
      pneib=getpid(neibs[i]);
      //wait for receiving reccou
      if(pneib!=pid&&neibs[i]!=-1){
  	    MPI_Wait (&coureq[fs][i],&coustat[fs][i]);
	  }
	}
  }

  //we follow the same order, so every rec[fs][i][..] referes to a reccou[fs][i][..]
  for(fs=0;fs<vi;fs++){
  	//relative f=fs
    //absolute f=f
    f=fs+pid*vi;
	//create neighbor array	
    neib(fs,&neibs[0]);
    for(i=0;i<26;i++){
      pneib=getpid(neibs[i]);
      //if neib exists and neib dont belong to our process
   	  if(pneib!=pid&&neibs[i]!=-1){
        
		forig=neibs[i];				//absolute f of neib box
        fcount=reccou[fs][i][1];	//count receives
        fdest=reccou[fs][i][0];		//id of the box for our process that neibs with neib[i] box
        
        //check if aac[fs] line sent before
        if(checksend[pneib][fs]==-1){
        	
	      //send here aaq[fs] whole
          MPI_Isend(&aac[fs][0], countc[fs], pointtype, pneib, f, MPI_COMM_WORLD, &reqs[fs][i]);
          //set flag
		  checksend[pneib][fs]=1;
    	}
        
        //check if aac line not yet sent by pneib process
        if(checkrec[forig][0]==-1){
          //making array with sent count
          rec[fs][i]= (point *) malloc((fcount) * sizeof(point));
		  
		  //receive here c
          MPI_Irecv(&rec[fs][i][0], fcount+vi, pointtype, pneib, neibs[i], MPI_COMM_WORLD, &req[fs][i]);//opws kai parapanw an evaza swsto count eixa provlima (trucated message i invalid buffer i pernage midenika) wstoso oxi panta
          //set flag
          checkrec[forig][0]=fs;
          checkrec[forig][1]=i;
		}
		else{
		   rec[fs][i]=NULL;
		}
        
      }//if f belongs to curent process
      else if(pneib==pid){
        forig=neibs[i]-pid*vi;		//relative f of neib
        fcount=countc[forig];		//count neib's elements
        fdest=reccou[fs][i][0];		//box to check its q
        //check c from our process neib box
        for(j=0;j<countq[fdest];j++){
          tempmini=findmin(fdest,j,forig);
          tempmin=dest(fdest,j,forig,tempmini);
          currentmin=dest(fdest,j,min[fdest][j][0],min[fdest][j][1]);//current min distance
		  if(tempmin<currentmin){
		    min[fdest][j][0]=forig;
		    min[fdest][j][1]=tempmini;
		  }
        }
        //fix rec without send-receive
        rec[fs][i]= NULL;
	  }//if neib does not exists
	  else{
      //nop
      rec[fs][i]=NULL;
	  }
    }
  }
 
  //wait for sending
  for(fs=0;fs<vi;fs++){
  	neib(fs,&neibs[0]);
  	for(i=0;i<26;i++){
      pneib=getpid(neibs[i]);
      //if neib exists and neib dont belong to our process
      if(pneib!=pid&&neibs[i]!=-1){
      	//if send real made
        if(checksend[pneib][fs]==1){
          MPI_Wait (&reqs[fs][i],&stats[fs][i]);
          checksend[pneib][fs]=2;//for not checking second time the same send
		}
	  }
	}
  }
  
  //some send-receive haven't finished due to non exists and belongs to process neib etc...
  //wait for receiving and finding min when received
  for(fs=0;fs<vi;fs++){
  	neib(fs,&neibs[0]);
    for(i=0;i<26;i++){
      pneib=getpid(neibs[i]);
      //if neib exists and neib dont belong to our process
      if(pneib!=pid&&neibs[i]!=-1){
      	forig=neibs[i];
      	//if we real have to wait for aac line
        if(checkrec[forig][0]==fs&&checkrec[forig][1]==i){
		  MPI_Wait (&req[fs][i],&stat[fs][i]);
 		  //call routine for last min
	      minc(fs,i,rec,reccou);
  		}//else if aac line arrived before
	    else{
	      //call routine for last min non received (before received)
	      minc(checkrec[forig][0],checkrec[forig][1],rec,reccou);
		}
	    //call routine for last min
	  }
    }
  }
  //free every local pointer from memory 
  free(rec);
  free(reccou);
  free(coureq);
  free(coureqs);
  free(stat);
  free(stats);
  free(coustats);
  free(coustat);
  free(reqs);
  free(req);
  free(checksend);
  free(checkrec);
}

/**
receives an absolute f of a box and returns the pid of the process that the box belongs to
**/
int getpid(int f){
  int nn;
  if(f>=0&&f<v)
    nn=(int)f/vi;
  else
    nn=-1;
  return nn;
}

/**
Tests if the distances between q and the boxes are minimums
and forwards the nearest c of the box for each q.
**/
void minc(int fs,int irec,point ***rec,int ***reccou){
  int i,j,fdest,fcount,ilast,flast,jlast;
  double last,temp,dx,dy,dz;
  if(reccou[fs][irec][1]>0){
  	fdest=reccou[fs][irec][0];
    fcount=reccou[fs][irec][1];
  	//if received before, fs and irec was fixed by the call of the minc
	for(i=0;i<countq[fdest];i++){
      //if last min was in our process
      if(min[fdest][i][2]==-1){
        last=dest(fdest,i,min[fdest][i][0],min[fdest][i][1]);
	  }//if last was in neibs process
	  else{
	    flast=min[fdest][i][0];
	    ilast=min[fdest][i][1];
	    jlast=min[fdest][i][2];
	    dx=aaq[fdest][i].x-rec[flast][ilast][jlast].x;
	    dy=aaq[fdest][i].y-rec[flast][ilast][jlast].y;
	    dz=aaq[fdest][i].z-rec[flast][ilast][jlast].z;
	    last=sqrt(dx*dx+dy*dy+dz*dz);
	  }

      for(j=0;j<fcount;j++){
        //find distance aaq[fdest][i]-----rec[fs][irec][j]
      	dx=aaq[fdest][i].x-rec[fs][irec][j].x;
	    dy=aaq[fdest][i].y-rec[fs][irec][j].y;
	    dz=aaq[fdest][i].z-rec[fs][irec][j].z;
      	temp=sqrt(dx*dx+dy*dy+dz*dz);
      	if(temp<last){
      	  last=temp;
		  min[fdest][i][0]=fs;
		  min[fdest][i][1]=irec;
		  min[fdest][i][2]=j;
		}
	  }
	}
  }
}
