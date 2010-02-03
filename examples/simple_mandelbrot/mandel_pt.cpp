#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <queue>
#include <sys/time.h>
#include <math.h>
#include "marX2.h"


#define DIM 800
#define ITERATION 1024

using namespace std;

#define LOCK_INIT(l)      pthread_mutex_init(l,NULL)
#define LOCK(l)           pthread_mutex_lock(l)
#define UNLOCK(l)         pthread_mutex_unlock(l)
#define COND_INIT(c)      pthread_cond_init(c,NULL)
#define COND_SIGNAL(c)    pthread_cond_signal(c)
#define COND_BROADCAST(c) pthread_cond_broadcast(c)
#define COND_WAIT(c,l)    pthread_cond_wait(c,l)


inline  double diffmsec(struct timeval  a,  struct timeval  b) {
  long sec  = (a.tv_sec  - b.tv_sec);
  long usec = (a.tv_usec - b.tv_usec);
    
  if(usec < 0) {
	--sec;
	usec += 1000000;
  }
  return ((double)(sec*1000)+ (double)usec/1000.0);
}



class mpmc {

public:
  mpmc(const int len) {
	qlen = len;
	ns = 1;
	LOCK_INIT(&mux);
	COND_INIT(&cv);
  };

  mpmc(const int len, const int nsources) {
	qlen = len;
	ns = nsources;
	LOCK_INIT(&mux);
	COND_INIT(&cv);
  };
  

  void push (void * item) {
	LOCK(&mux);
	if (queue.empty()) {
	  // COND_BROADCAST(&cv);
	  // In this particolar example it never happens that many threads are waiting 
	  // on a queue 
	  COND_SIGNAL(&cv);
	  //cout << "Producer signal on empty queue " << ns << endl; 
	  // fflush(stdout);
	}
	queue.push(item);
	UNLOCK(&mux);
  }

  void * pop() {
	void * tmp;
	LOCK(&mux);
	while( queue.empty() ) {
      //cout << "Consumer stops on empty queue: " << ns << endl;  
	  //fflush(stdout);
	  COND_WAIT(&cv, &mux);
    }
	tmp = queue.front();
	queue.pop();
	UNLOCK(&mux);
	return (tmp);
  }

private:
  int qlen;
  std::queue<void *>queue;
  pthread_mutex_t mux;
  pthread_cond_t cv;
  int ns;
};

typedef struct p {
  mpmc * in;
  mpmc * out;
  int name;
} param_t;

typedef struct outitem {
  unsigned char * M;
  int line;
} ostream_t;

const double range=3.0;
const double init_a=-2.125,init_b=-1.5;

double step = range/((double) DIM);
int dim = DIM;
int niter = ITERATION;

static void * worker(void * arg) {
  param_t * q = (param_t *) arg;
  int * v,t;
  while (1) {
	v = (int *) q->in->pop();
	if (v==NULL) {
	  //free(v);
	  // cout << "Sending EOS\n";
	  q->out->push(NULL);
	  break;
	}
	else {
	  t = *v;
	  free(v);
	  // Work HERE --------------
	  // int * t = v;
	  ostream_t * oi = new ostream_t( );
	  oi->M = (unsigned char *) malloc(dim*sizeof(char));
	  // this is just a cut&paste of the sequential algorithim 
	  //cout << "Worker got a task " << t << " \n";
	  //fflush(stdout);
	  {
		// this is just alpha renamining
		int i = 	oi->line = t;
		//
		int j,k;
		double im,a,b,a2,b2,cr;
		
		im=init_b+(step*i);
		for (j=0;j<dim;j++)   {         
		  a=cr=init_a+step*j;
		  b=im;
		  k=0;
		  for (k=0;k<niter;k++)
			{
			  a2=a*a;
			  b2=b*b;
			  if ((a2+b2)>4.0) break;
			  b=2*a*b+im;
			  a=a2-b2+cr;
			}
		  oi->M[j] =  (unsigned char) 255-((k*255/niter)); 
		  //printf("%3d ",	oi->M[j] );
		} 
		//printf("\n");
	  }
	  // ---------------------
	  //cout << "Worker send a task\n";
	  q->out->push(oi);
	}
  }
  return NULL;
}

static void * collector(void * arg) {
  param_t * q = (param_t *) arg;
  int eos = 0;
  while (1) {
	ostream_t * t;
	t = (ostream_t *) q->in->pop();
	//cout << "Collector got a task " << t->line << " \n";
	if (t==NULL) { 
	  //cout << "Got " << eos+1 << " eos" << endl;
	  if (++eos == q->name) {
		//	cout << "DONE" << endl;
		break;
	  }
	}
	else {
	  // COLLECTOR WORK HERE
#if !defined(NO_DISPLAY)
	  ShowLine(t->M,dim,t->line); 
#endif
	  // cout << "task received" << endl;
	  free(t->M);
	  free(t);
	  // -----------------
	}
  }
  return NULL;
}


int main (int argc, char ** argv) {
  int workers = 2;
  pthread_t * children;
  param_t * pars;
  mpmc * gather_ch;
  //
  int r,retries=1;
  double avg=0, var, * runs;
  struct timeval t1,t2;


  if (argc<5) {
	printf("Usage: mandel_pt size niterations retries nworkers\n\n\n");
  }
  else {
	dim = atoi(argv[1]);
	niter = atoi(argv[2]);
	step = range/((double) dim);
	retries = atoi(argv[3]);
	workers = atoi(argv[4]);
  }
  
  runs = (double *) malloc(retries*sizeof(double));

  printf("Mandebroot set from (%g+I %g) to (%g+I %g)\n",
		 init_a,init_b,init_a+range,init_b+range);
  printf("resolution %d pixel, Max. n. of iterations %d - Using %d workers\n",dim*dim,niter,workers);	

#if !defined(NO_DISPLAY)
  SetupXWindows(dim,dim,1,NULL,"FF Mandelbroot");
#endif  


  for (r=0;r<retries;r++) {

  children = new pthread_t[workers+1];
  pars = new param_t [workers+1]; 
  gather_ch = new mpmc(10,-5);
  // workers
  for(int i=0;i<workers;++i) {
	pars[i].in = new mpmc(10,i);
	pars[i].out = gather_ch;
	pars[i].name = i;
	if (pthread_create(&children[i],NULL,&worker,&pars[i]) != 0) {
	  perror("error during thread creation");
	  exit(1);
	} 
  }             

  pars[workers].in = gather_ch;
  pars[workers].out = NULL;
  pars[workers].name = workers;
   // collector 
  
  
  if (pthread_create(&children[workers],NULL,&collector,&pars[workers]) != 0) {
	perror("error during thread creation");
	exit(1);
  } 

  // Start time
  gettimeofday(&t1,NULL);
  

  int nextw = 0;
  for(int i=0;i<dim;++i) {
	int * p = new int(i);
	pars[nextw].in->push(p);
	nextw = (nextw+1) % workers;
  }
  for(int k=0;k<workers;++k) {
	pars[k].in->push(NULL);
  }
  
  for(int i=0;i<workers+1;++i) 
	pthread_join(children[i],NULL);
 
  // Stop time
  gettimeofday(&t2,NULL);
	
  avg += runs[r] = diffmsec(t2,t1);
  printf("Run [%d] DONE, time= %f (ms)\n",r, runs[r]);

  // free channels
  for(int i=0;i<workers;++i) {
	free(pars[i].in);
  }
  free(gather_ch);
  free(pars);
  free(children);
  }
   // stats
  avg = avg / (double) retries;
  var = 0;
  for (r=0;r<retries;r++) {
	var += (runs[r] - avg) * (runs[r] - avg);
  }
  var /= retries;
  std::cerr << "Average on " << retries << " experiments = " << avg << " (ms) Std. Dev. " << sqrt(var) << "\n\nPress a key\n" << std::endl;
  getchar();
  
#if !defined(NO_DISPLAY)
  CloseXWindows();
#endif

  return 0;
}	  
