#include <pthread.h>

#include <vector>
#include <iostream>
#include <ff/farm.hpp>


#include <pthread.h>

#include <vector>
#include <iostream>
#include <ff/farm.hpp>

using namespace ff;

// a task requires to compute the matrix multiply C = A x B
// we assume square matrixes, for the sake of simplicity
typedef struct {
  int n;
  float **a;
  float **b;
  float **c;
  int tag;    // used to gather partial results from the same task
} TASK;

// a subtask is the computation of the inner product or A, row i, by B, col j
typedef struct {
  int i,j;
  TASK * t;
} SUBTASK;

// a partial result is the i,j item in the result matrix
typedef struct {
  int i,j;
  float x;
  TASK * t;
} PART_RESULT;




// this node is used to generate the task list out of the initial data
// kind of user defined iterator over tasks
class Split: public  ff_node {
    void * svc(void * t) {
	TASK * task = (TASK *) t;     // tasks come in already allocated
	for(int i=0; i<task->n; i++)
	    for(int j=0; j< task->n; j++)  {
		// SUBTASKe are allocated in the splitter and destroyed in the worker
		SUBTASK * st = (SUBTASK *) calloc(1,sizeof(SUBTASK));
		st->i = i;
		st->j = j;
		st->t = task;
		ff_send_out((void *)st);
	    }
	return GO_ON;
    }
};

// this node is used to consolidate the subresults into the result matrix
#define MAXDIFF 16

class Compose: public ff_node {
public:
   Compose() {
     n = 0;
     for(int i=0; i<MAXDIFF; i++)
       tags[i] = 0;
   }

   void * svc(void * t) {
     PART_RESULT * r = (PART_RESULT *) t;
     TASK * tt = r->t;
     // consolidate result in memory
     ((r->t)->c)[r->i][r->j] = r->x;
     tags[((r->t)->tag)%MAXDIFF]++;
     if(tags[((r->t)->tag)%MAXDIFF] == ((r->t)->n)*((r->t)->n)) {
       tags[((r->t)->tag)%MAXDIFF]  = 0;
       free(t);
       return(tt);
     } else {
       free(t);
       return GO_ON;
     }
   }
private:
  int n;
  int tags[MAXDIFF];
};


// this is the node actually computing the task (IP)
class Worker: public ff_node {
public:
    
    void * svc(void * task) {
	SUBTASK * t = (SUBTASK *) task;
        float x=0.0;
	for(int k=0; k<(t->t)->n; k++) {
	    x += (t->t->a)[t->i][k] * (t->t->b)[k][t->j];
        }
        // prepare the partial result to be delivered
        PART_RESULT * pr = (PART_RESULT *) calloc(1,sizeof(PART_RESULT));
        pr->i = t->i;
        pr->j = t->j;
        pr->t = t->t;
        pr->x = x;
        // the subtask is no more useful, deallocate it
        free(task);
        // return the partial result
        return pr;
    }
};

int main(int argc, char * argv[]) {
    
    if (argc<4) {
        std::cerr << "use: " << argv[0] << " nworkers n m\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    int n=atoi(argv[2]);
    int m=atoi(argv[3]);
    
    // this is the map setup code ----------------------------------------------
    ff_farm<> farm(true);

    farm.add_emitter(new Split());    // add the splitter emitter
    farm.add_collector(new Compose());  // add the composer collector
    std::vector<ff_node *> w;           // add the convenient # of workers
    for(int i=0;i<nworkers;++i)
    w.push_back(new Worker);
    farm.add_workers(w);
    farm.run_then_freeze();             // run it as an accelerator
    // end of map setup --------------------------------------------------------

    // now we can test it by offloading a task
    srand(getpid());

    // output m tasks
    for(int task=0; task<m; task++) {
      TASK * t1 = (TASK *) calloc(1,sizeof(TASK));

      float ** A = new float*[n];
      float ** B = new float*[n];
      float ** C = new float*[n];
      for(int i=0; i<n; i++) {
        A[i] = new float[n];
        B[i] = new float[n];
        C[i] = new float[n];
      }

      for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
          A[i][j] = rand() % 10 - 5.0;
          B[i][j] = rand() % 10 - 5.0;
          C[i][j] = 0.0;
        }
      }
      t1->a = A;
      t1->b = B;
      t1->c = C;
      t1->n = n;
      t1->tag = task;

      farm.offload(t1);
    }
    farm.offload((void *) FF_EOS);
    farm.wait();


    std::cerr << "Nw = " << nworkers << " n= " << n << " m= " << m << " msecs: " << farm.ffTime() << std::endl;

/*
    float C1[n][n];
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
        C1[i][j] = 0.0;
        for(int k=0; k<n; k++)
          C1[i][j] += A[i][k]*B[k][j];
        if(C1[i][j] != C[i][j]) {
       std::cerr << "Error at " << i << j << std::endl;
        }
      }

    }
*/

    return 0;
} 
