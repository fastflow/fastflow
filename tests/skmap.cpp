#include <pthread.h>

#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>

using namespace ff;

// a task requires to compute the matrix multiply C = A x B 
// we assume square matrixes, for the sake of simplicity
typedef struct {
  int n;
  float **a;
  float **b;
  float **c;
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

class SeqMM: public ff_node {
  void * svc(void * task) {
    TASK * t = (TASK *)task;
    
    for(int i=0; i< (t->n); i++) {
      for(int j=0; j< (t->n); j++) {
        (t->c)[i][j]=0;
        for(int k=0; k< (t->n); k++) 
          (t->c)[i][j] = (t->c)[i][j] + (t->a)[i][k]*(t->b)[k][j];
      }
    }
    return GO_ON;
  }
};

// this node is used to generate the task list out of the initial data
// kind of user defined iterator over tasks
class Split: public  ff_node {
    int svc_init() {
	std::cout << "Split svc started" << std::endl;
	return 0;
    }
   void * svc(void * t) {
     std::cout << "Split svc got a task" << std::endl;
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
class Compose: public ff_node {
    int svc_init() {
	std::cout << "Compose svc started" << std::endl;
	return 0;
    }
   void * svc(void * t) {
     PART_RESULT * r = (PART_RESULT *) t;
     ((r->t)->c)[r->i][r->j] = r->x; 
     // only for printing the result
     t = r->t;
     // deallocate the partial result got on the input stream
     //free(t);
     return GO_ON;
   }


   void svc_end() {
     std::cout << "Result matrix" << std::endl;
     for(int i=0;i<t->n;i++)  {
       for(int j=0;j<t->n;j++)
         std::cout << (t->c)[i][j] << " ";
       std::cout << std::endl;
     }
   }

private: 
   TASK * t;
};


// this is the node actually computing the task (IP)
class Worker: public ff_node {
public:
    
    void * svc(void * task) {
        std::cout << "Worker svc started" << std::endl;
	SUBTASK * t = (SUBTASK *) task;
        float  x = 0.0;
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
        //free(task);
        // return the partial result
        return pr;
    }
};

class ff_map {
public:

  // map constructor
  // takes as parameters: the splitter, the string of workers and the result rebuilder
  ff_map(ff_node * splt, std::vector<ff_node *> wrks, ff_node * cmps) {

      exec.add_emitter(splt);	// add the splitter emitter
      exec.add_collector(cmps);   // add the composer collector
      exec.add_workers(wrks);     // add workers 
  }

    operator ff_node*() {
	return (ff_node*)&exec;
    }

private: 
  ff_farm<> exec;
};

class GenStream: public ff_node {
public:
  GenStream(int nn) {
    std::cout << "GenStream constructor" << std::endl;
    n = nn; 
    return; 
  }

  int svc_init() {
    std::cout << "GenStream svc_init" << std::endl;
    srand(getpid());
    return 0;
  }
    
  void * svc (void * task) {
    std::cout << "GenStream svc started" << std::endl;
    float ** A = new float*[n];
    for(int i=0; i<n; i++) 
      A[i] = new float[n]; 
    float ** B = new float*[n];
    for(int i=0; i<n; i++) 
      B[i] = new float[n]; 
    float ** C = new float*[n];
    for(int i=0; i<n; i++) 
      C[i] = new float[n]; 

    // std::cout << "A" << std::endl;
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
        A[i][j] = rand() % 10 - 5;
        // std::cout << A[i][j] << " " ;
      }
    // std::cout << std::endl;
    }

    // std::cout << "B" << std::endl;
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
        B[i][j] = rand() % 10 - 5;
        // std::cout << B[i][j] << " " ;
      }
    // std::cout << std::endl;
    }

    // std::cout << "C" << std::endl;
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
        C[i][j] = rand() % 10 - 5;
        // std::cout << C[i][j] << " " ;
      }
    // std::cout << std::endl;
    }

    TASK * t1 = (TASK *) calloc(1,sizeof(TASK));
    t1->a = A;
    t1->b = B; 
    t1->c = C; 
    t1->n = n; 
    
    ff_send_out(t1);
    return(NULL); 
  }
 
private: 
  int n; 
};

int main(int argc, char * argv[]) {

    if (argc<3) {
        std::cerr << "use: " 
                  << argv[0] 
                  << " nworkers n\n";
        return -1;
    }
    
    int nworkers=atoi(argv[1]);
    int n=atoi(argv[2]);
   
    std::vector<ff_node *> w;
    for(int i=0; i<nworkers; i++) 
      w.push_back(new Worker());

    ff_pipeline pipe;
    std::cout << "Created pipeline" << std::endl;
    pipe.add_stage(new GenStream(n));
    std::cout << "Created pipeline: added GenStream" << std::endl;
    ff_map mm(new Split(), w, new Compose()); 
    std::cout << "Created map" << std::endl;
    pipe.add_stage(mm); // this adds the parallel code

    //pipe.add_stage(new SeqMM()); // this adds the sequential code
    std::cout << "Created pipeline: added GenStream: added mm" << std::endl;

    std::cout << "Starting pipeline" << std::endl;
    pipe.run_and_wait_end();

    return 0;
}
