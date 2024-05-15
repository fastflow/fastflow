#include <iostream>
#include <vector>
#include <ff/ff.hpp>

#include "fnodetask.hpp"

using namespace std;
using namespace ff;

class generator : public ff_node_t<FTask> {
private: 
  int ntasks;
  int dimvec;
public: 
  generator(int ntasks, int dimvec):ntasks(ntasks), dimvec(dimvec) {}

  int svc_init() {
    std::cout << "generator started" << std::endl; 
    return 0; 
  }

  void svc_end() {
    std::cout << "generator ended" << std::endl; 
    return; 
  }

FTask * svc(FTask *) {
    // generate as many tasks as ntasks
    for(int i=0; i<ntasks; i++) {
      // first allocate a new FTask
      FTask * t = new FTask();
      // then allocate the vectors
      // ret |= posix_memalign((void **)&a, 4096, dimvec*sizeof(int));
      int * a = (int *) calloc(dimvec, sizeof(int));
      int * b = (int *) calloc(dimvec, sizeof(int));
      int * c = (int *) calloc(dimvec, sizeof(int));
      // and the scalar
      int * n = (int *) malloc(sizeof(int)); 

     
      // initialize the vectors
      for(int i=0; i<dimvec; i++) a[i] = b[i] = i; 
      // initialize the scalar
      *n = dimvec;
      // assign pointers to task cats
      t->add_input(a, dimvec*sizeof(int),0); 
      t->add_input(b, dimvec*sizeof(int),1); 
      t->add_output(c, dimvec*sizeof(int),2);
      t->add_scalar(n, sizeof(int));
      // task ready
      // send it
      ff_send_out((void *) t); 
      // some reporting
      // std::cout << "Generator sent task " << i  << std::endl;
    }
  return(EOS);
  }
};


class printer : public ff_node_t<FTask> {

  int svc_init() {
    std::cout << "printer started" << std::endl; 
    return 0; 
  }

  void svc_end() {
    std::cout << "printer ended" << std::endl; 
    return; 
  }

  FTask * svc(FTask * t) {
    std::cout << "Printer got task" << std::endl;
    if(t) {
      int * v = (int *) t->outputs[0].ptr;
      int n = *((int *) (t->scalars[0].ptr));

      for(int i=0; i<n; i++)
        cout << v[i] << ", ";
      cout << endl;
    }
    return(GO_ON);
  }
}; 

int main(int argc, char * argv[]) {

  cout << "XCL_MEM_TOPOLOGY " << std::hex << XCL_MEM_TOPOLOGY << endl; 
  cout << "CL_MEM_EXT_PTR_XILINX " << std::hex << CL_MEM_EXT_PTR_XILINX << endl; 
  string bitstream = std::string(argv[1]); 
  string kernelname = std::string(argv[2]); 
  int chain = atoi(argv[3]); 

  FDevice device(bitstream); 

  ff_pipeline prog;
  generator source(10,4);
  printer drain; 

  prog.add_stage(source);
  prog.add_stage(new FNodeTask(device, kernelname, chain));
  prog.add_stage(drain);

  prog.run_and_wait_end(); 
  return(0);
}





   
