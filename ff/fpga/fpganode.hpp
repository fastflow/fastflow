#pragma once

#include <iostream>
#include <ff/ff.hpp>

#include "oggetto.hpp"
#include "task.hpp" 

#include "utimer.cpp"

using namespace std; 
using namespace ff;

class FNode : public ff::ff_node {
private:
  FTask task_description;
  FpgaDevice * d;
public:

  FNode(FTask task_description) : task_description(task_description) {}
  
  int svc_init(void) {
    utimer svcinit("SVC INIT");
    d = new FpgaDevice{};
    if(!d->DeviceExists()) 
      return(-1);
    std::cerr << "MDLOG: device found!" << std::endl;
    
    if(!d->CreateKernel("krnl_vadd.xclbin", "krnl_vadd"))
      return(-1);
    std::cerr << "MDLOG: kernel created!" << std::endl;
    
    if(!d->setupBuffers(task_description))
      return(-1);
    std::cerr << "MDLOG: bufferes created!" << std::endl;
    return(0);
  }

  void svc_end() {
    utimer svcend("SVC END");
    if(d->finalize())
      cout << "Errore nella finalize " << endl;
    else
      cout << "Finalizzazione terminata con successo "<< endl;

    d->cleanup();
    return;
  }
  
  void * svc (void * t) {


//XRT allocates memory space in 4K boundary for internal memory management. If the host memory pointer is not aligned to a page boundary, XRT performs extra memcpy to make it aligned.

    FTask * task = (FTask *) t;
    if(!d->computeTask(task))
      return(NULL);
    
    if(!d->waitTask())
      return(NULL);
    std::cerr << "MDLOG:  task computed!" << std::endl;

    return t; 
  }
};
