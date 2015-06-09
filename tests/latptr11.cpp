#include <string>
#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/mapper.hpp>
using namespace ff;


struct firstStage: public ff_node_t<unsigned long> {
  firstStage(unsigned long nmsgs):nmsgs(nmsgs),counter(0) {}
    unsigned long *svc(unsigned long *task) {
#if 1   // best case test  (queue always full)

        if (task==NULL) {
	  for(unsigned long i=0;i<nmsgs;++i)
	    if (!ff_send_out((void*)(i+1))) abort();	    
	  return EOS;
	}
	return GO_ON;
#else  // worst case test (queue always empty)

	if (task == NULL) {
	  if (!ff_send_out((void*)++counter)) abort();
	  return GO_ON;
        }
	if (++counter>=nmsgs) return EOS;
        return task;
#endif
    };
    unsigned long nmsgs;
    unsigned long counter;
};


struct lastStage: ff_node_t<unsigned long> {
    unsigned long *svc(unsigned long *task) {  return task; }
};


void usage(char * name) {
    std::cerr << "usage: \n";
    std::cerr << "      " << name << " [num-messages \"#P,#C\"]\n";
}

int main(int argc, char *argv[]) {
    unsigned long nmsgs=1000000;
    std::string worker_mapping("0,1");

    if (argc>2 && argc != 3) { 
	usage(argv[0]);
	return -1;
    } 
    if (argc>=2) nmsgs  = atol(argv[1]);
    if (argc==3) worker_mapping = std::string(argv[2]);
    
    std::cerr << "cmd: " << argv[0] << " " << nmsgs << " " << worker_mapping << "\n";
    
    threadMapper::instance()->setMappingList(worker_mapping.c_str());

    ff_pipeline pipe(false, 512, 512, true);
    pipe.add_stage(new firstStage(nmsgs));
    pipe.add_stage(new lastStage);
    pipe.wrap_around();
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    double time= pipe.ffTime();
    printf("Time: %g (ms)  Avg Latency: %f (ns)\n", time,(1000000.0*time)/(nmsgs*2));
    return 0;


}
