#include <string>
#include <iostream>
#include <ff/ff.hpp>
#include <ff/mapper.hpp>
using namespace ff;


static long WITER = 2400;
static float comptime=0.0;

struct firstStage: public ff_node_t<unsigned long> {

  int svc_init() {
    sleep(1);
    printf("starting\n");
    ::ffTime(START_TIME);
    return 0;
  }
  firstStage(unsigned long nmsgs):nmsgs(nmsgs),counter(0) {}
    unsigned long *svc(unsigned long *task) {
#if defined(BEST_CASE) 
	// best case test  (queue always full)
	// in this case the cache line is full and we 
	// obtains the best performance
        if (task==NULL) {
	  for(unsigned long i=0;i<nmsgs;++i)
	    if (!ff_send_out((void*)(i+1))) abort();	    
	  ff_send_out(EOS);
	}
	return GO_ON;
#else  
	// worst case test (queue always empty)
	// in this case there is only one single element
	// per cache line.

	// simulate some works
	ticks_wait(WITER);
	if (task == NULL) {
	  if (!ff_send_out((void*)++counter)) abort();
	  return GO_ON;
        }
	if (++counter>=nmsgs) return EOS;
        return task;
#endif
    };
  void svc_end() {
      ::ffTime(STOP_TIME);
      comptime = ::ffTime(GET_TIME);
#if defined(BEST_CASE)
      printf("Time: %g (ms)  Avg Latency: %f (ns)\n", ::ffTime(GET_TIME),(1000000.0*::ffTime(GET_TIME))/(nmsgs*2));    
#endif
  }
    unsigned long nmsgs;
    unsigned long counter;
};


struct lastStage: ff_node_t<unsigned long> {
    unsigned long *svc(unsigned long *task) {  
#if !defined(BEST_CASE)
	// simulate some works
      ticks_wait(WITER);
#endif
	return task; 
    }
};


void usage(char * name) {
    std::cerr << "usage: \n";
    std::cerr << "      " << name << " [num-messages \"#P,#C\"]\n";
}

int main(int argc, char *argv[]) {
#if defined(FF_BOUNDED_BUFFER)
    std::cerr << "This test requires unbounded buffers\n";
    return 0;
#endif
    unsigned long nmsgs=1000000;
    std::string worker_mapping("0,1");

    if (argc>3 && argc != 4) { 
	usage(argv[0]);
	return -1;
    } 
    if (argc>=2) nmsgs  = atol(argv[1]);
    if (argc>=3) worker_mapping = std::string(argv[2]);
    if (argc==4) WITER = atol(argv[3]);

    std::cerr << "cmd: " << argv[0] << " " << nmsgs << " " << worker_mapping << " " << WITER << "\n";
    
    threadMapper::instance()->setMappingList(worker_mapping.c_str());

    ff_pipeline pipe(false, 32, 32, true);
    //ff_pipeline pipe(false, 8192, 8192, true);
    pipe.add_stage(new firstStage(nmsgs));
    pipe.add_stage(new lastStage);
    pipe.wrap_around();
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    float overhead = 0.0;
   
#if !defined(BEST_CASE)
    ::ffTime(START_TIME);
    for(unsigned long i=0;i<nmsgs;++i) {
	ticks_wait(WITER);
	ticks_wait(WITER);
	//for(volatile long i=0;i<WITER;++i);
	//for(volatile long i=0;i<WITER;++i);
    }
    ::ffTime(STOP_TIME);
    //printf("Overhead: %f (ns)  per msg= %f (ns)\n", ::ffTime(GET_TIME),(1000000.0*::ffTime(GET_TIME))/(nmsgs*2));    
    printf("Ticks=%ld (~%f (us))\n", WITER, (WITER / (1.0*(ff_getCpuFreq()/1000000.0))));
    overhead = ::ffTime(GET_TIME);
    printf("comptime =%f (ms) overhead =%f (ms)\n", comptime, overhead);
#endif
    printf("Latency= %.3f (ns)\n", (1000000.0*(comptime-overhead))/(nmsgs*2));

    pipe.ffStats(std::cout);
	   
    return 0;
}
