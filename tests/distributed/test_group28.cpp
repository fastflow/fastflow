/* 
 * FastFlow concurrent network:
 *
 * 
 *             -----------------------------------------|
 *            |            |--> Filter 1  | --> Sink 1  |
 *            |  Source1-->|              |             |
 *            |            |--> Filter 2  | --> Sink 2  |
 *            |  Source2-->|              |
 *             -----------------------------------------                            
 */


#include <map>
#include <mutex>
#include <iostream>
#include <ff/dff.hpp>

#if defined(ENABLE_MPI) || defined(DFF_MPI)
#include <mpi.h>
#endif

using namespace ff;
std::mutex mtx;   // used for pretty printing

struct Source : ff_monode_t<std::string>{
    Source(int id) : id(id) {}
	std::string* svc(std::string* in){
		int numWorkers = get_num_outchannels();
        for(int i = 0; i < numWorkers; i++)
            ff_send_out_to(new std::string("Task generated from " + std::to_string(id) + " for " + std::to_string(i)), i);
        return EOS;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "Source ENDED!\n";
    }
    
    int id;
};

struct Filter : ff_minode_t<std::string>{
    int filterID;
    Filter(int id) : filterID(id) {}
    std::string* svc(std::string* in){
        auto* out = new std::string(*in + " filtered by Filter " + std::to_string(filterID));
        delete in;
        return out;
    }
};

struct FilterOut : ff_monode_t<std::string>{
    std::string* svc(std::string* in){
        ff_send_out(in);
        return this->GO_ON;
    }

     void eosnotify(ssize_t i){
        //const std::lock_guard<std::mutex> lock(mtx);
        //ff::cout << "FilterOut: received an EOS from " << i << std::endl;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "Filter Out ENDED!\n";
    }
};


struct Sink : ff_minode_t<std::string>{
    int sinkID;
    size_t eos_received_ = 0;
    size_t tasks_received_ = 0;
    Sink(int id): sinkID(id) {}

    std::string* svc(std::string* in){
        ++tasks_received_;
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << *in << " received by Sink " << sinkID << " from " << get_channel_id() << "\n";
        delete in;
        return this->GO_ON;
    }

    void eosnotify(ssize_t i){
        ++eos_received_;
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "Sink [" << sinkID << "] received an EOS from " << i << std::endl;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "Sink ENDED!\n";
        // Each sink must receive one task and one EOS from every input channel.
        if (eos_received_ != this->get_num_inchannels()){
            ff::cout << "Number of EOS received wrong! Expected " << this->get_num_inchannels() << " received " << eos_received_ << "!Aborting!\n";
            abort();
        }
        if (tasks_received_ != this->get_num_inchannels()){
            ff::cout << "Number of tasks received wrong! Expected " << this->get_num_inchannels() << " received " << tasks_received_ << "!Aborting!\n";
            abort();
        }
        ff::cout << "RESULT OK\n";
    }
};


int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}

    ff_a2a a2a, a2a_inner;
    std::vector<Source*> firstSet;
    std::vector<ff_node*> secondSet;
    std::vector<Sink*> thirdSet;
    auto g1 = createGroup("G1");
    auto g2 = createGroup("G2");

    for(int i  = 0 ; i < 2; i++){
      auto s = new Source(i);
      firstSet.push_back(s);
      if (i == 0) g1 << s;
      else g2 << s;
    }
    
    for(int i = 0; i < 2; i++){
      auto s = new ff_comb(new Filter(i), new FilterOut, true, true);
      secondSet.push_back(s);
      if (i == 0) g1 << s;
      else g2 << s;
    }
    
    for(int i  = 0; i < 2; i++){
        auto s = new Sink(i);
        thirdSet.push_back(s);
         if (i == 0) g1 << s;
        else g2 << s;
    }
#if 1
    a2a.add_firstset(firstSet);
    a2a_inner.add_firstset(secondSet);
    a2a_inner.add_secondset(thirdSet);

    a2a.add_secondset<ff_node>({new ff_Pipe(a2a_inner)});
#else
    a2a_inner.add_firstset(firstSet);
    a2a_inner.add_secondset(secondSet);

    a2a.add_firstset<ff_node>({new ff_Pipe(a2a_inner)});
    a2a.add_secondset(thirdSet);
#endif

#if 0
	// for MPI debugging
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	std::fprintf(stderr, "Rank %d PID %d pronto per gdb\n", rank, getpid());
	std::fflush(stderr);
	volatile int wait_for_gdb = 1;
	while( wait_for_gdb) {
		sleep(1);
	}
#endif	
	
    if (a2a.run_and_wait_end()<0) {
		error("running pipe");
		return -1;
	}
	return 0;
}
